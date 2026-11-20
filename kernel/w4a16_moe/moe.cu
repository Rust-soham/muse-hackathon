// Muse Spark MoE only: plan, norm-first gather, fused SwiGLU experts, weight-first combine.
// Layout: 32 experts, H=I=4096; packed NVFP4 code/FP8 scale blocks (4608 bytes).
// Grouped NVFP4 x BF16 expert GEMM for Muse Spark's MoE (one H200 = 32 local experts).
// out^T[w_rows, tokens] = W_e[w_rows, K] (NVFP4, dequantized in registers) x X^T[K, tokens] (BF16)
// sm_90: warp-specialized (1 producer warpgroup issuing cp.async.bulk, 2 consumer warpgroups
//        running wgmma m64nNk16 with the weights as the register A operand).
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

// ---- programmatic dependent launch: every kernel lets its successor launch as soon as all of its blocks
// are running, and waits for its predecessor's results before touching global memory (a no-op when the kernel was not
// launched with the PDL attribute) ----
__device__ __forceinline__ void pdl_enter() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
// Each lane arrives on the empty barrier to order its reads before slot reuse.
#define REL_LANES 32

namespace okc_moe {

constexpr int KS = 64;        // k-stages of 64 elements (K = 4096)
constexpr int WBLK = 4608;    // bytes per (ntile, kstage) weight block: 4096 code bytes + 512 scale bytes
constexpr int MAX_STAGES = 24;

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEVI void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEVI void mbar_inval(uint32_t bar) { asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(bar) : "memory"); }
// Norm weights are the model's BF16 effective weights (the exact values stock's norms multiply by), widened to fp32.
// g1p4: 4 consecutive weights (8-byte aligned) -> fp32.
DEVI float4 g1p4(const __nv_bfloat16* g) {
  const uint2 v = __ldg(reinterpret_cast<const uint2*>(g));
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
DEVI void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred P1;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@!P1 bra WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
DEVI void mbar_arrive(uint32_t bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(bar) : "memory");
}

DEVI void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEVI void prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
// A ring slot whose weight codes and scales the consumers read with generic loads (LDS) is about to be overwritten by
// the async proxy (bulk copy): after the empty-barrier acquire, a proxy fence orders those generic reads before the
// async writes (PTX memory model; compute-sanitizer racecheck flags the slot reuse without it).
DEVI void rel_sync() {
}
DEVI void slot_proxy_fence() {
}
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
// shared -> global bulk copy (async proxy), tracked per thread with bulk groups
DEVI void bulk_s2g(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes) : "memory");
}
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
DEVI void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}


DEVI void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// SwiGLU's x * sigmoid(x), written as stock SGLang's activation kernel writes it
// (kernels/jit/csrc/elementwise/activation.cuh; both builds use --use_fast_math).
DEVI float silu_fast(float x) {
  return x / (1.0f + expf(-x));
}

// wgmma descriptor for a K-major, 128B-swizzled B tile (rows of 128 B, 8-row atoms of 1024 B).
DEVI uint64_t desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;           // LBO (unused for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32; // SBO: 8 rows x 128 B
  d |= (uint64_t)1 << 62;           // SWIZZLE_128B
  return d;
}

template <int N>
DEVI void wgmma_rs(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d);
template <> DEVI void wgmma_rs<32>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<64>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
      "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<128>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, "
    "%41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
      "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
      "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<256>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, "
    "%41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, "
    "%91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, "
    "1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
      "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
      "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]),
      "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]),
      "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]),
      "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<96>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, "
    "%41, %42, %43, %44, %45, %46, %47}, {%48, %49, %50, %51}, %52, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
      "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<160>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %85, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n160k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, "
    "%41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79}, {%80, %81, %82, %83}, %84, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
      "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
      "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<192>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %101, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, "
    "%41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, "
    "%91, %92, %93, %94, %95}, {%96, %97, %98, %99}, %100, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
      "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
      "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]),
      "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<224>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %117, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n224k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, "
    "%41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, "
    "%91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111}, {%112, %113, %114, %115}, %116, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
      "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
      "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]),
      "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]),
      "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}


DEVI uint32_t bf162_u32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }
// mma.sync m16n8k16 (bf16 x bf16 -> fp32): the A fragment layout equals one warp's share of the wgmma RS m64 A operand
DEVI void mma16816(float* c, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEVI __nv_bfloat162 u32_bf162(uint32_t v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }

// Two E4M3 block scales (low byte: row g, high byte: row g+8) -> bf16x2 of scale * 2^119.
// Prepare rejects zero and subnormal E4M3 scales. For a normal byte b,
// the BF16 bits of scale * 2^119 are (b << 4) + 0x7780.
DEVI uint32_t scale_pair(uint32_t s2) {
  return __byte_perm(s2, 0u, 0x4140) * 16u + 0x77807780u;  // [b0, 0, b1, 0] -> (b << 4) + 0x7780 per half
}

// The scale pair of one k16 step straight from the 32-bit word holding two steps' pairs: the byte permute selects
// bytes 0,1 (hi = 0) or 2,3 (hi = 1) itself, so no shift or mask (hi is a compile-time constant in the unrolled loops).
DEVI uint32_t scale_pair_sel(uint32_t w, int hi, uint32_t m16 = 16u) {
  return __byte_perm(w, 0u, hi ? 0x4342 : 0x4140) * m16 + 0x77807780u;  // m16 opaque: IMAD on the FMA pipe

}

// 8 E2M1 codes -> 4 bf16x2 A-fragment registers (scaled). The repack places each code's bits so that one rotate and
// one mask put it in place: a register half holds (sign << 15 | magnitude << 6), which as bf16 bits is exactly
// e2m1 * 2^-126 (subnormal for magnitude 1), times scale * 2^119. Bit positions in q (sign, m2, m1, m0):
//   a0 (row g):    code A (15, 8, 7, 6),   code B (31, 24, 23, 22)    q & M
//   a1 (row g+8):  code A (21, 14, 13, 12), code B (5, 30, 29, 28)    rotr(q, 6) & M
//   a2 (row g):    code A (25, 18, 17, 16), code B (9, 2, 1, 0)       rotr(q, 10) & M
//   a3 (row g+8):  code A (11, 4, 3, 10),  code B (27, 20, 19, 26)    rotl(q, 4) & 0x81808180 | rotr(q, 4) & 0x00400040
// (9 ALU ops for 8 weights, no IMAD shifts; the repack in the template writes this layout.)
// Dequant layout (via the scout): 7 integer-pipe ops + 2 IMAD shifts per 8 weights. Two of the shifts
// become multiplies by 16 / 64 held in registers (m16 = 16 + opaque 0, m64 = 64 + opaque 0), which run on the FMA pipe
// as IMAD instead of on the half-rate integer pipe. Bit positions in q (m0, m1, m2, sign):
//   a0: code A (6, 7, 8, 15),   code B (22, 23, 24, 31)   q & M
//   a1: code A (2, 3, 4, 11),   code B (18, 19, 20, 27)   (q * 16) & M
//   a2: code A (28, 29, 30, 5), code B (12, 13, 14, 21)   rotl(q, 10) & M
//   a3: code A (0, 1, 26, 9),   code B (16, 17, 10, 25)   (q * 64) & 0x80C080C0 | rotl(q, 14) & 0x01000100
DEVI uint32_t opaque_zero() {
  uint32_t z;
  asm volatile("mov.u32 %0, 0;" : "=r"(z));
  return z;
}
DEVI void convert8(uint32_t q, uint32_t sb, uint32_t (&a)[4], uint32_t m16 = 16u, uint32_t m64 = 64u) {
  const __nv_bfloat162 s = u32_bf162(sb);
  const __nv_bfloat162 slo = __low2bfloat162(s), shi = __high2bfloat162(s);
  constexpr uint32_t M = 0x81C081C0u;
  const uint32_t r0 = q & M;
  const uint32_t r1 = (q * m16) & M;
  const uint32_t r2 = __funnelshift_l(q, q, 10) & M;
  const uint32_t r3 = ((q * m64) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  a[0] = bf162_u32(__hmul2(u32_bf162(r0), slo));
  a[1] = bf162_u32(__hmul2(u32_bf162(r1), shi));
  a[2] = bf162_u32(__hmul2(u32_bf162(r2), slo));
  a[3] = bf162_u32(__hmul2(u32_bf162(r3), shi));
}


struct Args {
  const uint8_t* wrep;   // [E][NT][KS][WBLK]
  const uint8_t* xt;     // activations in tile layout (byte base; a tile starting at row r0 is at r0 * 8192)
  const int4* tiles;     // (expert, row0, n, 0)
  const int* ntiles;     // device scalar
  const float* gscale;   // [E]
  uint8_t* out;          // GATED: tile layout (bytes); else row-major bf16 [rows, 4096]
  int NT;
  int smem_bytes;
  int* counter;          // persistent kernel work queue (zeroed before launch)
  int flags;             // bit 2: skip NVFP4 conversion for diagnostics (clear in production)
  uint32_t mul16, mul64;  // 16 and 64, from the host: multiplies by them stay IMAD (FMA-heavy pipe), since ptxas sees
                          // through an inline-asm zero and turns `x * (16 + 0)` back into an integer-pipe shift
};

template <int N, bool GATED>
DEVI void consumer_tile(const Args& p, uint8_t* ring, uint32_t bars, int stride, int nst, int e, int row0,
                        int ntile, int* done = nullptr) {
  const uint32_t zz_ = opaque_zero(), m16 = 16u + zz_, m64 = 64u + zz_;  // opaque multipliers (IMAD shifts)
  const int ctid = threadIdx.x - 128;  // 0..255
  const int wg = ctid >> 7;
  const int w = (ctid >> 5) & 3;
  const int lane = ctid & 31;
  const int g = lane >> 2, t = lane & 3;
  float acc[128];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  uint32_t afrag[2][4];
  int s = 0, prev_s = 0;
  uint32_t ph = 0;
  for (int ks = 0; ks < KS; ++ks) {
    mbar_wait(bars + 8 * s, ph);
    uint8_t* st = ring + s * stride;
    const uint4 wq = *reinterpret_cast<const uint4*>(st + N * 128 + ((wg * 4 + w) * 32 + lane) * 16);
    const uint2 sc = *reinterpret_cast<const uint2*>(st + N * 128 + 4096 + ((wg * 4 + w) * 8 + g) * 8);
    const uint64_t desc = desc_sw128(smem_u32(st));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
      const uint32_t sw = j < 2 ? sc.x : sc.y;  // scales of k16 steps (j & ~1, j | 1): bytes 0,1 and 2,3
      convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j & 1], m16, m64);
      wg_fence();
      wgmma_rs<N>(acc, afrag[j & 1], desc + 2 * j, 1);
      wg_commit();
      wg_wait<1>();
    }
    // steps 0-2 of this stage are complete: every wgmma reading the previous stage is done
    if (ks > 0) { rel_sync(); mbar_arrive(bars + 8 * MAX_STAGES + 8 * prev_s); }
    prev_s = s;
    if (++s == nst) { s = 0; ph ^= 1; }
  }
  wg_wait<0>();
  // ---- epilogue ----
  const float cs = p.gscale[e] * 128.f;  // weights were scaled by 2^-7
  named_bar_sync(1, 256);               // every consumer is done reading the ring
  uint8_t* stg = ring;
  if constexpr (GATED) {
    // rows g / g+8 of each 16-row slice are gate / up of inter column c = 8*slice + g
    const int q = wg * 4 + w;
#pragma unroll
    for (int jj = 0; jj < N / 8; ++jj) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const float gt = acc[4 * jj + b] * cs, up = acc[4 * jj + 2 + b] * cs;
        const float h = silu_fast(gt) * up;
        const int tok = 8 * jj + 2 * t + b;
        *reinterpret_cast<__nv_bfloat16*>(stg + tok * 128 + ((q ^ (tok & 7)) << 4) + g * 2) = __float2bfloat16_rn(h);
      }
    }
    named_bar_sync(1, 256);
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)row0 * 8192 + (size_t)ntile * N * 128);
    const uint4* src = reinterpret_cast<const uint4*>(stg);
    for (int i = ctid; i < N * 8; i += 256) dst[i] = src[i];
    if (done) {  // fused kernel: publish this n-tile of the tile's intermediate rows
      __threadfence();
      named_bar_sync(1, 256);
      if (ctid == 0) atomicAdd(done, 1);
    }
  } else {
    // output columns (within the 128-wide ntile) c0 = 16*slice + g, c1 = c0 + 8; staging rows of 272 B
    const int c0 = (wg * 4 + w) * 16 + g;
#pragma unroll
    for (int jj = 0; jj < N / 8; ++jj) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const int tok = 8 * jj + 2 * t + b;
        __nv_bfloat16* r = reinterpret_cast<__nv_bfloat16*>(stg + tok * 272);
        r[c0] = __float2bfloat16_rn(acc[4 * jj + b] * cs);
        r[c0 + 8] = __float2bfloat16_rn(acc[4 * jj + 2 + b] * cs);
      }
    }
    named_bar_sync(1, 256);
    for (int i = ctid; i < N * 16; i += 256) {
      const int tok = i >> 4, c = i & 15;
      *reinterpret_cast<uint4*>(p.out + ((size_t)(row0 + tok) * 4096 + ntile * 128) * 2 + c * 16) =
          *reinterpret_cast<const uint4*>(stg + tok * 272 + c * 16);
    }
  }
}

template <bool GATED>
__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Args p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);  // full[MAX_STAGES], empty[MAX_STAGES]
  uint8_t* ring = base + 1024;
  const int ring_bytes = p.smem_bytes - 1024 - (int)pad;
  const int ntile = blockIdx.x;
  const int ntiles = *p.ntiles;
  if (threadIdx.x < 128) {
    // ---------------- producer warpgroup ----------------
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      const int stride = (n * 128 + WBLK + 1023) & ~1023;
      int nst = ring_bytes / stride;
      nst = nst > MAX_STAGES ? MAX_STAGES : nst;
      if (threadIdx.x == 0) {
        for (int s = 0; s < nst; ++s) {
          mbar_init(bars + 8 * s, 1);
          mbar_init(bars + 8 * MAX_STAGES + 8 * s, 8 * REL_LANES);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      __syncthreads();
      const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
      const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
      const uint32_t xbytes = n * 128;
      if (threadIdx.x == 0) {
        for (int ks = 0; ks < KS; ++ks) {
          const int s = ks % nst;
          const uint32_t ph = (ks / nst) & 1;
          mbar_wait(bars + 8 * MAX_STAGES + 8 * s, ph ^ 1);
          slot_proxy_fence();
          const uint32_t full = bars + 8 * s;
          mbar_expect_tx(full, xbytes + WBLK);
          const uint32_t dst = smem_u32(ring + s * stride);
          bulk_g2s(dst, xsrc + (size_t)ks * xbytes, xbytes, full);
          bulk_g2s(dst + xbytes, wsrc + (size_t)ks * WBLK, WBLK, full);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        for (int s = 0; s < nst; ++s) {
          mbar_inval(bars + 8 * s);
          mbar_inval(bars + 8 * MAX_STAGES + 8 * s);
        }
      }
      __syncthreads();
    }
  } else {
    // ---------------- consumer warpgroups ----------------
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      const int stride = (n * 128 + WBLK + 1023) & ~1023;
      int nst = ring_bytes / stride;
      nst = nst > MAX_STAGES ? MAX_STAGES : nst;
      __syncthreads();
      switch (n) {
        case 32: consumer_tile<32, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 64: consumer_tile<64, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 128: consumer_tile<128, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 160: consumer_tile<160, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 192: consumer_tile<192, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        default: consumer_tile<256, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
      }
      __syncthreads();
      __syncthreads();
    }
  }
}

// ---- persistent variant: grid = #SMs, items (tile, ntile) round-robin, fixed slot ring + separate staging ----
constexpr int SLOT = 43008;     // bytes per ring slot (1024-aligned): fits kps k-stages of any tile size
constexpr int NSLOT = 4;
constexpr int STG_BYTES = 45056; // epilogue staging

// k-blocks per ring stage by token-tile size (a stage must fit SLOT: N=32 allows 1/2/4, N=64 and 128 1/2, N=256 1)
#define MOE_KPS32 4
#define MOE_KPS64 2
#define MOE_KPS128 2
#define MOE_GROUP4_MAX 128  // token tiles up to this size commit one wgmma group per stage (see GROUP4)
// tile widths: 32, 64 (small: weight-conversion-bound), 96, 128 (two k-blocks a stage), 160..256 in steps of 32 (one
// k-block a stage; an expert's last tile is its remaining rows rounded up to 32, not to 128/256)
DEVI int kps_for(int n) { return n >= 160 ? 1 : (n <= 32 ? MOE_KPS32 : (n <= 64 ? MOE_KPS64 : MOE_KPS128)); }
// A stage must fit its ring slot, and its k-block count must divide KS.
static_assert(MOE_KPS32 * (32 * 128 + WBLK) <= SLOT && 64 % MOE_KPS32 == 0, "MOE_KPS32: stage does not fit a ring slot");
static_assert(MOE_KPS64 * (64 * 128 + WBLK) <= SLOT && 64 % MOE_KPS64 == 0, "MOE_KPS64: stage does not fit a ring slot");
static_assert(MOE_KPS128 * (128 * 128 + WBLK) <= SLOT && 64 % MOE_KPS128 == 0, "MOE_KPS128: stage does not fit a ring slot");
static_assert(256 * 128 + WBLK <= SLOT, "N=256 stage does not fit a ring slot");

// Copy-out queue in shared memory (fused sm_90 kernel): pubq = base + 256, 8 entries of 32 bytes {dst, done, kind,
// bytes or tokens}; base + 240 = jobs published (st.release by consumer thread 0; negative = -(count + 1) at exit),
// base + 244 = consumer thread 0's own count, base + 248 = jobs whose staging reads the publisher has finished.
DEVI void enqueue_out(volatile long long* pubq, void* dst, int* done, int kind, int amount, int row0 = 0, int nreal = 0) {
  volatile int* cnt = reinterpret_cast<volatile int*>(pubq) - 3;  // base + 244
  const int n = *cnt;
  volatile long long* e = pubq + 4 * (n & 7);
  e[0] = (long long)dst;
  e[1] = (long long)done;
  reinterpret_cast<volatile int*>(e)[4] = kind;
  reinterpret_cast<volatile int*>(e)[5] = amount;
  reinterpret_cast<volatile int*>(e)[6] = row0;   // GEMM2 tile's first row and real row count
  reinterpret_cast<volatile int*>(e)[7] = nreal;
  *cnt = n + 1;
  asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32((const void*)(pubq - 2))), "r"(n + 1) : "memory");
}
// consumer thread 0: wait until the publisher has read every queued staging buffer
DEVI void wait_staging_free(volatile long long* pubq) {
  const int n = *(reinterpret_cast<volatile int*>(pubq) - 3);
  const uint32_t a = smem_u32((const void*)(reinterpret_cast<volatile int*>(pubq) - 2));  // base + 248
  while (true) {
    int f;
    asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(f) : "r"(a) : "memory");
    if (f >= n) break;
  }
}

// pubq (fused sm_90 kernel): instead of fencing and bumping `done` itself, the consumers hand the GEMM1 n-tile's
// completion to the producer warpgroup's publisher warp (smem queue at pubq = base + 256, published count at base + 240,
// consumer thread 0's running count at base + 244), so the next item's main loop does not wait for the store fence.
#define MOE_CB 128  // first consumer thread
#define MOE_PB 0  // first producer thread
#define MOE_FUSED_THREADS 384
template <int N, int CB = 128>
DEVI void consumer_item(const Args& p, uint8_t* ring, uint8_t* stg, uint32_t bars, int& slot, uint32_t& ph,
                        int e, int row0, int ntile, bool GATED, int* done = nullptr, unsigned long long* t_mid = nullptr,
                        volatile long long* pubq = nullptr, int nreal = 256) {
#define MOE_IMAD_MAXN 64
  const uint32_t zz_ = N <= MOE_IMAD_MAXN ? opaque_zero() : 0u;
  // small tiles: multipliers ptxas cannot see (host values), so the shifts run as IMAD on the FMA-heavy pipe
  const uint32_t m16 = N <= MOE_IMAD_MAXN ? p.mul16 : 16u + zz_;
  const uint32_t m64 = N <= MOE_IMAD_MAXN ? p.mul64 : 64u + zz_;
  constexpr int KPS = N >= 160 ? 1 : (N <= 32 ? MOE_KPS32 : (N <= 64 ? MOE_KPS64 : MOE_KPS128));
  constexpr int XB = N * 128;  // X bytes per k-stage
  const int ctid = threadIdx.x - CB;
  const int wg = ctid >> 7;
  const int w = (ctid >> 5) & 3;
  const int lane = ctid & 31;
  const int g = lane >> 2, t = lane & 3;
  // small token tiles: the wgmma is short, so one commit group carries a whole k-stage (4 x k16) and the next
  // stage's conversions overlap it (one group in flight, double-buffered A fragments)
  // Grouping the 96-row tile causes ptxas to serialize WGMMA.
  constexpr bool GROUP4 = N <= MOE_GROUP4_MAX && N != 96;
  // small token tiles: a commit group's MMAs are shorter than the wgmma latency, so consecutive groups accumulate into
  // two independent register sets (summed before the epilogue) instead of one dependency chain
#define MOE_NACC4_MAXN 32
  // The smallest grouped tiles also split each group into two accumulator chains.
  constexpr int NACC = GROUP4 ? (N <= MOE_NACC4_MAXN ? 4 : 2) : 1;
  float acc[128];
  float acc2[128];  // only the first N/2 are live (and only when NACC > 1)
  float acc3[128], acc4[128];  // NACC == 4 only
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (NACC > 1 ? N / 2 : 0); ++i) acc2[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (NACC > 2 ? N / 2 : 0); ++i) acc3[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (NACC > 3 ? N / 2 : 0); ++i) acc4[i] = 0.f;
  // Two A-fragment buffers alternate by k-block parity.
  constexpr int AFB = 2;
  uint32_t afrag[GROUP4 ? 4 * AFB : 2][4];
  int prev = -1;
  const int woff = ((wg * 4 + w) * 32 + lane) * 16, soff = 4096 + ((wg * 4 + w) * 8 + g) * 8;
  // packed weights/scales are loaded one k-block ahead, across stage boundaries too
  mbar_wait(bars + 8 * slot, ph);
  uint4 nwq = *reinterpret_cast<const uint4*>(ring + slot * SLOT + KPS * XB + woff);
  uint2 nsc = *reinterpret_cast<const uint2*>(ring + slot * SLOT + KPS * XB + soff);
  for (int it = 0; it < KS / KPS; ++it) {
    uint8_t* st = ring + slot * SLOT;
    const uint8_t* wbase = st + KPS * XB + woff;
    const uint8_t* sbase = st + KPS * XB + soff;
    const uint64_t dbase = desc_sw128(smem_u32(st));
    if (it > 0) {
      mbar_wait(bars + 8 * slot, ph);
      nwq = *reinterpret_cast<const uint4*>(wbase);
      nsc = *reinterpret_cast<const uint2*>(sbase);
    }
#pragma unroll
    for (int kk = 0; kk < KPS; ++kk) {
      const uint4 wq = nwq;
      const uint2 sc = nsc;
      if (kk + 1 < KPS) {
        nwq = *reinterpret_cast<const uint4*>(wbase + (kk + 1) * WBLK);
        nsc = *reinterpret_cast<const uint2*>(sbase + (kk + 1) * WBLK);
      }

      // the k-block's X tile sits kk * XB bytes into the slot: its descriptor is the slot's plus kk * XB / 16
      const uint64_t desc = dbase + (uint64_t)((kk * XB) >> 4);
      if constexpr (GROUP4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;  // scales of k16 steps (j & ~1, j | 1): bytes 0,1 and 2,3
          convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[(kk & 1) * 4 + j], m16, m64);
        }
        wg_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j)
          wgmma_rs<N>(NACC > 2 ? ((kk & 1) ? ((j & 1) ? acc4 : acc2) : ((j & 1) ? acc3 : acc))
                               : ((NACC > 1 && (kk & 1)) ? acc2 : acc),
                      afrag[(kk & 1) * 4 + j], desc + 2 * j, 1);
        wg_commit();
        wg_wait<1>();
        // The previous stage completes at this wait on the first k-block.
        if (kk == 0 && prev >= 0) { rel_sync(); mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;  // scales of k16 steps (j & ~1, j | 1): bytes 0,1 and 2,3
          convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j & 1], m16, m64);
          wg_fence();
          wgmma_rs<N>(acc, afrag[j & 1], desc + 2 * j, 1);
          wg_commit();
          wg_wait<1>();
          if (kk == 0 && j == 0 && prev >= 0) { rel_sync(); mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
        }
      }
    }
    prev = slot;
    if (++slot == NSLOT) { slot = 0; ph ^= 1; }
  }
  wg_wait<0>();
  rel_sync();
  mbar_arrive(bars + 8 * NSLOT + 8 * prev);
  if (t_mid) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *t_mid = t;
  }
  if constexpr (NACC > 3) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[i] = (acc[i] + acc3[i]) + (acc2[i] + acc4[i]);
  } else if constexpr (NACC > 2) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[i] = (acc[i] + acc2[i]) + acc3[i];
  } else if constexpr (NACC > 1) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[i] += acc2[i];
  }
  // ---- epilogue through the dedicated staging buffer ----
  const float cs = p.gscale[e] * 128.f;
  if (pubq && ctid == 0) wait_staging_free(pubq);
  named_bar_sync(1, 256);  // previous item's copy-out is done with the staging buffer
  if (GATED) {
    const int q = wg * 4 + w;
    // stmatrix.trans: the accumulator fragment of an 8-token group is an 8x8 matrix [row g][token]; stored transposed,
    // each token's 8 rows (16 bytes) land in its tile-layout chunk in one instruction per 4 groups
#pragma unroll
    for (int j4 = 0; j4 < N / 32; ++j4) {
      uint32_t r[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int jj = 4 * j4 + u;
        const float h0 = silu_fast(acc[4 * jj] * cs) * (acc[4 * jj + 2] * cs);
        const float h1 = silu_fast(acc[4 * jj + 1] * cs) * (acc[4 * jj + 3] * cs);
        r[u] = bf162_u32(__floats2bfloat162_rn(h0, h1));
      }
      const int rr = lane & 7, tokr = 8 * (4 * j4 + (lane >> 3)) + rr;
      const uint32_t addr = smem_u32(stg + tokr * 128 + ((q ^ rr) << 4));
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r[0]),
                   "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
    }
    if (pubq) {  // fused sm_90: the publisher warp copies the staged tile out (bulk copy) and then publishes `done`
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      named_bar_sync(1, 256);
      if (ctid == 0) enqueue_out(pubq, p.out + (size_t)row0 * 8192 + (size_t)ntile * N * 128, done, 0, N * 128);
    } else {
      named_bar_sync(1, 256);
      uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)row0 * 8192 + (size_t)ntile * N * 128);
      const uint4* src = reinterpret_cast<const uint4*>(stg);
      for (int i = ctid; i < N * 8; i += 256) dst[i] = src[i];
      if (done) {  // fused kernel: publish this n-tile of the tile's intermediate rows
        __threadfence();
        named_bar_sync(1, 256);
        if (ctid == 0) atomicAdd(done, 1);
      }
    }
  } else {
    const int c0 = (wg * 4 + w) * 16 + g;
    // halves of up to 128 tokens through the 128-row staging buffer: half 1 (tiles over 128 rows) holds N - 128 rows
    auto gemm2_half = [&]<int half, int NJ>() {
      if (half) named_bar_sync(1, 256);  // (half 0 was copied out by the consumers themselves)
      // stmatrix.trans: per 8-token group, two 8x8 matrices (this warp's columns 16w..16w+7 and 16w+8..16w+15)
      if constexpr (NJ >= 2) {
#pragma unroll
        for (int j2 = 0; j2 < NJ / 2; ++j2) {
          uint32_t r[4];
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            const int ai = 4 * (2 * j2 + u + half * 16);
            r[2 * u] = bf162_u32(__floats2bfloat162_rn(acc[ai] * cs, acc[ai + 1] * cs));
            r[2 * u + 1] = bf162_u32(__floats2bfloat162_rn(acc[ai + 2] * cs, acc[ai + 3] * cs));
          }
          const int m = lane >> 3, rr = lane & 7;
          const int tl = 8 * (2 * j2 + (m >> 1)) + rr;
          const uint32_t addr = smem_u32(stg + tl * 272 + ((wg * 4 + w) * 16 + 8 * (m & 1)) * 2);
          asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r[0]),
                       "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
        }
      } else {
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
          for (int b = 0; b < 2; ++b) {
            const int tl = 8 * jj + 2 * t + b;
            const int ai = 4 * (jj + half * 16) + b;
            __nv_bfloat16* r = reinterpret_cast<__nv_bfloat16*>(stg + tl * 272);
            r[c0] = __float2bfloat16_rn(acc[ai] * cs);
            r[c0 + 8] = __float2bfloat16_rn(acc[ai + 2] * cs);
          }
        }
      }
      constexpr int ntok = 8 * NJ;
      // the last half goes to the publisher warp (its copy loop overlaps the next item); a first half is copied by the
      // consumers, since the second half needs the staging buffer right away
      if (pubq && half == (N > 128 ? 1 : 0)) {
        named_bar_sync(1, 256);
        if (ctid == 0)
          enqueue_out(pubq, p.out + ((size_t)(row0 + half * 128) * 4096 + ntile * 128) * 2, done, 1, ntok, row0, nreal);
      } else {
        named_bar_sync(1, 256);
        for (int i = ctid; i < ntok * 16; i += 256) {
          const int tl = i >> 4, c = i & 15;
          *reinterpret_cast<uint4*>(p.out + ((size_t)(row0 + half * 128 + tl) * 4096 + ntile * 128) * 2 + c * 16) =
              *reinterpret_cast<const uint4*>(stg + tl * 272 + c * 16);
        }
      }
    };
    gemm2_half.template operator()<0, (N > 128 ? 16 : N / 8)>();
    if constexpr (N > 128) gemm2_half.template operator()<1, (N - 128) / 8>();
  }
}

template <bool GATED>
__global__ void __launch_bounds__(384, 1) moe_gemm_persistent(Args p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);  // full[NSLOT], empty[NSLOT], ifull[2], iempty[2]
  const uint32_t ibar = bars + 16 * NSLOT;
  volatile int* iring = reinterpret_cast<volatile int*>(base + 512);
  uint8_t* ring = base + 1024;
  uint8_t* stg = ring + NSLOT * SLOT;
  const int nitems = *p.ntiles * p.NT;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NSLOT; ++s) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * NSLOT + 8 * s, 8 * REL_LANES);
    }
    for (int k = 0; k < 2; ++k) {
      mbar_init(ibar + 8 * k, 1);
      mbar_init(ibar + 16 + 8 * k, 256);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  int slot = 0;
  uint32_t ph = 0;
  if (threadIdx.x < 128) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (threadIdx.x == 0) {
      int k = 0;
      uint32_t iph = 0;
      while (true) {
        mbar_wait(ibar + 16 + 8 * k, iph ^ 1);
        const int item = atomicAdd(p.counter, 1);
        iring[k] = item;
        mbar_arrive(ibar + 8 * k);
        if (item >= nitems) break;
        const int tile = item / p.NT, ntile = item - tile * p.NT;
        const int4 ti = p.tiles[tile];
        const int e = ti.x, row0 = ti.y, n = ti.z;
        const int kps = kps_for(n);
        const uint32_t xb = n * 128;
        const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
        const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
        for (int it = 0; it < KS / kps; ++it) {
          mbar_wait(bars + 8 * NSLOT + 8 * slot, ph ^ 1);
          slot_proxy_fence();
          const uint32_t full = bars + 8 * slot;
          mbar_expect_tx(full, kps * (xb + WBLK));
          const uint32_t dst = smem_u32(ring + slot * SLOT);
          bulk_g2s(dst, xsrc + (size_t)it * kps * xb, kps * xb, full);
          bulk_g2s(dst + kps * xb, wsrc + (size_t)it * kps * WBLK, kps * WBLK, full);
          if (++slot == NSLOT) { slot = 0; ph ^= 1; }
        }
        k ^= 1;
        if (k == 0) iph ^= 1;
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    int k = 0;
    uint32_t iph = 0;
    while (true) {
      mbar_wait(ibar + 8 * k, iph);
      const int item = iring[k];
      named_bar_sync(1, 256);  // every consumer has read the item
      mbar_arrive(ibar + 16 + 8 * k);  // Each lane orders its own reads before releasing the slot.
      if (item >= nitems) break;
      k ^= 1;
      if (k == 0) iph ^= 1;
      const int tile = item / p.NT, ntile = item - tile * p.NT;
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      switch (n) {
        case 32: consumer_item<32>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 64: consumer_item<64>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 128: consumer_item<128>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 160: consumer_item<160>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 192: consumer_item<192>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        default: consumer_item<256>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
      }
    }
  }
}

// ---- both expert GEMMs in one persistent launch: the queue holds every GEMM1 item, then every GEMM2 item; a GEMM2
// item's loads wait until all 64 GEMM1 n-tiles of its token tile have been written (per-tile counters) ----
struct FusedArgs {
  Args g1, g2;     // GEMM1 (gated, writes inter in tile layout) and GEMM2 (reads inter, writes y)
  int* done;       // [max tiles] GEMM1 n-tiles finished per token tile (zeroed before launch)
  unsigned long long* dbg;  // optional per-item timeline: [item][start ns, end ns, smid, cta] (nullptr: off)
  int flags;       // bits 0/1: limit X/W loads to the first NSLOT stages; bit 8: fine remainder tiles
  // Retained for the host argument layout. The fused kernel uses a separate combine launch.
  int* cit;        // [max tiles] GEMM2 n-tiles finished per tile
  int* qctl;       // [0] queue head (pops), [1] queue tail (pushes), [2] error flag (a spin limit hit)
  int* tok_left;   // [T] local routes of each token not yet finished
  int* cq;         // [T] combine queue: token + 1 (0: not yet written)
  const int* row_tok;  // [rows] token of each routed row
  const int* rrow; const int* lid; const float* topw; const __nv_bfloat16* pw;  // the combine's inputs
  __nv_bfloat16* out;  // [T, 4096] combined expert output
  int T;
};

template <int FINE>
__global__ void __launch_bounds__(MOE_FUSED_THREADS, 1) moe_fused_persistent(FusedArgs fa) {
  pdl_enter();
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);
  const uint32_t ibar = bars + 16 * NSLOT;
  volatile int* iring = reinterpret_cast<volatile int*>(base + 512);
  uint8_t* ring = base + 1024;
  uint8_t* stg = ring + NSLOT * SLOT;
  // publisher queue (sm_90): count at base + 240, 32 done-pointers at base + 256
  volatile long long* pubq = reinterpret_cast<volatile long long*>(base + 256);
  if (threadIdx.x == 0) {
    *reinterpret_cast<volatile int*>(base + 240) = 0;
    *reinterpret_cast<volatile int*>(base + 244) = 0;
    *reinterpret_cast<volatile int*>(base + 248) = 0;
    for (int s = 0; s < NSLOT; ++s) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * NSLOT + 8 * s, 8 * REL_LANES);
    }
    for (int k = 0; k < 2; ++k) {
      mbar_init(ibar + 8 * k, 1);
      mbar_init(ibar + 16 + 8 * k, 256);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nt = *fa.g1.ntiles;
  const int n1 = nt * fa.g1.NT, nitems = n1 + nt * fa.g2.NT;
  int slot = 0;
  uint32_t ph = 0;
  if (threadIdx.x < 128) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (threadIdx.x == MOE_PB) {
      int k = 0;
      uint32_t iph = 0;
      while (true) {
        mbar_wait(ibar + 16 + 8 * k, iph ^ 1);
        const int item = atomicAdd(fa.g1.counter, 1);
        iring[k] = item;
        mbar_arrive(ibar + 8 * k);
        if (item >= nitems) break;
        const bool second = item >= n1;
        const Args& p = second ? fa.g2 : fa.g1;
        const int it2 = second ? item - n1 : item;
        const int tile = it2 / p.NT, ntile = it2 - tile * p.NT;
        const int4 ti = p.tiles[tile];
        const int e = ti.x, row0 = ti.y, n = ti.z;
        if (second) {
          const int need = fa.g1.NT;
          int v;
          while (true) {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(fa.done + tile) : "memory");
            if (v >= need) break;
            __nanosleep(64);
          }
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        const int kps = kps_for(n);
        const uint32_t xb = n * 128;
        const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
        const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
        // Prefetch the first stage while consumers finish the previous item.
        for (int j = 0; j < 1 && j < KS / kps; ++j) prefetch_l2(wsrc + (size_t)j * kps * WBLK, kps * WBLK);
        for (int it = 0; it < KS / kps; ++it) {
          if (it + 1 < KS / kps) prefetch_l2(wsrc + (size_t)(it + 1) * kps * WBLK, kps * WBLK);
          mbar_wait(bars + 8 * NSLOT + 8 * slot, ph ^ 1);
          slot_proxy_fence();
          const uint32_t full = bars + 8 * slot;
          const bool lx = !(fa.flags & 1) || it < NSLOT, lw = !(fa.flags & 2) || it < NSLOT;
          mbar_expect_tx(full, (lx ? kps * xb : 0) + (lw ? kps * WBLK : 0) + (!lx && !lw ? 16 : 0));
          const uint32_t dst = smem_u32(ring + slot * SLOT);
          if (lx) bulk_g2s(dst, xsrc + (size_t)it * kps * xb, kps * xb, full);
          if (lw) bulk_g2s(dst + kps * xb, wsrc + (size_t)it * kps * WBLK, kps * WBLK, full);
          if (!lx && !lw) bulk_g2s(dst, xsrc, 16, full);
          if (++slot == NSLOT) { slot = 0; ph ^= 1; }
        }
        k ^= 1;
        if (k == 0) iph ^= 1;
      }
    } else if (threadIdx.x >= MOE_PB + 32 && threadIdx.x < MOE_PB + 64) {
      // publisher warp: copies each staged epilogue tile out with bulk copies (the consumers go on with their next
      // item), frees the staging buffer once read, and for GEMM1 tiles waits for the writes, then a GPU-scope release
      // and the tile's done counter (read with acquire by other CTAs' producers)
      const int lane = threadIdx.x - (MOE_PB + 32);
      const uint32_t stg_s = smem_u32(stg);
      int seen = 0;
      while (true) {
        int r = 0;
        asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(r) : "r"(smem_u32(base + 240)) : "memory");
        r = __shfl_sync(0xffffffffu, r, 0);  // one value for the warp (a later lane may see a newer count)
        const int avail = r < 0 ? -r - 1 : r;
        while (seen < avail) {
          volatile long long* e = pubq + 4 * (seen & 7);
          uint8_t* dst = reinterpret_cast<uint8_t*>(e[0]);
          int* d = reinterpret_cast<int*>(e[1]);
          const int kind = reinterpret_cast<volatile int*>(e)[4], amount = reinterpret_cast<volatile int*>(e)[5];
          if (kind == 0) {  // one contiguous tile-layout block: a single bulk copy
            if (lane == 0) {
              bulk_s2g(dst, stg_s, amount);
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
              asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            }
          } else {  // token rows of 256 bytes (272-byte staging pitch): the warp copies them, 16 bytes per lane
            for (int i = lane; i < amount * 16; i += 32) {
              const int tl = i >> 4, c = i & 15;
              *reinterpret_cast<uint4*>(dst + (size_t)tl * 8192 + c * 16) = *reinterpret_cast<const uint4*>(stg + tl * 272 + c * 16);
            }
          }
          __syncwarp();  // after lane 0's bulk-copy read of the staging buffer has completed
          asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32(base + 248)), "r"(seen + 1) : "memory");
          if (lane == 0) {
            if (d) {
              if (kind == 0) {  // a bulk (async-proxy) copy: wait for its writes first
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                asm volatile("fence.proxy.async.global;" ::: "memory");
              }
              asm volatile("fence.acq_rel.gpu;" ::: "memory");
              atomicAdd(d, 1);
            }
          }
          ++seen;
        }
        if (r < 0) break;
        __nanosleep(64);
      }
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    int k = 0;
    uint32_t iph = 0;
    volatile long long* pq = pubq;
    while (true) {
      mbar_wait(ibar + 8 * k, iph);
      const int item = iring[k];
      named_bar_sync(1, 256);
      mbar_arrive(ibar + 16 + 8 * k);  // Each lane orders its own reads before releasing the slot.
      if (item >= nitems) {
        if (threadIdx.x == MOE_CB) {  // the publisher drains the queue and exits
          const int pubn = *reinterpret_cast<volatile int*>(base + 244);
          asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32(base + 240)), "r"(-(pubn + 1)) : "memory");
        }
        break;
      }
      k ^= 1;
      if (k == 0) iph ^= 1;
      unsigned long long t_start = 0;
      if (fa.dbg && threadIdx.x == MOE_CB) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
      const bool first = item < n1;
      const Args& p = first ? fa.g1 : fa.g2;
      const int it2 = first ? item : item - n1;
      const int tile = it2 / p.NT, ntile = it2 - tile * p.NT;
      const int4 ti = p.tiles[tile];
      int* dn = first ? fa.done + tile : nullptr;
      unsigned long long t_mid = 0;
      unsigned long long* tm = (fa.dbg && threadIdx.x == MOE_CB) ? &t_mid : nullptr;
      switch (ti.z) {
        case 32: consumer_item<32, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w); break;
        case 64: consumer_item<64, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w); break;
        case 128: consumer_item<128, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w); break;
        case 160:
          if constexpr (FINE != 0) {
            consumer_item<160, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
          }
          break;
        case 192:
          if constexpr (FINE != 0) {
            consumer_item<192, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
          }
          break;
        default: consumer_item<256, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w); break;
      }
      if (fa.dbg && threadIdx.x == MOE_CB) {
        unsigned long long t_end;
        unsigned int smid;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        fa.dbg[(size_t)item * 4 + 0] = t_start;
        fa.dbg[(size_t)item * 4 + 1] = t_end;
        fa.dbg[(size_t)item * 4 + 2] = smid | ((unsigned long long)(t_mid - t_start) << 16);
        fa.dbg[(size_t)item * 4 + 3] = blockIdx.x;
      }
    }
  }
}

// ---- routing plan: per-expert tiles of 256 rows plus one remainder tile of 32/64/128/256 ----
// an expert's last (remainder) tile: 32 or 64 rows when it fits, else the rows rounded up to 32
DEVI int rem_tile(int r, int fine) {
  if (r == 0) return 0;
  if (r <= 32) return 32;
  if (r <= 64) return 64;
  if (!fine) return r <= 128 ? 128 : 256;
  if (r <= 128) return 128;
  if (r <= 160) return 160;
  if (r <= 192) return 192;
  return 256;
}
// Without fine widths, split 129..192 remainder rows into 128 + 32/64.
DEVI void rem_tiles(int r, int fine, int& na, int& nb) {
  if (!fine && r > 128 && r <= 192) {
    na = 128;
    nb = rem_tile(r - 128, 0);
  } else {
    na = rem_tile(r, fine);
    nb = 0;
  }
}
DEVI void plan_warp(const int* __restrict__ counts, int* __restrict__ offsets, int4* __restrict__ tiles,
                    int* __restrict__ ntiles, int* __restrict__ rows_per_expert, int order, int e) {
  const int fine = order >> 8;  // bit 8: fine remainder tiles (the expert kernel must be the FINE instantiation)
  order &= 255;
  // Tile list order (the persistent kernel takes items from it through an atomic counter):
  //   0: expert order; 1: biggest first (all 256-row tiles, then 128, 64, 32: longest-processing-time);
  //   2: the 256/128-row (compute-bound) tiles biggest first with the 64/32-row (weight-streaming) tiles spread
  //      between them, the last fifth of the small tiles at the end.
  // Each expert's rows stay contiguous (offsets) in tile-row order.
  const int c = counts[e];  // one warp, lane e = expert e
  const int full = c >> 8, r = c & 255;
  int nr, nb;  // remainder tile widths (nb: second remainder tile or 0)
  rem_tiles(r, fine, nr, nb);
  const int rows = full * 256 + nr + nb;
  const int nt = full + (nr > 0) + (nb > 0);
  // size classes, widest first: 256, 224, 192, 160, 128, 96 ("big", tensor-bound), 64, 32 ("small")
  // cnt: [0] rows, [1 + k] tiles of class k (k = 0..7), [9] tiles
  int cnt[10];
  cnt[0] = rows;
#pragma unroll
  for (int k = 0; k < 8; ++k) cnt[1 + k] = (k == 0 ? full : 0) + (nr == 256 - 32 * k) + (nb == 256 - 32 * k);
  cnt[9] = nt;
  int incl[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) {
    int v = cnt[k];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a2 = __shfl_up_sync(0xffffffffu, v, o);
      if (e >= o) v += a2;
    }
    incl[k] = v;
  }
  int tot[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) tot[k] = __shfl_sync(0xffffffffu, incl[k], 31);
  const int r0 = incl[0] - rows;
  offsets[e] = r0;
  rows_per_expert[e] = rows;
  if (e == 31) *ntiles = tot[9];
  if (order == 0) {
    const int t0 = incl[9] - nt;
    for (int i = 0; i < full; ++i) tiles[t0 + i] = make_int4(e, r0 + 256 * i, 256, 256);
    if (nr) tiles[t0 + full] = make_int4(e, r0 + 256 * full, nr, nb ? nr : r);
    if (nb) tiles[t0 + full + 1] = make_int4(e, r0 + 256 * full + nr, nb, r - nr);
    return;
  }
  // class bases in widest-first order; big ranks k (classes 0..5), small ranks j (classes 6, 7)
  int cbase[8];
  {
    int acc = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) { cbase[k] = (k == 6) ? 0 : acc; acc = (k == 5) ? 0 : acc + tot[1 + k]; }
  }
  const int B = tot[1] + tot[2] + tot[3] + tot[4] + tot[5] + tot[6], S = tot[7] + tot[8];
  auto pos_big = [&](int k) -> int {
    if (order == 1 || B == 0) return k;
    return k + min(S, (4 * k * S) / (5 * B));
  };
  auto pos_small = [&](int j) -> int {
    if (order == 1) return B + j;
    if (B == 0) return j;
    const int lim = (4 * S) / 5;
    const int bb = j < lim ? (5 * B * (j + 1) + 4 * S - 1) / (4 * S) : B;
    return j + bb;
  };
  // class 0 (256): this expert's full tiles and a 256-row remainder, at in-class ranks incl - cnt ..
  const int p256 = cbase[0] + incl[1] - cnt[1];
  for (int i = 0; i < full; ++i) tiles[pos_big(p256 + i)] = make_int4(e, r0 + 256 * i, 256, 256);
  if (nr == 256) tiles[pos_big(p256 + full)] = make_int4(e, r0 + 256 * full, 256, r);
#pragma unroll
  for (int k = 1; k < 6; ++k)
    if (nr == 256 - 32 * k) tiles[pos_big(cbase[k] + incl[1 + k] - 1)] = make_int4(e, r0 + 256 * full, nr, nb ? nr : r);
  if (nr == 64) tiles[pos_small(cbase[6] + incl[7] - 1)] = make_int4(e, r0 + 256 * full, 64, r);
  if (nr == 32) tiles[pos_small(cbase[7] + incl[8] - 1)] = make_int4(e, r0 + 256 * full, 32, r);
  // a split remainder's second tile (32 or 64 rows; the first one, 128 rows, was placed above)
  if (nb == 64) tiles[pos_small(cbase[6] + incl[7] - 1)] = make_int4(e, r0 + 256 * full + nr, 64, r - nr);
  if (nb == 32) tiles[pos_small(cbase[7] + incl[8] - 1)] = make_int4(e, r0 + 256 * full + nr, 32, r - nr);
}

__global__ void moe_plan_kernel(const int* __restrict__ counts, int* __restrict__ offsets, int4* __restrict__ tiles,
                                int* __restrict__ ntiles, int* __restrict__ rows_per_expert, int order) {
  pdl_enter();
  plan_warp(counts, offsets, tiles, ntiles, rows_per_expert, order, threadIdx.x);  // 32 threads
}

// ---- normalize the pooler output and scatter each token's row into its local routes' tile slots ----
// p: [T, 4096] bf16; lid/pos: [T, 8]; writes xt (tile layout); zero-fills pad rows; rrow[t*8+k] = row (or -1)
#define MOE_GATHER_MINB 1
#define MOE_PADFILL_PARTS 8  // pad-fill blocks per expert (fewer blocks: the gather grid fits one wave at small T)
// PLAN: the block works out the expert offsets itself (one warp scan over the 32 counts) and block 0 also writes the
// whole plan (offsets, tile list, tile count, rows per expert) for the expert kernel: no separate plan launch.
template <bool PLAN>
__global__ void __launch_bounds__(256, MOE_GATHER_MINB) moe_gather_kernel(const __nv_bfloat16* __restrict__ p,
                                                         const __nv_bfloat16* __restrict__ gw, const int* __restrict__ lid,
                                                         const int* __restrict__ pos, const int* __restrict__ offsets_in,
                                                         const int* __restrict__ counts, uint8_t* __restrict__ xt,
                                                         int* __restrict__ rrow, int T, int* __restrict__ offsets_out = nullptr,
                                                         int4* __restrict__ tiles = nullptr, int* __restrict__ ntiles = nullptr,
                                                         int* __restrict__ rows_out = nullptr, int order = 0,
                                                         int* __restrict__ row_tok = nullptr, int* __restrict__ cq = nullptr,
                                                         int* __restrict__ qtail = nullptr, bool normalize = true) {
  pdl_enter();
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  const int fine = order >> 8;
  __shared__ float red[8];
  __shared__ int off_s[32];
  const int* offsets = offsets_in;
  if constexpr (PLAN) {
    if (tid < 32) {
      const int c = counts[tid];
      const int full = c >> 8, r = c & 255;
      int nr, nb;
      rem_tiles(r, fine, nr, nb);
      const int rows = full * 256 + nr + nb;
      int v = rows;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int a2 = __shfl_up_sync(0xffffffffu, v, o);
        if (tid >= o) v += a2;
      }
      off_s[tid] = v - rows;
      if (blockIdx.x == 0) plan_warp(counts, offsets_out, tiles, ntiles, rows_out, order, tid);
    }
    offsets = off_s;
  }
  if (tok >= T) {
    if constexpr (PLAN) __syncthreads();
    // pad-row zero fill: block (T + e*PARTS + part) zeroes part of expert e's pad rows
    const int idx = tok - T, e = idx / MOE_PADFILL_PARTS, part = idx % MOE_PADFILL_PARTS;
    const int c = counts[e];
    const int full = c >> 8, r = c & 255;
    int nr, nb;
    rem_tiles(r, fine, nr, nb);
    // remainder tile(s): pad rows [used, width) of each; tile layout: 64 stage blocks of width * 128 B
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int n = q ? nb : nr;
      if (n == 0) continue;
      const int r0 = offsets[e] + 256 * full + (q ? nr : 0);
      const int used = q ? r - nr : (r < nr ? r : nr);
      for (int lr = used + part; lr < n; lr += MOE_PADFILL_PARTS) {
        for (int i = tid; i < 64 * 8; i += 256) {
          const int stg = i >> 3, ch = i & 7;
          *reinterpret_cast<uint4*>(xt + (size_t)r0 * 8192 + (size_t)stg * n * 128 + lr * 128 + ch * 16) =
              make_uint4(0, 0, 0, 0);
        }
      }
    }
    return;
  }
  // thread handles elements [16*tid, 16*tid+16): stage tid/4, chunks 2*(tid%4), 2*(tid%4)+1
  const uint4* src = reinterpret_cast<const uint4*>(p + (size_t)tok * 4096 + tid * 16);
  uint4 v0 = src[0], v1 = src[1];
  // the token's routes (row base, tile rows, row in tile), worked out by 8 threads while the row load is in flight
  __shared__ long long mrow[8];
  __shared__ int mn[8], mlr[8];
  if constexpr (PLAN) __syncwarp();
  if (tid < 8) {
    const int k = tid;
    const int l = lid[tok * 8 + k];
    if (l < 0) {
      rrow[tok * 8 + k] = -1;
      mn[k] = 0;
    } else {
      const int ps = pos[tok * 8 + k];
      const int c = counts[l];
      const int full = c >> 8, r = c & 255;
      int nr, nb;
      rem_tiles(r, fine, nr, nb);
      const int tl = ps >> 8;
      int row0 = offsets[l] + 256 * tl, lr = ps & 255, n = tl < full ? 256 : nr;
      if (tl >= full && nb && lr >= nr) {  // a split remainder's second tile
        row0 += nr;
        lr -= nr;
        n = nb;
      }
      rrow[tok * 8 + k] = row0 + lr;
      if (row_tok != nullptr) row_tok[row0 + lr] = tok;  // CIT: which token a GEMM2 row belongs to
      mrow[k] = (long long)row0 * 8192 + lr * 128;
      mn[k] = n;
      mlr[k] = lr;
    }
    if (cq != nullptr) {  // CIT: a token without local routes is ready now (its combine writes zeros)
      const unsigned lm = __ballot_sync(0xffu, l >= 0);
      if (k == 0 && lm == 0) {
        const int i = atomicAdd(qtail, 1);
        asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(cq + i), "r"(tok + 1) : "memory");
      }
    }
  }
  float f[16];
  const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
  const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 a = __bfloat1622float2(h0[i]), b = __bfloat1622float2(h1[i]);
    f[2 * i] = a.x; f[2 * i + 1] = a.y; f[8 + 2 * i] = b.x; f[8 + 2 * i + 1] = b.y;
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) ss += f[i] * f[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) red[tid >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float rr = normalize ? rsqrtf(tot / 4096.f + 1e-5f) : 1.f;
  uint32_t outw[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 gg = normalize ? g1p4(gw + tid * 16 + 4 * i) : make_float4(1.f, 1.f, 1.f, 1.f);
    __nv_bfloat162 a = __floats2bfloat162_rn(f[4 * i] * rr * gg.x, f[4 * i + 1] * rr * gg.y);
    __nv_bfloat162 b = __floats2bfloat162_rn(f[4 * i + 2] * rr * gg.z, f[4 * i + 3] * rr * gg.w);
    outw[2 * i] = bf162_u32(a);
    outw[2 * i + 1] = bf162_u32(b);
  }
  const int stg = tid >> 2, ch0 = 2 * (tid & 3);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int n = mn[k];  // (written before the reduction's __syncthreads)
    if (n == 0) continue;
    const int lr = mlr[k];
    uint8_t* rowp = xt + mrow[k] + (size_t)stg * n * 128;
    const int x = lr & 7;
    *reinterpret_cast<uint4*>(rowp + ((ch0 ^ x) << 4)) = make_uint4(outw[0], outw[1], outw[2], outw[3]);
    *reinterpret_cast<uint4*>(rowp + (((ch0 + 1) ^ x) << 4)) = make_uint4(outw[4], outw[5], outw[6], outw[7]);
  }
}

// ---- routing: top-8 experts per token from the router logits (hi + lo halves), one warp per token ----
// Same math and tie-break as the Triton reference version (sigmoid(logit / sqrt(32)), + bias for selection, the
// highest biased score first, lowest expert index on ties; weights = selected scores / their sum), but each lane
// holds 8 experts and the 8 rounds of arg-max are warp shuffles (no block barriers). Local routes (experts
// local_start .. +31) get a row position from a per-expert counter: positions are first taken from block-local
// shared-memory counters, then each block reserves its range with one global atomic per expert (hot experts take
// thousands of routes, and per-route global atomics on one address serialize).
#define ROUTE_WARPS 16
__global__ void __launch_bounds__(ROUTE_WARPS * 32) moe_route_kernel(const float* __restrict__ logits,
                                                                     const float* __restrict__ bias,
                                                                     const int* __restrict__ lstart, int* __restrict__ ids,
                                                                     float* __restrict__ wout, int* __restrict__ lid,
                                                                     int* __restrict__ pos, int* __restrict__ cnt, int T,
                                                                     int* __restrict__ tok_left = nullptr, int* __restrict__ cq = nullptr) {
  pdl_enter();
  __shared__ int s_cnt[32], s_base[32];
  const int t = blockIdx.x * ROUTE_WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (threadIdx.x < 32) s_cnt[threadIdx.x] = 0;
  __syncthreads();
  int e_mine = 0, loc = 0, lpos = 0;
  bool local = false;
  float w_mine = 0.f;
  if (t < T) {
    const float* lg = logits + (size_t)t * 512;
    float sv[8], bv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = lane + 32 * j;
      const float x = (lg[e] + lg[256 + e]) * 0.1767766952966369f;
      sv[j] = 1.f / (1.f + expf(-x));
      bv[j] = sv[j] + bias[e];
    }
    int sel_e[8];
    float sel_s[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bvv = bv[0];
      int be = lane;
#pragma unroll
      for (int j = 1; j < 8; ++j) {  // ascending expert index within the lane: strict > keeps the lowest on ties
        if (bv[j] > bvv) { bvv = bv[j]; be = lane + 32 * j; }
      }
      {
        const uint32_t u = __float_as_uint(bvv);
        const uint32_t key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // unsigned order == float order
        const uint32_t mk = __reduce_max_sync(0xffffffffu, key);
        be = (int)__reduce_min_sync(0xffffffffu, key == mk ? (uint32_t)be : 0xffffffffu);  // lowest expert on ties
      }
      float mys = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j == (be >> 5)) mys = sv[j];
      }
      sel_s[k] = __shfl_sync(0xffffffffu, mys, be & 31);
      sel_e[k] = be;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (lane + 32 * j == be) bv[j] = -INFINITY;
      }
    }
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) sum += sel_s[k];
    e_mine = sel_e[0];
    float sk = sel_s[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) {
      if (lane == k) { e_mine = sel_e[k]; sk = sel_s[k]; }
    }
    w_mine = sk / (sum + 1e-15f);
    loc = (e_mine - *lstart + 256) % 256;
    local = lane < 8 && loc < 32;
    if (local) lpos = atomicAdd(&s_cnt[loc], 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int c = s_cnt[threadIdx.x];
    s_base[threadIdx.x] = c ? atomicAdd(cnt + threadIdx.x, c) : 0;
  }
  __syncthreads();
  if (t < T && tok_left != nullptr) {  // CIT: the token's local route count, and its combine-queue slot cleared
    const unsigned lm = __ballot_sync(0xffffffffu, local);
    if (lane == 0) {
      tok_left[t] = __popc(lm);
      cq[t] = 0;
    }
  }
  if (t < T && lane < 8) {
    ids[t * 8 + lane] = e_mine;
    wout[t * 8 + lane] = w_mine;
    lid[t * 8 + lane] = local ? loc : -1;
    pos[t * 8 + lane] = local ? s_base[loc] + lpos : 0;
  }
}

// ---- combine: out[t] = sum over t's local routes of norm(y[row] * pw) * w, one block of 256 threads per token ----
// (the post-expert norm: scale by pw = 1 + w first, then RMSNorm with eps 1e-8). The token's local routes (about two of
// its eight) are compacted first; their sums of squares share one block reduction, and the second pass re-reads each
// route row from L1/L2. Few registers, so several blocks per SM keep enough loads in flight.
__global__ void __launch_bounds__(256) moe_combine_kernel(const __nv_bfloat16* __restrict__ y, const int* __restrict__ rrow,
                                                          const int* __restrict__ lid, const float* __restrict__ topw,
                                                          const __nv_bfloat16* __restrict__ pw, __nv_bfloat16* __restrict__ out) {
  // one token per block, 16 elements per thread: each local route's row post-normed on its own (rsqrt of its mean
  // square + 1e-8, times g_post_expert_norm), weighted by its routing weight, summed (as the Triton _combine_kernel);
  // up to 4 routes' rows in flight at once, one block reduction per group of 4
  pdl_enter();
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[4][8];
  int rows[8];
  float wts[8];
  int nl = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int l = lid[tok * 8 + k];
    if (l >= 0) {
      rows[nl] = rrow[tok * 8 + k];
      wts[nl] = topw[tok * 8 + k];
      ++nl;
    }
  }
  float g[16];
  {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 q = g1p4(pw + tid * 16 + 4 * i);
      g[4 * i] = q.x; g[4 * i + 1] = q.y; g[4 * i + 2] = q.z; g[4 * i + 3] = q.w;
    }
  }
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#define COMB_GROUP 2
  for (int k0 = 0; k0 < nl; k0 += COMB_GROUP) {
    float x[COMB_GROUP][16];
    float ss[COMB_GROUP];
#pragma unroll
    for (int u = 0; u < COMB_GROUP; ++u) {
      ss[u] = 0.f;
      if (k0 + u < nl) {
        const uint4* src = reinterpret_cast<const uint4*>(y + (size_t)rows[k0 + u] * 4096 + tid * 16);
        const uint4 v0 = src[0], v1 = src[1];
        const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
        const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 f = __bfloat1622float2(h0[i]), q = __bfloat1622float2(h1[i]);
          x[u][2 * i] = f.x * g[2 * i]; x[u][2 * i + 1] = f.y * g[2 * i + 1];
          x[u][8 + 2 * i] = q.x * g[8 + 2 * i]; x[u][8 + 2 * i + 1] = q.y * g[8 + 2 * i + 1];
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) ss[u] += x[u][i] * x[u][i];
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) x[u][i] = 0.f;
      }
    }
#pragma unroll
    for (int u = 0; u < COMB_GROUP; ++u) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss[u] += __shfl_xor_sync(0xffffffffu, ss[u], o);
    }
    if (k0) __syncthreads();  // the previous group's red[] has been read
    if (lane == 0) {
#pragma unroll
      for (int u = 0; u < COMB_GROUP; ++u) red[u][warp] = ss[u];
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < COMB_GROUP; ++u) {
      if (k0 + u < nl) {
        float tot = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) tot += red[u][w];
        const float sc = rsqrtf(tot / 4096.f + 1e-8f) * wts[k0 + u];
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] += x[u][i] * sc;
      }
    }
  }
  uint4 o0, o1;
  __nv_bfloat162* b0 = reinterpret_cast<__nv_bfloat162*>(&o0);
  __nv_bfloat162* b1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    b0[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    b1[i] = __floats2bfloat162_rn(acc[8 + 2 * i], acc[8 + 2 * i + 1]);
  }
  uint4* dst = reinterpret_cast<uint4*>(out + (size_t)tok * 4096 + tid * 16);
  dst[0] = o0;
  dst[1] = o1;
}

}  // namespace okc_moe

#include <c10/util/Exception.h>

#include <cuda_runtime.h>
using namespace okc_moe;
template <typename... KArgs, typename... Args>
static inline void okc_launch(void (*k)(KArgs...), dim3 g, dim3 b, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, k, ((KArgs)args)...);
  TORCH_CHECK(err == cudaSuccess, "Muse Spark MoE kernel launch failed: ", cudaGetErrorString(err));
}
void k_moe_plan(const int* counts, int* offsets, int* tiles, int* ntiles, int* rows, int order, cudaStream_t st) {
  okc_launch(moe_plan_kernel, dim3(1), dim3(32), 0, st, counts, offsets, (int4*)tiles, ntiles, rows, order);
}
void k_moe_gather(const void* p, const void* gw, const int* lid, const int* pos, const int* offsets, const int* counts,
                  void* xt, int* rrow, int T, cudaStream_t st, bool normalize) {
  okc_launch(moe_gather_kernel<false>, dim3(T + 32 * MOE_PADFILL_PARTS), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, offsets, counts, (uint8_t*)xt, rrow, T, (int*)nullptr, (int4*)nullptr, (int*)nullptr, (int*)nullptr, 0,
             (int*)nullptr, (int*)nullptr, (int*)nullptr, normalize);
}

void k_moe_gather_plan(const void* p, const void* gw, const int* lid, const int* pos, const int* counts, void* xt, int* rrow,
                       int* offsets, int* tiles, int* ntiles, int* rows, int order, int T, cudaStream_t st,
                       int* row_tok, int* cq, int* qtail, bool normalize) {
  okc_launch(moe_gather_kernel<true>, dim3(T + 32 * MOE_PADFILL_PARTS), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, (const int*)nullptr, counts, (uint8_t*)xt, rrow, T,
             offsets, (int4*)tiles, ntiles, rows, order, row_tok, cq, qtail, normalize);
}
void k_moe_combine(const void* y, const int* rrow, const int* lid, const float* topw, const void* pw, void* out, int T,
                   cudaStream_t st) {
  okc_launch(moe_combine_kernel, dim3(T), dim3(256), 0, st, (const __nv_bfloat16*)y, rrow, lid, topw, (const __nv_bfloat16*)pw, (__nv_bfloat16*)out);
}

static Args make_args(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out,
                      int NT, int smem, int* counter) {
  Args a;
  a.wrep = (const uint8_t*)wrep; a.xt = (const uint8_t*)xt; a.tiles = (const int4*)tiles; a.ntiles = ntiles;
  a.gscale = gscale; a.out = (uint8_t*)out; a.NT = NT; a.smem_bytes = smem; a.counter = counter; a.flags = 0; a.mul16 = 16u; a.mul64 = 64u;
  return a;
}

void k_moe_fused(const void* w13, const void* w2, const void* xt, void* inter, void* y, const int* tiles, const int* ntiles,
                 const float* gs13, const float* gs2, int nctas, int smem, int* counter, int* done, void* dbg, int flags,
                 cudaStream_t st, int* cit, int* qctl, int* tok_left, int* cq, const int* row_tok, const int* rrow,
                 const int* lid, const float* topw, const void* pw, void* out, int T) {
  FusedArgs fa;
  fa.cit = cit; fa.qctl = qctl; fa.tok_left = tok_left; fa.cq = cq; fa.row_tok = row_tok; fa.rrow = rrow; fa.lid = lid;
  fa.topw = topw; fa.pw = (const __nv_bfloat16*)pw; fa.out = (__nv_bfloat16*)out; fa.T = T;
  fa.dbg = (unsigned long long*)dbg;
  fa.flags = flags;
  fa.g1 = make_args(w13, xt, tiles, ntiles, gs13, inter, 64, smem, counter);
  fa.g2 = make_args(w2, inter, tiles, ntiles, gs2, y, 32, smem, counter);
  fa.g1.flags = flags;
  fa.g2.flags = flags;
  fa.done = done;
  if ((fa.flags >> 8) & 1)  // bit 8: fine remainder tiles (the plan used the same flag)
    okc_launch(moe_fused_persistent<1>, dim3((unsigned)nctas), dim3(MOE_FUSED_THREADS), (size_t)smem, st, fa);
  else
    okc_launch(moe_fused_persistent<0>, dim3((unsigned)nctas), dim3(MOE_FUSED_THREADS), (size_t)smem, st, fa);
}

void k_moe_init(int smem) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused_persistent<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused_persistent<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_gemm_persistent<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_gemm_persistent<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_gemm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
}
