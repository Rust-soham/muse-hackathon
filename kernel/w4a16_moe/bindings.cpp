#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
void k_moe_plan(const int* counts, int* offsets, int* tiles, int* ntiles, int* rows, int order, cudaStream_t st);
void k_moe_gather(const void* p, const void* gw, const int* lid, const int* pos, const int* offsets, const int* counts,
                  void* xt, int* rrow, int T, cudaStream_t st, bool normalize);
void k_moe_gather_plan(const void* p, const void* gw, const int* lid, const int* pos, const int* counts, void* xt, int* rrow,
                       int* offsets, int* tiles, int* ntiles, int* rows, int order, int T, cudaStream_t st,
                       int* row_tok, int* cq, int* qtail, bool normalize);
void k_moe_combine(const void* y, const int* rrow, const int* lid, const float* topw, const void* pw, void* out, int T,
                   cudaStream_t st);
void k_moe_fused(const void* w13, const void* w2, const void* xt, void* inter, void* y, const int* tiles, const int* ntiles,
                 const float* gs13, const float* gs2, int nctas, int smem, int* counter, int* done, void* dbg, int flags,
                 cudaStream_t st, int* cit, int* qctl, int* tok_left, int* cq, const int* row_tok, const int* rrow,
                 const int* lid, const float* topw, const void* pw, void* out, int T);
void k_moe_init(int smem);
static cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream(); }

void moe_plan(torch::Tensor counts, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor rows,
              int64_t order) {
  k_moe_plan(counts.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
             (int)order, cur());
}

void moe_gather(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor offsets,
                torch::Tensor counts, torch::Tensor xt, torch::Tensor rrow, bool normalize) {
  k_moe_gather(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), offsets.data_ptr<int>(),
               counts.data_ptr<int>(), xt.data_ptr(), rrow.data_ptr<int>(), (int)p.size(0), cur(), normalize);
}

void moe_gather_plan(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor counts,
                     torch::Tensor xt, torch::Tensor rrow, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles,
                     torch::Tensor rows, int64_t order, torch::Tensor cit, torch::Tensor tok, bool normalize) {
  // CIT (cit.numel() > 0): queue tail = cit[numel - 3]; tok = [tok_left (T) | cq (T) | row_tok]
  const bool on = cit.numel() > 0;
  const int T = (int)p.size(0);
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_gather_plan(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), counts.data_ptr<int>(), xt.data_ptr(),
                    rrow.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
                    (int)order, T, cur(), on ? tk + 2 * T : nullptr, on ? tk + T : nullptr,
                    on ? cit.data_ptr<int>() + (cit.numel() - 3) : nullptr, normalize);
}

void moe_combine(torch::Tensor y, torch::Tensor rrow, torch::Tensor lid, torch::Tensor topw, torch::Tensor pw, torch::Tensor out) {
  k_moe_combine(y.data_ptr(), rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), pw.data_ptr(),
                out.data_ptr(), (int)out.size(0), cur());
}

void moe_fused(torch::Tensor w13, torch::Tensor w2, torch::Tensor xt, torch::Tensor inter, torch::Tensor y, torch::Tensor tiles,
               torch::Tensor ntiles, torch::Tensor gs13, torch::Tensor gs2, int64_t nctas, int64_t smem, torch::Tensor counter,
               torch::Tensor done, torch::Tensor dbg, int64_t flags, torch::Tensor cit, torch::Tensor tok, torch::Tensor rrow,
               torch::Tensor lid, torch::Tensor topw, torch::Tensor pw, torch::Tensor out) {
  // CIT (cit.numel() > 0): cit = [max tiles GEMM2 counters | head, tail, error, pad]; tok = [tok_left (T) | cq (T) | row_tok]
  const bool on = cit.numel() > 0;
  const int T = (int)rrow.size(0);
  int* c = on ? cit.data_ptr<int>() : nullptr;
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_fused(w13.data_ptr(), w2.data_ptr(), xt.data_ptr(), inter.data_ptr(), y.data_ptr(), tiles.data_ptr<int>(),
              ntiles.data_ptr<int>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), (int)nctas, (int)smem, counter.data_ptr<int>(),
              done.data_ptr<int>(), dbg.numel() ? dbg.data_ptr() : nullptr, (int)flags, cur(),
              c, on ? c + (cit.numel() - 4) : nullptr, tk, on ? tk + T : nullptr, on ? tk + 2 * T : nullptr,
              rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), pw.data_ptr(), on ? out.data_ptr() : nullptr, T);
}

void moe_init(int64_t smem) { k_moe_init((int)smem); }
