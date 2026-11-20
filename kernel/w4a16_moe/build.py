"""One extension build per process, with persistent torch extension cache."""

import fcntl
from functools import lru_cache
from pathlib import Path

_NAME = "muse_w4a16_moe_v1"
_FUNCTIONS = ["moe_plan", "moe_gather", "moe_gather_plan", "moe_combine", "moe_fused", "moe_init"]


@lru_cache(maxsize=1)
def _module():
    """Compile once at weight load (never inside graph capture) and set the kernels' shared memory.

    load_inline writes its sources before taking torch's build lock, so on a cold cache concurrent
    TP ranks can truncate a file another rank is compiling; an exclusive lock serializes the build.
    """
    from torch.utils.cpp_extension import _get_build_directory, load_inline

    src = Path(__file__).parent
    build_dir = Path(_get_build_directory(_NAME, verbose=False))
    build_dir.mkdir(parents=True, exist_ok=True)
    with open(build_dir.parent / f"{_NAME}.build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            mod = load_inline(
                name=_NAME,
                cpp_sources=(src / "bindings.cpp").read_text(),
                cuda_sources=(src / "moe.cu").read_text(),
                functions=_FUNCTIONS,
                extra_cflags=["-O3", "-std=c++20"],
                # Fast math as in stock SGLang's JIT kernels (e.g. kernels/ops/activation).
                extra_cuda_cflags=[
                    "--use_fast_math",
                    "-O3",
                    "-std=c++20",
                    "-gencode=arch=compute_90a,code=sm_90a",
                ],
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    mod.moe_init(227 * 1024)
    return mod
