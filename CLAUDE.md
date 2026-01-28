# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational CUDA parallel programming repository. All CUDA kernels are embedded as C++ strings in Python files and JIT-compiled at runtime via `torch.utils.cpp_extension.load_inline`. There is no build system — just run Python scripts directly.

**Stack**: Python 3.12, PyTorch 2.11+ (nightly cu128), CUDA 13.1, GCC, WSL2 (Ubuntu 22.04)
**GPU**: NVIDIA RTX 5070 Ti (Blackwell, Compute Capability 12.0)

## Running Scripts

All execution happens in WSL2:

```bash
wsl -d Ubuntu
cd /mnt/d/develop/cuda_study
source ~/cuda_env/bin/activate
python lecture5/lecture5_review_week1.py
```

PATH must have CUDA 13.1 first: `/usr/local/cuda-13.1/bin`

Clear JIT cache when changing CUDA versions: `rm -rf ~/.cache/torch_extensions`

Profiling: `ncu --set roofline python script.py` or `nsys profile python script.py`

## Architecture

Each `lectureN/` directory corresponds to a week in a 15-week curriculum. Every Python file is self-contained and follows this pattern:

1. Set `CC`/`CXX` env vars to `gcc`/`g++`
2. Define CUDA kernel as a string (`cuda_source`)
3. Define C++ header declarations (`cpp_header`)
4. JIT compile with `load_inline(name=..., cpp_sources=[cpp_header], cuda_sources=[cuda_source], functions=[...])`
5. Python wrapper functions → verification against PyTorch reference → benchmarks

Curriculum progression: basics (L1) → GPU architecture/warps (L2) → shared memory/tiling (L3) → profiling (L4) → hands-on review (L5)

## Code Conventions

- Documentation language: Korean
- CUDA kernels: `snake_case_kernel`, use `const float* __restrict__` for inputs, always check bounds
- Python functions: type hints required, use `cdiv(n, d)` for ceiling division
- Constants: `UPPER_SNAKE_CASE` (e.g., `TILE_SIZE`, `BLOCK_SIZE`)
- Verification: compare CUDA output against PyTorch with `torch.allclose(rtol=1e-5, atol=1e-5)`
- Benchmarking: always warm up GPU, use `torch.cuda.Event` for timing, call `torch.cuda.synchronize()` before measurements
- Shared memory bank conflict avoidance: pad with `+1` (e.g., `tile[TILE_SIZE][TILE_SIZE + 1]`)
- Input tensors: assert correct dim/dtype/device, call `.contiguous()` before passing to kernel

## Blackwell-Specific Notes

- Requires CUDA Toolkit 13.1+ (CUDA 12.x fails with "Unsupported gpu architecture 'compute_120'")
- PyTorch official releases may lack cu131 — use nightly builds with cu128
- `TORCH_CUDA_ARCH_LIST` is auto-detected
