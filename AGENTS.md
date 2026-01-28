# AGENTS.md - CUDA Study Repository Guide

## Project Overview

Educational repository for CUDA parallel programming using PyTorch's JIT compilation pipeline. All CUDA kernels are embedded as strings and compiled at runtime via `torch.utils.cpp_extension.load_inline`.

**Tech Stack**: Python 3.12, PyTorch 2.11+ (nightly), CUDA 13.1, GCC

---

## Environment Setup (WSL2)

### Hardware & Software Info
- **OS**: Windows 11 + WSL2 (Ubuntu 22.04)
- **GPU**: NVIDIA GeForce RTX 5070 Ti (Blackwell, Compute Capability 12.0)
- **Driver**: 591.44 (Windows) / 590.44.01 (WSL)
- **CUDA Toolkit**: 13.1 (required for Blackwell architecture)
- **PyTorch**: 2.11.0.dev (nightly, cu128)
- **Python**: 3.12.3

### Path Mapping
| Windows | WSL |
|---------|-----|
| `D:\develop\cuda_study` | `/mnt/d/develop/cuda_study` |
| N/A | `~/cuda_env` (Python virtual environment) |

### Initial Setup (One-time)

#### 1. Create Python Virtual Environment
```bash
# In WSL Ubuntu
cd ~
python3 -m venv cuda_env
source ~/cuda_env/bin/activate
```

#### 2. Install CUDA Toolkit 13.1
```bash
# Download and install CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-1
```

#### 3. Configure PATH (Add to ~/.bashrc)
```bash
# IMPORTANT: CUDA 13.1 must come FIRST in PATH
echo 'export PATH=/usr/local/cuda-13.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 4. Install PyTorch (Nightly with CUDA 12.8 support)
```bash
source ~/cuda_env/bin/activate
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 5. Verify Installation
```bash
nvcc --version  # Should show 13.1.x
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

---

## Running Scripts (WSL2)

### Quick Start
```bash
# 1. Open WSL
wsl -d Ubuntu

# 2. Navigate to project
cd /mnt/d/develop/cuda_study

# 3. Activate virtual environment
source ~/cuda_env/bin/activate

# 4. Run script
python lecture5/lecture5_review_week1.py
```

### Full PATH Setup (If Commands Not Found)
If you see "command not found" errors, reset PATH:
```bash
export PATH=/usr/local/cuda-13.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
source ~/cuda_env/bin/activate
```

### Clear JIT Cache (When Changing CUDA Version)
```bash
rm -rf ~/.cache/torch_extensions
```

---

## Troubleshooting

### Error: "Unsupported gpu architecture 'compute_120'"
**Cause**: CUDA Toolkit version doesn't support Blackwell (compute_120)
**Solution**: 
1. Install CUDA Toolkit 13.1+
2. Update PATH to use `/usr/local/cuda-13.1/bin/nvcc`
3. Clear cache: `rm -rf ~/.cache/torch_extensions`

### Error: "command not found" (rm, python, etc.)
**Cause**: PATH was overwritten incorrectly
**Solution**: Reset PATH:
```bash
export PATH=/usr/local/cuda-13.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
```

### Error: "No module named 'torch'"
**Cause**: Virtual environment not activated
**Solution**: `source ~/cuda_env/bin/activate`

### Error: "CUDA error: no kernel image is available"
**Cause**: Compiled for wrong GPU architecture
**Solution**: 
1. Ensure CUDA 13.1 nvcc is being used: `which nvcc` should show `/usr/local/cuda-13.1/bin/nvcc`
2. Clear cache: `rm -rf ~/.cache/torch_extensions`
3. Re-run script

### PyTorch CUDA 13.1 Not Available
**Cause**: Official PyTorch doesn't have cu131 yet
**Solution**: Use nightly with cu128:
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Build / Run / Test Commands

### Running Scripts (No Build Step Required)
```bash
# Run any lecture script directly - JIT compilation happens automatically
python lecture1/lecture1_basic.py
python lecture2/lecture2_hw_conv1d.py
python lecture3/lecture3_matmul_tiled.py

# Run Week 5 review scripts
python lecture5/lecture5_review_week1.py
python lecture5/lecture5_review_week3_matmul.py
python lecture5/lecture5_capstone_conv2d.py
```

### Environment Variables (Set in Python scripts)
```python
import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
# TORCH_CUDA_ARCH_LIST is auto-detected for RTX 5070 Ti
```

### Profiling
```bash
# NVIDIA Nsight Compute profiling
ncu --set roofline python lecture4/lecture4_ncu_profiling.py

# NVIDIA Nsight Systems timeline
nsys profile python lecture4/lecture4_nsys_timeline.py
```

---

## Code Style Guidelines

### File Structure Pattern
Every Python file follows this structure:
```python
"""
Module docstring with:
- Week/Lecture number and topic
- Learning objectives
- Key concepts explained
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

# Compiler settings (REQUIRED - must be before load_inline)
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

# Constants
TILE_SIZE = 16

# CUDA kernel source (embedded C++)
cuda_source = """..."""

# C++ header declarations
cpp_header = """..."""

# JIT compilation
module = load_inline(
    name='extension_name',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['function_names'],
    verbose=True
)

# Python wrapper functions
def kernel_wrapper(...): ...

# Verification functions
def verify_correctness(...): ...

# Benchmark functions  
def benchmark(...): ...

if __name__ == "__main__":
    # Main execution with verification and benchmarks
```

### Import Order
1. `import os` (first - for compiler settings)
2. Standard library imports (`math`, `subprocess`, etc.)
3. `import torch`
4. `from torch.utils.cpp_extension import load_inline`

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| CUDA Kernels | `snake_case_kernel` | `matmul_tiled_kernel` |
| Launcher Functions | `launch_*` or just function name | `launch_matmul_tiled` |
| Python Wrappers | `snake_case` | `matmul_tiled()` |
| Constants | `UPPER_SNAKE_CASE` | `TILE_SIZE`, `BLOCK_SIZE` |
| Test Functions | `verify_*` or `test_*` | `verify_correctness()` |
| Benchmark Functions | `benchmark*` | `benchmark_transpose_methods()` |

### Type Hints
Always use type hints for Python functions:
```python
def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division"""
    return (n + divisor - 1) // divisor

def conv1d_cuda(
    x: torch.Tensor,
    threads_per_block: int = 256,
    blocks_per_grid: int = None
) -> torch.Tensor:
```

### Input Validation Pattern
```python
def kernel_wrapper(input_tensor: torch.Tensor) -> torch.Tensor:
    assert input_tensor.dim() == 2, f"Expected 2D tensor, got {input_tensor.shape}"
    assert input_tensor.is_cuda, "Input tensor must be on CUDA device"
    assert input_tensor.dtype == torch.float32, "Input must be float32"
    input_tensor = input_tensor.contiguous()  # Ensure memory contiguity
    # ... rest of implementation
```

---

## CUDA Kernel Style

### Kernel Signature Pattern
```cpp
__global__ void kernel_name(
    const float* __restrict__ input,   // Use const + __restrict__ for inputs
    float* __restrict__ output,        // Use __restrict__ for outputs
    int M, int N                       // Dimensions as ints
) {
    // Implementation
}
```

### Shared Memory Declaration
```cpp
// Basic shared memory
__shared__ float tile[TILE_SIZE][TILE_SIZE];

// Bank conflict avoidance (add +1 padding)
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
```

### Thread Indexing
```cpp
// 1D indexing
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 2D indexing
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Grid-stride loop (preferred for scalability)
int stride = blockDim.x * gridDim.x;
for (int i = index; i < n; i += stride) {
    // Process element i
}
```

### Boundary Checks
```cpp
// Always check bounds before memory access
if (row < M && col < N) {
    output[row * N + col] = value;
}
```

---

## Documentation Style

- Use Korean for explanatory comments and docstrings
- Use emoji markers for visual organization:
  - Learning objectives
  - Code sections
  - Warnings/Important notes
  - Success/Failure indicators
- Use `=` separators (60-80 chars) for major sections:
```python
print("=" * 60)
print("Section Title")
print("=" * 60)
```

---

## Testing & Verification

### Verification Against PyTorch
```python
def verify_correctness(...):
    result_cuda = cuda_kernel(input_tensor)
    result_pytorch = pytorch_reference(input_tensor)
    
    is_correct = torch.allclose(result_cuda, result_pytorch, rtol=1e-5, atol=1e-5)
    max_diff = (result_cuda - result_pytorch).abs().max().item()
    
    if is_correct:
        print(f"Verification passed! (max error: {max_diff:.2e})")
    else:
        print(f"Verification FAILED! (max error: {max_diff:.2e})")
```

### Benchmarking Pattern
```python
def benchmark(n: int, iterations: int = 100):
    # Warm-up (always do this!)
    for _ in range(10):
        _ = kernel(input_tensor)
    torch.cuda.synchronize()
    
    # Timing with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        result = kernel(input_tensor)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iterations
    print(f"Average time: {time_ms:.4f} ms")
```

---

## Common Patterns

### Ceiling Division Helper
```python
def cdiv(n: int, divisor: int) -> int:
    return (n + divisor - 1) // divisor
```

### Grid/Block Configuration
```python
# 1D kernels
threads_per_block = 256
blocks_per_grid = cdiv(n, threads_per_block)

# 2D kernels  
block = (16, 16)  # or (32, 32)
grid = (cdiv(width, block[0]), cdiv(height, block[1]))

# Grid-stride loop (fixed grid size)
sm_count = torch.cuda.get_device_properties(0).multi_processor_count
blocks_per_grid = sm_count * 4  # Typical multiplier: 2-4
```

### GFLOPS Calculation
```python
def calculate_gflops(M: int, N: int, K: int, time_ms: float) -> float:
    total_ops = 2 * M * N * K  # For matmul: 2 ops per element (mul + add)
    time_seconds = time_ms / 1000.0
    return total_ops / time_seconds / 1e9
```

---

## Anti-Patterns to Avoid

1. **Never suppress CUDA errors** - Always check return values
2. **Never forget `torch.cuda.synchronize()`** before timing
3. **Never hardcode block sizes** - Use constants or calculated values
4. **Never ignore boundary checks** - Can cause silent corruption
5. **Never use `as any` type suppressions** - This is Python, but same principle applies

---

## Project Structure
```
cuda_study/
├── lecture1/           # Basics: Hello World, JIT compilation
├── lecture2/           # Architecture: Warps, Grid-Stride loops  
├── lecture3/           # Memory: Shared memory, Tiling, Coalescing
├── lecture4/           # Profiling: NCU, Nsight Systems
├── lecture5/           # Week 5: Hands-on Review (Week 1-4)
│   ├── lecture5.md                         # Week 5 overview and guide
│   ├── lecture5_review_week1.py            # Vector operations (Grid-Stride Loop)
│   ├── lecture5_review_week1_2d.py         # 2D image processing kernels
│   ├── lecture5_review_week3_matmul.py     # Matrix multiplication optimization
│   ├── lecture5_review_week3_transpose.py  # Transpose optimization (Coalescing)
│   ├── lecture5_review_week4_detective.py  # Performance debugging scenarios
│   ├── lecture5_review_week4_roofline.py   # Roofline model analysis
│   └── lecture5_capstone_conv2d.py         # 2D Convolution capstone project
├── AGENTS.md           # This file
└── README.md           # Project overview (Korean)
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start WSL | `wsl -d Ubuntu` |
| Navigate to project | `cd /mnt/d/develop/cuda_study` |
| Activate venv | `source ~/cuda_env/bin/activate` |
| Run script | `python lectureN/script.py` |
| Check nvcc version | `nvcc --version` |
| Check GPU | `nvidia-smi` |
| Clear JIT cache | `rm -rf ~/.cache/torch_extensions` |
| Profile with NCU | `ncu python script.py` |
| Profile timeline | `nsys profile python script.py` |

---

## RTX 5070 Ti (Blackwell) Specific Notes

### Compute Capability
- RTX 5070 Ti uses **Compute Capability 12.0** (Blackwell architecture)
- Requires **CUDA Toolkit 13.1+** for native support
- CUDA 12.x will fail with "Unsupported gpu architecture 'compute_120'"

### Performance Characteristics
- 16GB GDDR7 Memory
- ~300W TDP
- SM Count: Check with `torch.cuda.get_device_properties(0).multi_processor_count`

### Known Issues
1. PyTorch official releases may not support cu131 yet - use nightly builds
2. JIT compilation auto-detects compute_120, ensure CUDA 13.1 nvcc is in PATH
3. Some older CUDA samples may not compile without modification
