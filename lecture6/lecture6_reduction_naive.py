"""
Week 6: Parallel Reduction 기초
Naive → Sequential Addressing → First-add-during-load 최적화

목표:
1. Interleaved Addressing (divergent branch) 이해
2. Sequential Addressing (bank conflict 회피) 이해
3. First-add-during-load (idle thread 제거) 이해
4. Sum / Max / Min reduction 구현
5. 각 단계별 성능 비교
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

BLOCK_SIZE = 256


def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division"""
    return (n + divisor - 1) // divisor


# =============================================================================
# CUDA Kernel 소스코드
# =============================================================================

cuda_source = """
#include <torch/extension.h>
#include <cfloat>

#define BLOCK_SIZE 256

// =============================================================================
// Kernel 1: Interleaved Addressing (Divergent Branch)
// stride가 1 → 2 → 4 → ... 로 증가
// 문제: warp divergence가 심함 (if 조건이 thread마다 다름)
// =============================================================================
__global__ void reduce_interleaved_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Global memory → Shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Interleaved addressing: stride 1, 2, 4, ...
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// Kernel 2: Sequential Addressing
// stride가 blockDim/2 → blockDim/4 → ... 로 감소
// 개선: warp divergence 감소, bank conflict 회피
// =============================================================================
__global__ void reduce_sequential_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Sequential addressing: stride blockDim/2, blockDim/4, ...
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// Kernel 3: First-add-during-load
// 로드할 때 2개 원소를 미리 합산 → idle thread 절반 제거
// 블록 수를 절반으로 줄여 사용
// =============================================================================
__global__ void reduce_first_add_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    // 각 블록이 2 * BLOCK_SIZE 원소를 담당
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 로드 시 2개를 합산
    float val = 0.0f;
    if (idx < n) val += input[idx];
    if (idx + blockDim.x < n) val += input[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Sequential addressing reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// Max Reduction (Sequential Addressing)
// =============================================================================
__global__ void reduce_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = -FLT_MAX;
    if (idx < n) val = input[idx];
    if (idx + blockDim.x < n) val = fmaxf(val, input[idx + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// Min Reduction (Sequential Addressing)
// =============================================================================
__global__ void reduce_min_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = FLT_MAX;
    if (idx < n) val = input[idx];
    if (idx + blockDim.x < n) val = fminf(val, input[idx + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// Launch 함수들 - output 텐서를 Python에서 전달받음
// =============================================================================

void launch_reduce_interleaved(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_interleaved_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_reduce_sequential(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_sequential_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_reduce_first_add(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_first_add_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_reduce_max(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_max_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_reduce_min(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_min_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_header = """
void launch_reduce_interleaved(torch::Tensor input, torch::Tensor output, int n, int blocks);
void launch_reduce_sequential(torch::Tensor input, torch::Tensor output, int n, int blocks);
void launch_reduce_first_add(torch::Tensor input, torch::Tensor output, int n, int blocks);
void launch_reduce_max(torch::Tensor input, torch::Tensor output, int n, int blocks);
void launch_reduce_min(torch::Tensor input, torch::Tensor output, int n, int blocks);
"""

# JIT 컴파일
print("=" * 60)
print("Compiling Reduction Kernels (Naive)...")
print("=" * 60)

module = load_inline(
    name='reduction_naive_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_reduce_interleaved', 'launch_reduce_sequential',
               'launch_reduce_first_add', 'launch_reduce_max', 'launch_reduce_min'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# Python Wrapper 함수들
# =============================================================================

def reduce_sum_interleaved(x: torch.Tensor) -> float:
    """Interleaved Addressing Sum Reduction"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = cdiv(n, BLOCK_SIZE)
    output = torch.zeros(blocks, dtype=torch.float32, device='cuda')
    module.launch_reduce_interleaved(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.sum().item()


def reduce_sum_sequential(x: torch.Tensor) -> float:
    """Sequential Addressing Sum Reduction"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = cdiv(n, BLOCK_SIZE)
    output = torch.zeros(blocks, dtype=torch.float32, device='cuda')
    module.launch_reduce_sequential(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.sum().item()


def reduce_sum_first_add(x: torch.Tensor) -> float:
    """First-add-during-load Sum Reduction"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = cdiv(n, BLOCK_SIZE * 2)
    output = torch.zeros(blocks, dtype=torch.float32, device='cuda')
    module.launch_reduce_first_add(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.sum().item()


def reduce_max_cuda(x: torch.Tensor) -> float:
    """Max Reduction"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = cdiv(n, BLOCK_SIZE * 2)
    output = torch.full((blocks,), float('-inf'), dtype=torch.float32, device='cuda')
    module.launch_reduce_max(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.max().item()


def reduce_min_cuda(x: torch.Tensor) -> float:
    """Min Reduction"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = cdiv(n, BLOCK_SIZE * 2)
    output = torch.full((blocks,), float('inf'), dtype=torch.float32, device='cuda')
    module.launch_reduce_min(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.min().item()


# =============================================================================
# 검증
# =============================================================================

def verify_reductions():
    print("=" * 60)
    print("Reduction 검증")
    print("=" * 60)

    n = 1_000_003  # 비정렬 크기로 테스트
    x = torch.randn(n, dtype=torch.float32, device='cuda')

    ref_sum = x.sum().item()
    ref_max = x.max().item()
    ref_min = x.min().item()

    # Sum reductions
    for name, fn in [("Interleaved", reduce_sum_interleaved),
                     ("Sequential", reduce_sum_sequential),
                     ("First-add", reduce_sum_first_add)]:
        result = fn(x)
        error = abs(result - ref_sum) / (abs(ref_sum) + 1e-8)
        status = "PASS" if error < 1e-3 else "FAIL"
        print(f"[{status}] {name} Sum: {result:.4f} (ref: {ref_sum:.4f}, rel_err: {error:.2e})")

    # Max / Min
    result_max = reduce_max_cuda(x)
    result_min = reduce_min_cuda(x)
    max_ok = abs(result_max - ref_max) < 1e-5
    min_ok = abs(result_min - ref_min) < 1e-5
    print(f"[{'PASS' if max_ok else 'FAIL'}] Max: {result_max:.6f} (ref: {ref_max:.6f})")
    print(f"[{'PASS' if min_ok else 'FAIL'}] Min: {result_min:.6f} (ref: {ref_min:.6f})")


# =============================================================================
# 벤치마크
# =============================================================================

def benchmark_reductions():
    print("\n" + "=" * 60)
    print("성능 벤치마크: Reduction 비교")
    print("=" * 60)

    n = 10_000_000
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    iterations = 100

    methods = [
        ("Interleaved", lambda: reduce_sum_interleaved(x)),
        ("Sequential", lambda: reduce_sum_sequential(x)),
        ("First-add", lambda: reduce_sum_first_add(x)),
        ("PyTorch sum()", lambda: x.sum().item()),
    ]

    for name, fn in methods:
        # Warm-up
        for _ in range(10):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end) / iterations
        bandwidth = (n * 4) / (time_ms / 1000) / 1e9  # GB/s (read only)
        print(f"{name:20s}: {time_ms:.4f} ms  ({bandwidth:.1f} GB/s)")


if __name__ == "__main__":
    print("\nWeek 6: Parallel Reduction - Naive to Optimized".center(60))
    print("=" * 60)

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}\n")

    verify_reductions()
    benchmark_reductions()

    print("\n" + "=" * 60)
    print("다음: lecture6_warp_shuffle.py (Warp Shuffle Reduction)")
    print("=" * 60)
