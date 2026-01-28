"""
Week 6: Atomic Operations
atomicAdd, atomicMax, atomicMin, atomicCAS + Histogram 구현

목표:
1. Atomic 연산의 동작 원리 이해
2. Histogram 커널 구현 (실전 예제)
3. Atomic contention 문제와 privatization 해결법
4. Atomic vs Reduction 트레이드오프 비교
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

#define BLOCK_SIZE 256

// =============================================================================
// atomicAdd를 이용한 Global Sum Reduction
// 간단하지만 contention이 심함 → 대규모에서 느림
// =============================================================================
__global__ void reduce_atomic_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop
    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        local_sum += input[i];
    }

    // 각 스레드가 atomicAdd로 글로벌 합산
    atomicAdd(output, local_sum);
}

// =============================================================================
// atomicMax / atomicMin (정수 버전)
// float용 atomicMax는 없으므로 atomicCAS로 구현
// =============================================================================

// float atomicMax using atomicCAS
__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_val = __int_as_float(expected);
        if (old_val >= val) break;
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (old != expected);
    return __int_as_float(old);
}

__device__ float atomicMinFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_val = __int_as_float(expected);
        if (old_val <= val) break;
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (old != expected);
    return __int_as_float(old);
}

__global__ void reduce_atomic_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_max = -1e38f;
    for (int i = idx; i < n; i += stride) {
        local_max = fmaxf(local_max, input[i]);
    }

    atomicMaxFloat(output, local_max);
}

__global__ void reduce_atomic_min_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_min = 1e38f;
    for (int i = idx; i < n; i += stride) {
        local_min = fminf(local_min, input[i]);
    }

    atomicMinFloat(output, local_min);
}

// =============================================================================
// Histogram: Naive (Global Atomic)
// 모든 스레드가 글로벌 메모리에 직접 atomicAdd → contention 심함
// =============================================================================
__global__ void histogram_naive_kernel(
    const int* __restrict__ input,
    int* __restrict__ histogram,
    int n,
    int num_bins
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        int bin = input[i] % num_bins;
        atomicAdd(&histogram[bin], 1);
    }
}

// =============================================================================
// Histogram: Privatization (Shared Memory)
// 각 블록이 shared memory에 로컬 히스토그램 구축 → 마지막에 global에 합산
// contention 대폭 감소
// =============================================================================
__global__ void histogram_privatized_kernel(
    const int* __restrict__ input,
    int* __restrict__ histogram,
    int n,
    int num_bins
) {
    extern __shared__ int local_hist[];

    int tid = threadIdx.x;

    // Shared memory 히스토그램 초기화
    for (int i = tid; i < num_bins; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // 로컬 히스토그램에 카운트 (shared memory atomic → 훨씬 빠름)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        int bin = input[i] % num_bins;
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    // 로컬 → 글로벌 합산
    for (int i = tid; i < num_bins; i += blockDim.x) {
        if (local_hist[i] > 0) {
            atomicAdd(&histogram[i], local_hist[i]);
        }
    }
}

// =============================================================================
// Launch 함수들 - output 텐서를 Python에서 전달받음
// =============================================================================

void launch_reduce_atomic_sum(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_atomic_sum_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_reduce_atomic_max(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_atomic_max_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_reduce_atomic_min(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_atomic_min_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_histogram_naive(torch::Tensor input, torch::Tensor histogram, int n, int num_bins, int blocks) {
    histogram_naive_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<int>(), histogram.data_ptr<int>(), n, num_bins);
}

void launch_histogram_privatized(torch::Tensor input, torch::Tensor histogram, int n, int num_bins, int blocks) {
    int shared_mem = num_bins * sizeof(int);
    histogram_privatized_kernel<<<blocks, BLOCK_SIZE, shared_mem>>>(
        input.data_ptr<int>(), histogram.data_ptr<int>(), n, num_bins);
}
"""

cpp_header = """
void launch_reduce_atomic_sum(torch::Tensor input, torch::Tensor output, int n, int blocks);
void launch_reduce_atomic_max(torch::Tensor input, torch::Tensor output, int n, int blocks);
void launch_reduce_atomic_min(torch::Tensor input, torch::Tensor output, int n, int blocks);
void launch_histogram_naive(torch::Tensor input, torch::Tensor histogram, int n, int num_bins, int blocks);
void launch_histogram_privatized(torch::Tensor input, torch::Tensor histogram, int n, int num_bins, int blocks);
"""

# JIT 컴파일
print("=" * 60)
print("Compiling Atomic Operations Kernels...")
print("=" * 60)

module = load_inline(
    name='atomic_ops_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_reduce_atomic_sum', 'launch_reduce_atomic_max',
               'launch_reduce_atomic_min', 'launch_histogram_naive',
               'launch_histogram_privatized'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# Python Wrapper 함수들
# =============================================================================

def reduce_atomic_sum(x: torch.Tensor) -> float:
    """Atomic Sum Reduction"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = min(256, cdiv(n, BLOCK_SIZE))
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    module.launch_reduce_atomic_sum(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.item()


def reduce_atomic_max(x: torch.Tensor) -> float:
    """Atomic Max Reduction (CAS-based)"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = min(256, cdiv(n, BLOCK_SIZE))
    output = torch.full((1,), float('-inf'), dtype=torch.float32, device='cuda')
    module.launch_reduce_atomic_max(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.item()


def reduce_atomic_min(x: torch.Tensor) -> float:
    """Atomic Min Reduction (CAS-based)"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = min(256, cdiv(n, BLOCK_SIZE))
    output = torch.full((1,), float('inf'), dtype=torch.float32, device='cuda')
    module.launch_reduce_atomic_min(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.item()


def histogram_naive(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Naive Histogram (Global Atomic)"""
    assert x.is_cuda and x.dtype == torch.int32
    x = x.contiguous()
    n = x.numel()
    blocks = min(256, cdiv(n, BLOCK_SIZE))
    histogram = torch.zeros(num_bins, dtype=torch.int32, device='cuda')
    module.launch_histogram_naive(x, histogram, n, num_bins, blocks)
    torch.cuda.synchronize()
    return histogram


def histogram_privatized(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Privatized Histogram (Shared Memory)"""
    assert x.is_cuda and x.dtype == torch.int32
    x = x.contiguous()
    n = x.numel()
    blocks = min(256, cdiv(n, BLOCK_SIZE))
    histogram = torch.zeros(num_bins, dtype=torch.int32, device='cuda')
    module.launch_histogram_privatized(x, histogram, n, num_bins, blocks)
    torch.cuda.synchronize()
    return histogram


# =============================================================================
# 검증
# =============================================================================

def verify_atomic_reductions():
    print("=" * 60)
    print("Atomic Reduction 검증")
    print("=" * 60)

    n = 1_000_003
    x = torch.randn(n, dtype=torch.float32, device='cuda')

    ref_sum = x.sum().item()
    ref_max = x.max().item()
    ref_min = x.min().item()

    result_sum = reduce_atomic_sum(x)
    result_max = reduce_atomic_max(x)
    result_min = reduce_atomic_min(x)

    sum_err = abs(result_sum - ref_sum) / (abs(ref_sum) + 1e-8)
    max_ok = abs(result_max - ref_max) < 1e-4
    min_ok = abs(result_min - ref_min) < 1e-4

    print(f"[{'PASS' if sum_err < 1e-3 else 'FAIL'}] Atomic Sum: {result_sum:.4f} (ref: {ref_sum:.4f}, rel_err: {sum_err:.2e})")
    print(f"[{'PASS' if max_ok else 'FAIL'}] Atomic Max: {result_max:.6f} (ref: {ref_max:.6f})")
    print(f"[{'PASS' if min_ok else 'FAIL'}] Atomic Min: {result_min:.6f} (ref: {ref_min:.6f})")


def verify_histogram():
    print("\n" + "=" * 60)
    print("Histogram 검증")
    print("=" * 60)

    n = 1_000_000
    num_bins = 256
    x = torch.randint(0, num_bins, (n,), dtype=torch.int32, device='cuda')

    # PyTorch 레퍼런스
    ref = torch.bincount(x, minlength=num_bins)

    hist_naive = histogram_naive(x, num_bins)
    hist_priv = histogram_privatized(x, num_bins)

    naive_ok = torch.equal(hist_naive, ref)
    priv_ok = torch.equal(hist_priv, ref)

    print(f"[{'PASS' if naive_ok else 'FAIL'}] Naive Histogram")
    print(f"[{'PASS' if priv_ok else 'FAIL'}] Privatized Histogram")

    if naive_ok:
        print(f"  샘플 bins [0:8]: {hist_naive[:8].tolist()}")
        print(f"  PyTorch ref:     {ref[:8].tolist()}")


# =============================================================================
# 벤치마크
# =============================================================================

def benchmark_histogram():
    print("\n" + "=" * 60)
    print("성능 벤치마크: Histogram (Naive vs Privatized)")
    print("=" * 60)

    n = 10_000_000
    num_bins = 256
    x = torch.randint(0, num_bins, (n,), dtype=torch.int32, device='cuda')
    iterations = 100

    methods = [
        ("Naive (Global Atomic)", lambda: histogram_naive(x, num_bins)),
        ("Privatized (Shared Mem)", lambda: histogram_privatized(x, num_bins)),
        ("PyTorch bincount()", lambda: torch.bincount(x, minlength=num_bins)),
    ]

    for name, fn in methods:
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
        throughput = (n / 1e6) / (time_ms / 1000)  # M elements/s
        print(f"{name:30s}: {time_ms:.4f} ms  ({throughput:.0f} M elem/s)")


def benchmark_atomic_vs_reduction():
    print("\n" + "=" * 60)
    print("성능 벤치마크: Atomic Sum vs PyTorch")
    print("=" * 60)

    n = 10_000_000
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    iterations = 100

    methods = [
        ("Atomic Sum", lambda: reduce_atomic_sum(x)),
        ("PyTorch sum()", lambda: x.sum().item()),
    ]

    for name, fn in methods:
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
        bandwidth = (n * 4) / (time_ms / 1000) / 1e9
        print(f"{name:30s}: {time_ms:.4f} ms  ({bandwidth:.1f} GB/s)")


if __name__ == "__main__":
    print("\nWeek 6: Atomic Operations".center(60))
    print("=" * 60)

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}\n")

    verify_atomic_reductions()
    verify_histogram()
    benchmark_histogram()
    benchmark_atomic_vs_reduction()

    print("\n" + "=" * 60)
    print("Lecture 6 완료!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Atomic은 간단하지만 contention으로 인해 대규모에서 느림")
    print("- Privatization(shared mem)으로 contention 대폭 감소")
    print("- Reduction 커널이 대체로 atomic보다 빠름 (구조적 병렬성)")
