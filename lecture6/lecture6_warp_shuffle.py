"""
Week 6: Warp Shuffle을 활용한 고성능 Reduction
Warp-level 프리미티브와 하이브리드 최적화

목표:
1. __shfl_down_sync를 이용한 warp-level reduction
2. Shared Memory + Warp Shuffle 하이브리드 reduction
3. Warp Shuffle 프리미티브 이해 (broadcast, butterfly)
4. Naive vs Warp Shuffle vs PyTorch 성능 비교
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

BLOCK_SIZE = 256
WARP_SIZE = 32


def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division"""
    return (n + divisor - 1) // divisor


# =============================================================================
# CUDA Kernel 소스코드
# =============================================================================

cuda_source = """
#include <torch/extension.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFFu

// =============================================================================
// Warp-level reduction helper (합산)
// __shfl_down_sync: 현재 lane + offset의 값을 읽어옴
// =============================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;  // lane 0에 최종 합
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

// =============================================================================
// Kernel 4: Warp Shuffle Reduction
// 각 warp가 __shfl_down_sync로 reduction → warp 대표값을 shared memory에 모음
// → 마지막 warp가 다시 shuffle로 최종 합산
// =============================================================================
__global__ void reduce_warp_shuffle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    // First-add-during-load: 각 블록이 2 * BLOCK_SIZE 원소 담당
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (idx < n) val += input[idx];
    if (idx + blockDim.x < n) val += input[idx + blockDim.x];

    // Step 1: Warp 내부 reduction (shared memory 없이!)
    val = warp_reduce_sum(val);

    // Step 2: 각 warp의 lane 0이 shared memory에 저장
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];  // 최대 8개 warp
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Step 3: 첫 번째 warp가 warp_sums를 최종 reduction
    int num_warps = blockDim.x / WARP_SIZE;
    if (tid < num_warps) {
        val = warp_sums[tid];
    } else {
        val = 0.0f;
    }

    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// =============================================================================
// Kernel 5: 완전 최적화 Reduction
// Grid-Stride Loop + Warp Shuffle + Shared Memory 하이브리드
// 적은 수의 블록으로 전체 배열을 처리
// =============================================================================
__global__ void reduce_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int tid = threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop: 각 스레드가 여러 원소를 합산
    float val = 0.0f;
    for (int i = idx; i < n; i += grid_stride) {
        val += input[i];
    }

    // Warp 내부 reduction
    val = warp_reduce_sum(val);

    // Warp 대표값 → shared memory
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // 첫 번째 warp가 최종 reduction
    int num_warps = blockDim.x / WARP_SIZE;
    if (tid < num_warps) {
        val = warp_sums[tid];
    } else {
        val = 0.0f;
    }

    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// =============================================================================
// Warp Shuffle 프리미티브 데모 커널
// 결과를 output에 기록하여 Python에서 확인
// =============================================================================
__global__ void warp_shuffle_demo_kernel(
    float* __restrict__ output_broadcast,
    float* __restrict__ output_butterfly,
    int n
) {
    int tid = threadIdx.x;
    if (tid >= n) return;

    int lane = tid % WARP_SIZE;
    float val = (float)lane;  // 각 lane에 자기 lane 번호

    // Demo 1: Broadcast - lane 0의 값을 모든 lane에 전파
    float broadcast_val = __shfl_sync(FULL_MASK, val, 0);
    output_broadcast[tid] = broadcast_val;

    // Demo 2: Butterfly (XOR) reduction
    // XOR 패턴: lane^1, lane^2, lane^4, ...
    float butterfly_val = val;
    for (int mask = 1; mask < WARP_SIZE; mask <<= 1) {
        butterfly_val += __shfl_xor_sync(FULL_MASK, butterfly_val, mask);
    }
    output_butterfly[tid] = butterfly_val;  // 모든 lane에 합(0+1+...+31)=496
}

// =============================================================================
// Launch 함수들 - output 텐서를 Python에서 전달받음
// =============================================================================

void launch_reduce_warp_shuffle(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_warp_shuffle_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_reduce_optimized(torch::Tensor input, torch::Tensor output, int n, int blocks) {
    reduce_optimized_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_warp_shuffle_demo(torch::Tensor output_broadcast, torch::Tensor output_butterfly, int n) {
    warp_shuffle_demo_kernel<<<1, n>>>(
        output_broadcast.data_ptr<float>(),
        output_butterfly.data_ptr<float>(),
        n);
}
"""

cpp_header = """
void launch_reduce_warp_shuffle(torch::Tensor input, torch::Tensor output, int n, int blocks);
void launch_reduce_optimized(torch::Tensor input, torch::Tensor output, int n, int blocks);
void launch_warp_shuffle_demo(torch::Tensor output_broadcast, torch::Tensor output_butterfly, int n);
"""

# JIT 컴파일
print("=" * 60)
print("Compiling Warp Shuffle Reduction Kernels...")
print("=" * 60)

module = load_inline(
    name='warp_shuffle_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_reduce_warp_shuffle', 'launch_reduce_optimized',
               'launch_warp_shuffle_demo'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# Python Wrapper 함수들
# =============================================================================

def reduce_sum_warp_shuffle(x: torch.Tensor) -> float:
    """Warp Shuffle Sum Reduction"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = cdiv(n, BLOCK_SIZE * 2)
    output = torch.zeros(blocks, dtype=torch.float32, device='cuda')
    module.launch_reduce_warp_shuffle(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.sum().item()


def reduce_sum_optimized(x: torch.Tensor) -> float:
    """Fully Optimized Sum Reduction (Grid-Stride + Warp Shuffle)"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n = x.numel()
    blocks = min(256, cdiv(n, BLOCK_SIZE))
    output = torch.zeros(blocks, dtype=torch.float32, device='cuda')
    module.launch_reduce_optimized(x, output, n, blocks)
    torch.cuda.synchronize()
    return output.sum().item()


# =============================================================================
# 검증
# =============================================================================

def verify_reductions():
    print("=" * 60)
    print("Warp Shuffle Reduction 검증")
    print("=" * 60)

    n = 1_000_003
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    ref_sum = x.sum().item()

    for name, fn in [("Warp Shuffle", reduce_sum_warp_shuffle),
                     ("Optimized", reduce_sum_optimized)]:
        result = fn(x)
        error = abs(result - ref_sum) / (abs(ref_sum) + 1e-8)
        status = "PASS" if error < 1e-3 else "FAIL"
        print(f"[{status}] {name}: {result:.4f} (ref: {ref_sum:.4f}, rel_err: {error:.2e})")


def demo_warp_shuffle():
    print("\n" + "=" * 60)
    print("Warp Shuffle 프리미티브 데모")
    print("=" * 60)

    n = 32  # 1 warp
    output_broadcast = torch.zeros(n, dtype=torch.float32, device='cuda')
    output_butterfly = torch.zeros(n, dtype=torch.float32, device='cuda')
    module.launch_warp_shuffle_demo(output_broadcast, output_butterfly, n)
    torch.cuda.synchronize()

    broadcast = output_broadcast.cpu()
    butterfly = output_butterfly.cpu()

    print(f"\n각 lane 초기값: [0, 1, 2, ..., 31]")
    print(f"\n__shfl_sync (Broadcast from lane 0):")
    print(f"  결과: {broadcast[:8].tolist()} ... (모두 0.0)")
    print(f"\n__shfl_xor_sync (Butterfly reduction):")
    expected = sum(range(32))
    print(f"  결과: {butterfly[:8].tolist()} ... (모두 {expected})")
    print(f"  설명: XOR 패턴으로 전체 합산 -> 모든 lane이 sum(0..31)={expected} 보유")


# =============================================================================
# 벤치마크
# =============================================================================

def benchmark_reductions():
    print("\n" + "=" * 60)
    print("성능 벤치마크: Naive vs Warp Shuffle vs PyTorch")
    print("=" * 60)

    n = 10_000_000
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    iterations = 100

    methods = [
        ("Warp Shuffle", lambda: reduce_sum_warp_shuffle(x)),
        ("Optimized (Grid-Stride)", lambda: reduce_sum_optimized(x)),
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
        bandwidth = (n * 4) / (time_ms / 1000) / 1e9
        print(f"{name:30s}: {time_ms:.4f} ms  ({bandwidth:.1f} GB/s)")


if __name__ == "__main__":
    print("\nWeek 6: Warp Shuffle Reduction".center(60))
    print("=" * 60)

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}\n")

    verify_reductions()
    demo_warp_shuffle()
    benchmark_reductions()

    print("\n" + "=" * 60)
    print("다음: lecture6_atomic_ops.py (Atomic Operations)")
    print("=" * 60)
