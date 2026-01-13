"""
Week 4 복습: Roofline Model 실습 📊

학습 목표:
1. Roofline Model의 개념 완전 이해
2. Arithmetic Intensity (AI) 계산 실습
3. Compute-Bound vs Memory-Bound 구분
4. 다양한 AI를 가진 커널 구현 및 분석
5. 실제 성능 측정 및 Roofline Plot 생성

Roofline Model 핵심 개념:
- Arithmetic Intensity = FLOPs / Bytes Transferred
- Peak Performance (GFLOPS): GPU의 최대 연산 성능
- Peak Bandwidth (GB/s): GPU의 최대 메모리 대역폭
- Ridge Point: Compute-Bound와 Memory-Bound의 경계

구현할 커널들 (다양한 AI):
1. Memory Copy (AI ≈ 0): 순수 메모리 전송
2. Vector Scale (AI ≈ 0.25): 간단한 연산
3. AXPY (AI ≈ 0.5): 약간의 연산
4. Dot Product (AI ≈ 1.0): Reduction
5. Heavy Compute (AI ≈ 64): 연산 집약적
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

# Compiler 설정
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

# ============================================================
# CUDA Kernel Source Code
# ============================================================

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// =============================================================================
// Kernel 1: Memory Copy (AI ≈ 0)
// FLOPs: 0, Bytes: 2N (read + write)
// =============================================================================
__global__ void memory_copy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i];  // 순수 메모리 전송
    }
}

// =============================================================================
// Kernel 2: Vector Scale (AI ≈ 0.25)
// FLOPs: 1 per element, Bytes: 8 (read 4 + write 4)
// AI = 1 FLOP / 8 bytes = 0.125 FLOP/byte
// =============================================================================
__global__ void vector_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        output[i] = alpha * input[i];  // 1 FLOP
    }
}

// =============================================================================
// Kernel 3: AXPY (AI ≈ 0.5)
// FLOPs: 2 per element (mul + add), Bytes: 16 (read 8 + write 4 + read 4)
// AI = 2 FLOP / 16 bytes = 0.125 FLOP/byte
// =============================================================================
__global__ void axpy_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        y[i] = alpha * x[i] + y[i];  // 2 FLOPs
    }
}

// =============================================================================
// Kernel 4: Dot Product with Reduction (AI ≈ 1.0)
// FLOPs: 2N (mul + add), Bytes: 8N (read a + read b)
// AI = 2N / 8N = 0.25 FLOP/byte (global memory)
// 하지만 Shared Memory reduction으로 effective AI 증가
// =============================================================================
__global__ void dot_product_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ partial_sums,
    int n
) {
    __shared__ float shared[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;
    
    // Accumulate partial sum
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += a[i] * b[i];  // 2 FLOPs per element
    }
    shared[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

// =============================================================================
// Kernel 5: Heavy Compute (AI ≈ 64)
// 많은 연산, 적은 메모리 접근
// =============================================================================
__global__ void heavy_compute_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float x = input[i];
        float result = x;
        
        // 많은 연산 수행 (512 FLOPs)
        #pragma unroll
        for (int k = 0; k < 64; k++) {
            result = result * 1.01f + 0.001f;  // 2 FLOPs
            result = result * result;          // 1 FLOP
            result = result * 0.99f;           // 1 FLOP
            result = result + x * 0.001f;      // 2 FLOPs
            result = result * 0.999f + 0.001f; // 2 FLOP
        }
        
        output[i] = result;
        // Total: ~512 FLOPs per element
        // Bytes: 8 (read 4 + write 4)
        // AI = 512 / 8 = 64 FLOP/byte
    }
}

// Launcher Functions
void launch_memory_copy(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    int threads = 256;
    int sm_count = 84;  // Typical for modern GPUs
    int blocks = sm_count * 4;
    memory_copy_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_vector_scale(torch::Tensor input, torch::Tensor output, float alpha) {
    int n = input.numel();
    int threads = 256;
    int sm_count = 84;
    int blocks = sm_count * 4;
    vector_scale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), alpha, n);
}

void launch_axpy(torch::Tensor x, torch::Tensor y, float alpha) {
    int n = x.numel();
    int threads = 256;
    int sm_count = 84;
    int blocks = sm_count * 4;
    axpy_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), alpha, n);
}

void launch_dot_product(torch::Tensor a, torch::Tensor b, torch::Tensor partial_sums) {
    int n = a.numel();
    int threads = 256;
    int sm_count = 84;
    int blocks = sm_count * 4;
    dot_product_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), partial_sums.data_ptr<float>(), n);
}

void launch_heavy_compute(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    int threads = 256;
    int sm_count = 84;
    int blocks = sm_count * 4;
    heavy_compute_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_source = """
void launch_memory_copy(torch::Tensor input, torch::Tensor output);
void launch_vector_scale(torch::Tensor input, torch::Tensor output, float alpha);
void launch_axpy(torch::Tensor x, torch::Tensor y, float alpha);
void launch_dot_product(torch::Tensor a, torch::Tensor b, torch::Tensor partial_sums);
void launch_heavy_compute(torch::Tensor input, torch::Tensor output);
"""

# JIT 컴파일
roofline_module = load_inline(
    name='roofline_module',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=[
        'launch_memory_copy', 'launch_vector_scale', 'launch_axpy',
        'launch_dot_product', 'launch_heavy_compute'
    ],
    verbose=False
)

# ============================================================
# Roofline Analysis Functions
# ============================================================

def get_gpu_specs() -> tuple:
    """GPU 사양 가져오기"""
    props = torch.cuda.get_device_properties(0)
    
    # Peak GFLOPS (단정밀도)
    # A100: ~19.5 TFLOPS, RTX 3090: ~35.6 TFLOPS
    clock_rate_ghz = props.clock_rate / 1e6
    sm_count = props.multi_processor_count
    
    # Rough estimation (actual may vary)
    # 보수적으로 측정된 값 사용하는 것이 좋음
    peak_gflops = clock_rate_ghz * sm_count * 128  # Cores per SM varies
    
    # Peak Bandwidth (GB/s)
    # A100: ~1555 GB/s, RTX 3090: ~936 GB/s
    # 실측값 사용 권장
    peak_bandwidth = 900.0  # Placeholder - 실제 측정 필요
    
    return peak_gflops, peak_bandwidth


def calculate_arithmetic_intensity(flops_per_element: float, bytes_per_element: float) -> float:
    """Arithmetic Intensity 계산"""
    return flops_per_element / bytes_per_element


def benchmark_kernel(name: str, kernel_func, n: int, flops_per_element: float, 
                     bytes_per_element: float, iterations: int = 100) -> dict:
    """커널 벤치마크 및 분석"""
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print(f"{'='*70}")
    
    # Warm-up
    for _ in range(10):
        kernel_func()
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        kernel_func()
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iterations
    time_s = time_ms / 1000.0
    
    # Performance 계산
    total_flops = n * flops_per_element
    total_bytes = n * bytes_per_element
    
    achieved_gflops = (total_flops / time_s) / 1e9
    achieved_bandwidth = (total_bytes / time_s) / 1e9
    
    ai = calculate_arithmetic_intensity(flops_per_element, bytes_per_element)
    
    # Results
    print(f"⏱️  Time: {time_ms:.4f} ms")
    print(f"🧮 FLOPs per element: {flops_per_element}")
    print(f"💾 Bytes per element: {bytes_per_element}")
    print(f"📐 Arithmetic Intensity: {ai:.3f} FLOP/byte")
    print(f"⚡ Achieved Performance: {achieved_gflops:.2f} GFLOPS")
    print(f"🚀 Achieved Bandwidth: {achieved_bandwidth:.2f} GB/s")
    
    peak_gflops, peak_bandwidth = get_gpu_specs()
    print(f"📈 Peak Performance Utilization: {(achieved_gflops/peak_gflops)*100:.1f}%")
    print(f"📈 Peak Bandwidth Utilization: {(achieved_bandwidth/peak_bandwidth)*100:.1f}%")
    
    # Memory-bound or Compute-bound?
    ridge_point = peak_gflops / peak_bandwidth
    if ai < ridge_point:
        print(f"🏷️  Type: Memory-Bound (AI < Ridge Point {ridge_point:.2f})")
    else:
        print(f"🏷️  Type: Compute-Bound (AI >= Ridge Point {ridge_point:.2f})")
    
    return {
        'name': name,
        'ai': ai,
        'gflops': achieved_gflops,
        'bandwidth': achieved_bandwidth,
        'time_ms': time_ms
    }


def run_roofline_analysis(n: int = 100_000_000) -> None:
    """Roofline 분석 실행"""
    print("=" * 70)
    print("📊 Roofline Model 분석")
    print("=" * 70)
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"📏 Data Size: {n:,} elements ({n*4/1e6:.1f} MB)")
    
    peak_gflops, peak_bandwidth = get_gpu_specs()
    ridge_point = peak_gflops / peak_bandwidth
    
    print(f"\n🎯 GPU 사양:")
    print(f"  - Peak Performance: ~{peak_gflops:.0f} GFLOPS (estimated)")
    print(f"  - Peak Bandwidth: ~{peak_bandwidth:.0f} GB/s (estimated)")
    print(f"  - Ridge Point: {ridge_point:.2f} FLOP/byte")
    print()
    
    results = []
    
    # Kernel 1: Memory Copy (AI ≈ 0)
    input_tensor = torch.randn(n, device='cuda', dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)
    result = benchmark_kernel(
        "Memory Copy",
        lambda: roofline_module.launch_memory_copy(input_tensor, output_tensor),
        n, flops_per_element=0, bytes_per_element=8
    )
    results.append(result)
    
    # Kernel 2: Vector Scale (AI ≈ 0.125)
    result = benchmark_kernel(
        "Vector Scale",
        lambda: roofline_module.launch_vector_scale(input_tensor, output_tensor, 2.5),
        n, flops_per_element=1, bytes_per_element=8
    )
    results.append(result)
    
    # Kernel 3: AXPY (AI ≈ 0.125)
    x_tensor = torch.randn(n, device='cuda', dtype=torch.float32)
    y_tensor = torch.randn(n, device='cuda', dtype=torch.float32)
    result = benchmark_kernel(
        "AXPY",
        lambda: roofline_module.launch_axpy(x_tensor, y_tensor, 2.5),
        n, flops_per_element=2, bytes_per_element=12
    )
    results.append(result)
    
    # Kernel 4: Dot Product (AI ≈ 0.25 from global memory perspective)
    a_tensor = torch.randn(n, device='cuda', dtype=torch.float32)
    b_tensor = torch.randn(n, device='cuda', dtype=torch.float32)
    partial = torch.zeros(84*4, device='cuda', dtype=torch.float32)
    result = benchmark_kernel(
        "Dot Product",
        lambda: roofline_module.launch_dot_product(a_tensor, b_tensor, partial),
        n, flops_per_element=2, bytes_per_element=8
    )
    results.append(result)
    
    # Kernel 5: Heavy Compute (AI ≈ 64)
    input_heavy = torch.randn(n//100, device='cuda', dtype=torch.float32)  # 작은 크기
    output_heavy = torch.empty_like(input_heavy)
    result = benchmark_kernel(
        "Heavy Compute",
        lambda: roofline_module.launch_heavy_compute(input_heavy, output_heavy),
        n//100, flops_per_element=512, bytes_per_element=8
    )
    results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 Roofline 분석 요약")
    print("=" * 70)
    print(f"{'Kernel':<20} {'AI (FLOP/B)':<15} {'GFLOPS':<12} {'BW (GB/s)':<12} {'Type'}")
    print("-" * 70)
    
    for r in results:
        kernel_type = "Memory" if r['ai'] < ridge_point else "Compute"
        print(f"{r['name']:<20} {r['ai']:>13.3f}   {r['gflops']:>10.2f}  {r['bandwidth']:>10.2f}  {kernel_type}")
    
    print()


def print_roofline_explanation() -> None:
    """Roofline Model 설명"""
    print("=" * 70)
    print("📚 Roofline Model 이해하기")
    print("=" * 70)
    print()
    print("🎯 핵심 개념:")
    print("  1. Arithmetic Intensity (AI) = FLOPs / Bytes Transferred")
    print("     - 메모리 접근 대비 연산량")
    print("     - 높을수록 연산 집약적")
    print()
    print("  2. Ridge Point = Peak GFLOPS / Peak Bandwidth")
    print("     - Memory-Bound와 Compute-Bound의 경계")
    print("     - AI < Ridge Point → Memory-Bound")
    print("     - AI >= Ridge Point → Compute-Bound")
    print()
    print("  3. 최적화 전략:")
    print("     - Memory-Bound: 메모리 접근 최적화 (Coalescing, Caching)")
    print("     - Compute-Bound: 연산 최적화 (더 많은 Thread, Pipeline)")
    print()
    print("💡 실무 활용:")
    print("  - 커널의 성능 병목 파악")
    print("  - 최적화 방향 결정")
    print("  - 이론적 성능 한계 계산")
    print()


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("📊 Week 4 복습: Roofline Model 실습")
    print("=" * 70)
    
    # Roofline Model 설명
    print_roofline_explanation()
    
    # Roofline 분석 실행
    run_roofline_analysis(n=100_000_000)
    
    print("=" * 70)
    print("✅ Roofline Model 실습 완료!")
    print("=" * 70)
    print()
    print("🎓 학습 포인트:")
    print("  1. Arithmetic Intensity가 커널 성능 특성을 결정")
    print("  2. Ridge Point를 기준으로 최적화 전략 수립")
    print("  3. Memory-Bound 커널은 메모리 접근 최적화가 핵심")
    print("  4. Compute-Bound 커널은 연산 효율화가 핵심")
    print("  5. 실제 GPU 성능 측정으로 이론과 실전 연결")
    print()
