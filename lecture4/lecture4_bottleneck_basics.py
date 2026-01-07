"""
Week 4 실습 1: Bottleneck 기본 - CPU vs GPU 병목 현상 체험

목표:
1. CPU와 GPU의 서로 다른 병목 유형을 실제로 체험
2. 각 병목이 성능에 미치는 영향 정량화
3. 병목 진단 방법론 습득
4. Roofline Model을 실제 데이터로 구현

학습 포인트:
- Memory Bound vs Compute Bound 구분
- Arithmetic Intensity 계산법
- Occupancy와 성능의 관계
- Memory Coalescing의 실질적 영향
"""
import os
import math
import time
import torch
import numpy as np
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"


# =============================================================================
# CUDA 커널들 - 다양한 병목 시나리오
# =============================================================================

cuda_source = """
#include <torch/extension.h>

// 🔴 Memory Bound 시나리오 1: Element-wise Operations (낮은 AI)
__global__ void memory_bound_elementwise(
    const float* __restrict__ a,
    const float* __restrict__ b, 
    float* __restrict__ c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // AI = 1 FLOP / 12 bytes = 0.083 FLOP/byte (매우 Memory Bound)
        c[i] = a[i] + b[i];  // 1 FLOP, 3 memory accesses (12 bytes)
    }
}

// 🔴 Memory Bound 시나리오 2: Strided Access (비효율적 접근)
__global__ void memory_bound_strided(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n, int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * stride < n) {
        // Coalescing 완전 깨짐 → Memory Bandwidth 급감
        output[i] = input[i * stride] * 2.0f;
    }
}

// 🟡 Mixed Bound: Medium Arithmetic Intensity
__global__ void mixed_bound_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        // AI = 10 FLOPs / 12 bytes = 0.83 FLOP/byte (여전히 Memory Bound이지만 개선됨)
        for (int j = 0; j < 10; j++) {
            sum += a[i] * b[i];  // 10 FLOPs per iteration
        }
        c[i] = sum;
    }
}

// 🟢 Compute Bound: High Arithmetic Intensity
__global__ void compute_bound_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        // AI = 1000 FLOPs / 12 bytes = 83.3 FLOP/byte (Compute Bound!)
        for (int j = 0; j < 1000; j++) {
            sum += a[i] * b[i];  // 1000 FLOPs per element
        }
        c[i] = sum;
    }
}

// 🔵 Occupancy Test: 다양한 레지스터 사용량
__global__ void low_occupancy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 많은 레지스터 변수 사용 (occupancy 감소)
        float r1 = input[i], r2 = input[i], r3 = input[i], r4 = input[i];
        float r5 = input[i], r6 = input[i], r7 = input[i], r8 = input[i];
        float r9 = input[i], r10 = input[i], r11 = input[i], r12 = input[i];
        float r13 = input[i], r14 = input[i], r15 = input[i], r16 = input[i];
        
        // 복잡한 연산 (레지스터 압박)
        for (int j = 0; j < 100; j++) {
            r1 = r1 * 1.1f + r2; r2 = r2 * 1.1f + r3; r3 = r3 * 1.1f + r4; r4 = r4 * 1.1f + r5;
            r5 = r5 * 1.1f + r6; r6 = r6 * 1.1f + r7; r7 = r7 * 1.1f + r8; r8 = r8 * 1.1f + r9;
            r9 = r9 * 1.1f + r10; r10 = r10 * 1.1f + r11; r11 = r11 * 1.1f + r12; r12 = r12 * 1.1f + r13;
            r13 = r13 * 1.1f + r14; r14 = r14 * 1.1f + r15; r15 = r15 * 1.1f + r16; r16 = r16 * 1.1f + r1;
        }
        
        output[i] = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16;
    }
}

__global__ void high_occupancy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 적은 레지스터 사용 (occupancy 최적화)
        float sum = 0.0f;
        for (int j = 0; j < 1000; j++) {
            sum += input[i] * 1.001f;  // 간단한 연산
        }
        output[i] = sum;
    }
}

// 🟣 Warp Divergence 시나리오
__global__ void divergent_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float result;
        // 심각한 Warp Divergence (각 thread가 다른 분기)
        if (i % 32 == 0) {
            // 복잡한 연산 경로
            result = 0.0f;
            for (int j = 0; j < 100; j++) {
                result += sinf(input[i]) * cosf(input[i]);
            }
        } else if (i % 32 < 16) {
            // 중간 연산 경로  
            result = input[i] * input[i] + sqrtf(input[i]);
        } else {
            // 간단한 연산 경로
            result = input[i] + 1.0f;
        }
        output[i] = result;
    }
}

__global__ void non_divergent_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Warp-friendly: 모든 thread가 동일한 연산
        float result = input[i] + 1.0f;
        output[i] = result;
    }
}

// Python 인터페이스
torch::Tensor memory_bound_elementwise_cuda(torch::Tensor a, torch::Tensor b) {
    const int n = a.size(0);
    auto c = torch::zeros_like(a);
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    memory_bound_elementwise<<<grid_size, block_size>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n
    );
    
    return c;
}

torch::Tensor memory_bound_strided_cuda(torch::Tensor input, int stride) {
    const int n = input.size(0);
    const int output_size = (n + stride - 1) / stride;
    auto output = torch::zeros({output_size}, input.options());
    
    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;
    
    memory_bound_strided<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n, stride
    );
    
    return output;
}

torch::Tensor mixed_bound_cuda(torch::Tensor a, torch::Tensor b) {
    const int n = a.size(0);
    auto c = torch::zeros_like(a);
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    mixed_bound_kernel<<<grid_size, block_size>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n
    );
    
    return c;
}

torch::Tensor compute_bound_cuda(torch::Tensor a, torch::Tensor b) {
    const int n = a.size(0);
    auto c = torch::zeros_like(a);
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    compute_bound_kernel<<<grid_size, block_size>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n
    );
    
    return c;
}

torch::Tensor low_occupancy_cuda(torch::Tensor input) {
    const int n = input.size(0);
    auto output = torch::zeros_like(input);
    
    const int block_size = 64;  // 작은 블록 크기 (레지스터 압박으로)
    const int grid_size = (n + block_size - 1) / block_size;
    
    low_occupancy_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    
    return output;
}

torch::Tensor high_occupancy_cuda(torch::Tensor input) {
    const int n = input.size(0);
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;  // 큰 블록 크기
    const int grid_size = (n + block_size - 1) / block_size;
    
    high_occupancy_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    
    return output;
}

torch::Tensor divergent_cuda(torch::Tensor input) {
    const int n = input.size(0);
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    divergent_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    
    return output;
}

torch::Tensor non_divergent_cuda(torch::Tensor input) {
    const int n = input.size(0);
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    non_divergent_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    
    return output;
}
"""

# C++ wrapper
cpp_source = """
torch::Tensor memory_bound_elementwise_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor memory_bound_strided_cuda(torch::Tensor input, int stride);
torch::Tensor mixed_bound_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor compute_bound_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor low_occupancy_cuda(torch::Tensor input);
torch::Tensor high_occupancy_cuda(torch::Tensor input);
torch::Tensor divergent_cuda(torch::Tensor input);
torch::Tensor non_divergent_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("memory_bound_elementwise", &memory_bound_elementwise_cuda, "Memory Bound: Elementwise");
    m.def("memory_bound_strided", &memory_bound_strided_cuda, "Memory Bound: Strided Access");
    m.def("mixed_bound", &mixed_bound_cuda, "Mixed Bound");
    m.def("compute_bound", &compute_bound_cuda, "Compute Bound");
    m.def("low_occupancy", &low_occupancy_cuda, "Low Occupancy");
    m.def("high_occupancy", &high_occupancy_cuda, "High Occupancy");
    m.def("divergent", &divergent_cuda, "Warp Divergent");
    m.def("non_divergent", &non_divergent_cuda, "Non Divergent");
}
"""


# =============================================================================
# 컴파일 및 로딩
# =============================================================================

print("🔨 JIT 컴파일 중... (병목 시나리오 커널들)")
module = load_inline(
    name='bottleneck_kernels',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    verbose=False
)
print("✅ 컴파일 완료!")


# =============================================================================
# 성능 분석 함수들
# =============================================================================

def get_gpu_specs():
    """현재 GPU의 이론적 성능 스펙"""
    if not torch.cuda.is_available():
        return None
        
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    
    # 주요 GPU들의 이론적 성능 (대략적 값)
    gpu_specs = {
        "RTX 4090": {"peak_flops": 83e12, "peak_bandwidth": 1008e9},
        "RTX 3090": {"peak_flops": 36e12, "peak_bandwidth": 936e9},
        "RTX 3080": {"peak_flops": 30e12, "peak_bandwidth": 760e9},
        "H100": {"peak_flops": 67e12, "peak_bandwidth": 3350e9},
        "A100": {"peak_flops": 19.5e12, "peak_bandwidth": 1935e9},
        "V100": {"peak_flops": 15.7e12, "peak_bandwidth": 900e9},
    }
    
    # GPU 이름에서 매칭 시도
    for gpu_key, specs in gpu_specs.items():
        if gpu_key.replace(" ", "").lower() in gpu_name.replace(" ", "").lower():
            return specs
    
    # 기본값 (대략적)
    return {"peak_flops": 20e12, "peak_bandwidth": 800e9}


def calculate_roofline_metrics(flops, bytes_accessed, runtime_ms):
    """Roofline Model 메트릭 계산"""
    
    # 실제 성능
    actual_gflops = flops / (runtime_ms * 1e-3) / 1e9
    actual_bandwidth = bytes_accessed / (runtime_ms * 1e-3) / 1e9
    
    # Arithmetic Intensity
    arithmetic_intensity = flops / bytes_accessed
    
    # GPU 스펙
    specs = get_gpu_specs()
    if specs:
        peak_gflops = specs["peak_flops"] / 1e9
        peak_bandwidth = specs["peak_bandwidth"] / 1e9
        ridge_point = peak_gflops / peak_bandwidth
        
        # 효율성 계산
        compute_efficiency = (actual_gflops / peak_gflops) * 100
        memory_efficiency = (actual_bandwidth / peak_bandwidth) * 100
        
        # 병목 유형 판단
        if arithmetic_intensity < ridge_point:
            bottleneck_type = "Memory Bound"
            limiting_factor = f"Memory BW: {memory_efficiency:.1f}%"
        else:
            bottleneck_type = "Compute Bound"  
            limiting_factor = f"Compute: {compute_efficiency:.1f}%"
            
    else:
        ridge_point = peak_gflops = peak_bandwidth = None
        compute_efficiency = memory_efficiency = None
        bottleneck_type = "Unknown"
        limiting_factor = "GPU specs unknown"
    
    return {
        "arithmetic_intensity": arithmetic_intensity,
        "actual_gflops": actual_gflops,
        "actual_bandwidth": actual_bandwidth,
        "compute_efficiency": compute_efficiency,
        "memory_efficiency": memory_efficiency,
        "bottleneck_type": bottleneck_type,
        "limiting_factor": limiting_factor,
        "ridge_point": ridge_point
    }


def benchmark_kernel(name, kernel_func, *args, iterations=100):
    """커널 성능 벤치마크 및 분석"""
    
    print(f"\n🔍 {name}")
    print("-" * 50)
    
    # Warm-up
    for _ in range(10):
        result = kernel_func(*args)
    
    # 벤치마크
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        result = kernel_func(*args)
    end.record()
    torch.cuda.synchronize()
    
    runtime_ms = start.elapsed_time(end) / iterations
    
    # 메트릭 계산 (커널별로 다름)
    if "elementwise" in name.lower():
        n = args[0].numel()
        flops = n  # 1 FLOP per element (addition)
        bytes_accessed = n * 4 * 3  # 3 arrays × 4 bytes/float
    elif "strided" in name.lower():
        n = args[0].numel()
        flops = n // args[1]  # 1 FLOP per output element
        bytes_accessed = (n // args[1]) * 4 * 2  # input + output
    elif "mixed" in name.lower():
        n = args[0].numel()
        flops = n * 10  # 10 FLOPs per element
        bytes_accessed = n * 4 * 3  # 3 arrays
    elif "compute" in name.lower():
        n = args[0].numel()
        flops = n * 1000  # 1000 FLOPs per element
        bytes_accessed = n * 4 * 3  # 3 arrays
    elif "occupancy" in name.lower():
        n = args[0].numel()
        flops = n * 1000 if "high" in name.lower() else n * 1600  # 추정
        bytes_accessed = n * 4 * 2  # input + output
    elif "divergent" in name.lower():
        n = args[0].numel()
        flops = n * 50  # 평균 연산량 추정
        bytes_accessed = n * 4 * 2  # input + output
    else:
        # 기본값
        n = args[0].numel()
        flops = n
        bytes_accessed = n * 4 * 2
    
    # Roofline 분석
    metrics = calculate_roofline_metrics(flops, bytes_accessed, runtime_ms)
    
    print(f"  Runtime:              {runtime_ms:.3f} ms")
    print(f"  Arithmetic Intensity: {metrics['arithmetic_intensity']:.2f} FLOP/byte")
    print(f"  Actual Performance:   {metrics['actual_gflops']:.1f} GFLOPS")
    print(f"  Memory Bandwidth:     {metrics['actual_bandwidth']:.1f} GB/s")
    print(f"  Bottleneck Type:      {metrics['bottleneck_type']}")
    print(f"  Limiting Factor:      {metrics['limiting_factor']}")
    
    if metrics['ridge_point']:
        print(f"  Ridge Point:          {metrics['ridge_point']:.1f} FLOP/byte")
    
    return metrics


# =============================================================================
# 병목 시나리오 실험들
# =============================================================================

def experiment_arithmetic_intensity():
    """Arithmetic Intensity에 따른 성능 변화 실험"""
    
    print("\n" + "="*70)
    print("🧪 실험 1: Arithmetic Intensity와 병목 유형")
    print("="*70)
    
    device = torch.device('cuda')
    n = 1_000_000
    
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)
    
    scenarios = [
        ("Memory Bound (Elementwise)", module.memory_bound_elementwise, [a, b]),
        ("Mixed Bound (Medium AI)", module.mixed_bound, [a, b]),
        ("Compute Bound (High AI)", module.compute_bound, [a, b]),
    ]
    
    results = []
    for name, func, args in scenarios:
        metrics = benchmark_kernel(name, func, *args)
        results.append((name, metrics))
    
    # 결과 분석
    print(f"\n📊 Arithmetic Intensity 분석:")
    print(f"{'Scenario':<25} {'AI (FLOP/byte)':<15} {'Type':<15} {'Performance':<15}")
    print("-" * 70)
    
    for name, metrics in results:
        print(f"{name:<25} {metrics['arithmetic_intensity']:<15.2f} "
              f"{metrics['bottleneck_type']:<15} {metrics['actual_gflops']:<15.1f}")


def experiment_memory_coalescing():
    """Memory Coalescing이 성능에 미치는 영향"""
    
    print("\n" + "="*70)
    print("🧪 실험 2: Memory Coalescing 효과")
    print("="*70)
    
    device = torch.device('cuda')
    n = 10_000_000
    
    input_tensor = torch.randn(n, device=device, dtype=torch.float32)
    
    stride_tests = [1, 2, 4, 8, 16, 32]
    
    print(f"Input size: {n:,} elements")
    print(f"{'Stride':<10} {'Runtime (ms)':<15} {'Bandwidth (GB/s)':<20} {'Efficiency':<15}")
    print("-" * 60)
    
    baseline_bw = None
    
    for stride in stride_tests:
        metrics = benchmark_kernel(f"Strided Access (stride={stride})", 
                                 module.memory_bound_strided, 
                                 input_tensor, stride, iterations=50)
        
        if baseline_bw is None:
            baseline_bw = metrics['actual_bandwidth']
            efficiency = 100.0
        else:
            efficiency = (metrics['actual_bandwidth'] / baseline_bw) * 100
            
        print(f"{stride:<10} {metrics['actual_gflops']*1000/metrics['arithmetic_intensity']:<15.3f} "
              f"{metrics['actual_bandwidth']:<20.1f} {efficiency:<15.1f}%")
    
    print(f"\n💡 Coalescing 인사이트:")
    print(f"  - Stride=1 (Coalesced): 최고 성능")
    print(f"  - Stride 증가 → 성능 급격히 저하")
    print(f"  - GPU Memory Controller는 연속 접근에 최적화됨")


def experiment_occupancy():
    """Occupancy가 성능에 미치는 영향"""
    
    print("\n" + "="*70)
    print("🧪 실험 3: Occupancy와 성능")
    print("="*70)
    
    device = torch.device('cuda')
    n = 1_000_000
    
    input_tensor = torch.randn(n, device=device, dtype=torch.float32)
    
    scenarios = [
        ("Low Occupancy (많은 레지스터)", module.low_occupancy),
        ("High Occupancy (적은 레지스터)", module.high_occupancy),
    ]
    
    for name, func in scenarios:
        metrics = benchmark_kernel(name, func, input_tensor, iterations=50)
        
        # 추가 분석을 위한 설명
        if "Low" in name:
            print(f"    📝 Block Size: 64 (레지스터 압박으로)")
            print(f"    📝 예상 Occupancy: 낮음 (25-50%)")
        else:
            print(f"    📝 Block Size: 256 (최적화됨)")
            print(f"    📝 예상 Occupancy: 높음 (75%+)")


def experiment_warp_divergence():
    """Warp Divergence가 성능에 미치는 영향"""
    
    print("\n" + "="*70)
    print("🧪 실험 4: Warp Divergence 효과")  
    print("="*70)
    
    device = torch.device('cuda')
    n = 1_000_000
    
    input_tensor = torch.randn(n, device=device, dtype=torch.float32)
    
    scenarios = [
        ("Divergent Warps (분기 많음)", module.divergent),
        ("Non-Divergent Warps (분기 없음)", module.non_divergent),
    ]
    
    results = []
    for name, func in scenarios:
        metrics = benchmark_kernel(name, func, input_tensor, iterations=50)
        results.append((name, metrics))
    
    # 성능 비교
    divergent_perf = results[0][1]['actual_gflops'] 
    non_divergent_perf = results[1][1]['actual_gflops']
    divergence_penalty = (1 - divergent_perf / non_divergent_perf) * 100
    
    print(f"\n📊 Warp Divergence 분석:")
    print(f"  Divergence Penalty: {divergence_penalty:.1f}% 성능 저하")
    print(f"  💡 인사이트: Warp 내 모든 스레드는 같은 instruction 실행")


def experiment_roofline_visualization():
    """Roofline Model 시각화 (텍스트 기반)"""
    
    print("\n" + "="*70)
    print("📈 Roofline Model 시각화 (모든 실험 결과)")
    print("="*70)
    
    specs = get_gpu_specs()
    if not specs:
        print("❌ GPU 스펙을 알 수 없어 Roofline 분석을 건너뜁니다.")
        return
    
    ridge_point = (specs["peak_flops"] / 1e9) / (specs["peak_bandwidth"] / 1e9)
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Peak Performance: {specs['peak_flops']/1e12:.1f} TFLOPS")
    print(f"Peak Bandwidth: {specs['peak_bandwidth']/1e9:.0f} GB/s")
    print(f"Ridge Point: {ridge_point:.1f} FLOP/byte\n")
    
    # 간단한 ASCII Roofline
    print("Roofline Chart (Log Scale):")
    print("Performance")
    print("(GFLOPS) │")
    print(f"  {specs['peak_flops']/1e9:5.0f} ┤─────────────────  ← Peak Compute")
    print("         │    /")
    print("         │   /")
    print("         │  /   Memory Bound")  
    print("         │ /")
    print("         │/")
    print("         └────────────────────────► Arithmetic Intensity")
    print(f"         0    {ridge_point:5.1f}                    (FLOP/byte)")
    print("              ↑")
    print("         Ridge Point")
    
    print(f"\n💡 Roofline 해석 가이드:")
    print(f"  - AI < {ridge_point:.1f}: Memory Bound → Coalescing, Cache 최적화")
    print(f"  - AI > {ridge_point:.1f}: Compute Bound → 연산량 증가, Tensor Core 활용")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """모든 병목 실험 실행"""
    
    print("="*70)
    print("🔬 GPU 병목 분석 실험실")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA가 사용 불가능합니다.")
        return
    
    print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 실험 실행
    experiment_arithmetic_intensity()
    experiment_memory_coalescing()
    experiment_occupancy()
    experiment_warp_divergence()
    experiment_roofline_visualization()
    
    print("\n" + "="*70)
    print("🎓 병목 분석 실험 완료!")
    print("="*70)
    print("📚 핵심 학습:")
    print("  1. Arithmetic Intensity가 병목 유형을 결정한다")
    print("  2. Memory Coalescing은 성능에 극적인 영향을 미친다")
    print("  3. Occupancy는 중요하지만 만능해결책은 아니다")
    print("  4. Warp Divergence는 SIMD 효율성을 크게 저하시킨다")
    print("  5. Roofline Model로 최적화 방향을 결정할 수 있다")
    print("\n🔍 다음: Nsight Compute/Systems를 이용한 정밀 프로파일링!")


if __name__ == "__main__":
    main()