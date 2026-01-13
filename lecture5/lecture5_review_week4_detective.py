"""
Week 4 복습: Performance Detective Game 🕵️

학습 목표:
1. 주어진 느린 커널의 병목을 찾아내는 능력 배양
2. NCU Profiling 메트릭 해석 실전 연습
3. 최적화 전후 성능 비교 및 분석
4. 실무에서 자주 발생하는 성능 문제 패턴 학습

시나리오 구성:
- Scenario A: Low Occupancy (레지스터 사용 과다)
- Scenario B: Poor Memory Coalescing (Strided Access)
- Scenario C: Warp Divergence (과도한 Branch)

각 시나리오마다 "Broken" 버전과 "Fixed" 버전 제공
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
// Scenario A: Low Occupancy Problem
// =============================================================================

// BROKEN: 레지스터를 너무 많이 사용하여 Occupancy 저하
__global__ void vector_compute_broken_a(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 불필요하게 많은 지역 변수 (레지스터 낭비)
        float temp0 = input[idx];
        float temp1 = temp0 * 2.0f;
        float temp2 = temp1 + 3.0f;
        float temp3 = temp2 * temp0;
        float temp4 = temp3 - 1.0f;
        float temp5 = temp4 * temp4;
        float temp6 = temp5 + temp0;
        float temp7 = temp6 * 0.5f;
        float temp8 = temp7 - temp3;
        float temp9 = temp8 * temp8;
        float temp10 = temp9 + temp4;
        float temp11 = temp10 * temp1;
        float temp12 = temp11 - temp2;
        float temp13 = temp12 * temp12;
        float temp14 = temp13 + temp5;
        float temp15 = temp14 * 0.3f;
        
        // 더 많은 불필요한 연산...
        float result = temp15;
        for (int i = 0; i < 10; i++) {
            result = result * 1.01f + 0.001f;
        }
        
        output[idx] = result;
    }
}

// FIXED: 연산 최적화, 레지스터 사용 감소
__global__ void vector_compute_fixed_a(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        // 동일한 결과를 더 적은 레지스터로 계산
        float result = (x * 2.0f + 3.0f) * x - 1.0f;
        result = result * result + x;
        result = result * 0.5f - ((x * 2.0f + 3.0f) * x);
        result = result * result;
        
        // 불필요한 루프 제거 (컴파일 타임 계산 가능)
        result = result * 1.104622f + 0.01f;  // 10회 반복 결과 사전 계산
        
        output[idx] = result;
    }
}

// =============================================================================
// Scenario B: Poor Memory Coalescing
// =============================================================================

// BROKEN: Strided Memory Access로 인한 비효율
__global__ void strided_copy_broken_b(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 나쁜 접근 패턴: Stride 간격으로 읽기/쓰기
        // 같은 Warp의 Thread들이 멀리 떨어진 메모리 접근
        int strided_idx = idx * stride;
        if (strided_idx < n * stride) {
            output[strided_idx] = input[strided_idx] * 2.0f;
        }
    }
}

// FIXED: Coalesced Access로 개선
__global__ void strided_copy_fixed_b(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 개선된 접근 패턴: 연속적으로 읽고, 계산 후 Strided 쓰기
        // 또는 Shared Memory를 활용한 재배열
        float value = input[idx] * 2.0f;  // Coalesced read
        
        int strided_idx = idx * stride;
        if (strided_idx < n * stride) {
            output[strided_idx] = value;  // Write는 어쩔 수 없지만 Read는 개선
        }
    }
}

// FIXED (Better): Shared Memory를 활용한 완전한 해결
__global__ void strided_copy_fixed_b_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    int stride
) {
    __shared__ float tile[256];  // Shared Memory 버퍼
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Phase 1: Coalesced Read
    if (idx < n) {
        tile[tid] = input[idx] * 2.0f;
    }
    __syncthreads();
    
    // Phase 2: Strided Write (피할 수 없지만 Global 접근 횟수 최소화)
    if (idx < n) {
        int strided_idx = idx * stride;
        if (strided_idx < n * stride) {
            output[strided_idx] = tile[tid];
        }
    }
}

// =============================================================================
// Scenario C: Warp Divergence
// =============================================================================

// BROKEN: 과도한 Branch로 인한 Warp Divergence
__global__ void conditional_compute_broken_c(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        float result;
        
        // 나쁜 패턴: Warp 내에서 Thread마다 다른 경로
        if (idx % 32 == 0) {
            result = x * x * x;  // 1/32 threads
        } else if (idx % 16 == 0) {
            result = x * x;  // 1/16 threads
        } else if (idx % 8 == 0) {
            result = x * 2.0f;  // 1/8 threads
        } else if (idx % 4 == 0) {
            result = x + 1.0f;  // 1/4 threads
        } else if (idx % 2 == 0) {
            result = x - 1.0f;  // 1/2 threads
        } else {
            result = x;  // Remaining threads
        }
        
        output[idx] = result;
    }
}

// FIXED: Branch 최소화, Predication 활용
__global__ void conditional_compute_fixed_c(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        
        // 개선된 패턴: Predication을 활용하여 모든 Thread가 같은 코드 실행
        int mod32 = idx % 32;
        int mod16 = idx % 16;
        int mod8 = idx % 8;
        int mod4 = idx % 4;
        int mod2 = idx % 2;
        
        // 조건부 연산을 곱셈/덧셈으로 변환 (Branch 제거)
        float result = x;
        result = (mod32 == 0) ? (x * x * x) : result;
        result = (mod32 != 0 && mod16 == 0) ? (x * x) : result;
        result = (mod16 != 0 && mod8 == 0) ? (x * 2.0f) : result;
        result = (mod8 != 0 && mod4 == 0) ? (x + 1.0f) : result;
        result = (mod4 != 0 && mod2 == 0) ? (x - 1.0f) : result;
        
        output[idx] = result;
    }
}

// Launcher Functions
void launch_scenario_a_broken(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_compute_broken_a<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_scenario_a_fixed(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_compute_fixed_a<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_scenario_b_broken(torch::Tensor input, torch::Tensor output, int stride) {
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    strided_copy_broken_b<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n, stride);
}

void launch_scenario_b_fixed(torch::Tensor input, torch::Tensor output, int stride) {
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    strided_copy_fixed_b<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n, stride);
}

void launch_scenario_b_fixed_shared(torch::Tensor input, torch::Tensor output, int stride) {
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    strided_copy_fixed_b_shared<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n, stride);
}

void launch_scenario_c_broken(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    conditional_compute_broken_c<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_scenario_c_fixed(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    conditional_compute_fixed_c<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_source = """
void launch_scenario_a_broken(torch::Tensor input, torch::Tensor output);
void launch_scenario_a_fixed(torch::Tensor input, torch::Tensor output);
void launch_scenario_b_broken(torch::Tensor input, torch::Tensor output, int stride);
void launch_scenario_b_fixed(torch::Tensor input, torch::Tensor output, int stride);
void launch_scenario_b_fixed_shared(torch::Tensor input, torch::Tensor output, int stride);
void launch_scenario_c_broken(torch::Tensor input, torch::Tensor output);
void launch_scenario_c_fixed(torch::Tensor input, torch::Tensor output);
"""

# JIT 컴파일
detective_module = load_inline(
    name='detective_module',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=[
        'launch_scenario_a_broken', 'launch_scenario_a_fixed',
        'launch_scenario_b_broken', 'launch_scenario_b_fixed', 'launch_scenario_b_fixed_shared',
        'launch_scenario_c_broken', 'launch_scenario_c_fixed'
    ],
    verbose=False
)

# ============================================================
# Performance Detective Functions
# ============================================================

def benchmark_scenario_a(n: int = 10_000_000, iterations: int = 100) -> None:
    """Scenario A: Low Occupancy 문제"""
    print("=" * 70)
    print("🕵️ Scenario A: The Case of Low Occupancy")
    print("=" * 70)
    print(f"문제: 커널이 예상보다 훨씬 느립니다. 왜 그럴까요?")
    print(f"데이터 크기: {n:,} elements")
    print()
    
    input_tensor = torch.randn(n, device='cuda', dtype=torch.float32)
    output_broken = torch.empty_like(input_tensor)
    output_fixed = torch.empty_like(input_tensor)
    
    # Warm-up
    for _ in range(10):
        detective_module.launch_scenario_a_broken(input_tensor, output_broken)
        detective_module.launch_scenario_a_fixed(input_tensor, output_fixed)
    torch.cuda.synchronize()
    
    # Benchmark Broken
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        detective_module.launch_scenario_a_broken(input_tensor, output_broken)
    end.record()
    torch.cuda.synchronize()
    time_broken = start.elapsed_time(end) / iterations
    
    # Benchmark Fixed
    start.record()
    for _ in range(iterations):
        detective_module.launch_scenario_a_fixed(input_tensor, output_fixed)
    end.record()
    torch.cuda.synchronize()
    time_fixed = start.elapsed_time(end) / iterations
    
    # Results
    speedup = time_broken / time_fixed
    print(f"❌ BROKEN: {time_broken:.4f} ms")
    print(f"✅ FIXED:  {time_fixed:.4f} ms")
    print(f"⚡ Speedup: {speedup:.2f}x")
    print()
    print("🔍 원인:")
    print("  - 과도한 지역 변수 사용으로 레지스터 낭비")
    print("  - 레지스터 부족 → Spilling to Local Memory")
    print("  - 결과적으로 Occupancy 저하")
    print()
    print("💊 해결책:")
    print("  - 불필요한 중간 변수 제거")
    print("  - 연산 체인 최적화")
    print("  - 컴파일러가 레지스터를 효율적으로 사용하도록 도움")
    print()


def benchmark_scenario_b(n: int = 1_000_000, stride: int = 8, iterations: int = 100) -> None:
    """Scenario B: Poor Memory Coalescing 문제"""
    print("=" * 70)
    print("🕵️ Scenario B: The Case of Poor Memory Coalescing")
    print("=" * 70)
    print(f"문제: 간단한 Copy 연산인데 왜 이렇게 느릴까요?")
    print(f"데이터 크기: {n:,} elements, Stride: {stride}")
    print()
    
    input_tensor = torch.randn(n, device='cuda', dtype=torch.float32)
    output_broken = torch.zeros(n * stride, device='cuda', dtype=torch.float32)
    output_fixed = torch.zeros(n * stride, device='cuda', dtype=torch.float32)
    output_fixed_shared = torch.zeros(n * stride, device='cuda', dtype=torch.float32)
    
    # Warm-up
    for _ in range(10):
        detective_module.launch_scenario_b_broken(input_tensor, output_broken, stride)
        detective_module.launch_scenario_b_fixed(input_tensor, output_fixed, stride)
        detective_module.launch_scenario_b_fixed_shared(input_tensor, output_fixed_shared, stride)
    torch.cuda.synchronize()
    
    # Benchmark
    methods = [
        ("BROKEN (Strided R/W)", lambda: detective_module.launch_scenario_b_broken(input_tensor, output_broken, stride)),
        ("FIXED (Coalesced R)", lambda: detective_module.launch_scenario_b_fixed(input_tensor, output_fixed, stride)),
        ("FIXED (Shared Mem)", lambda: detective_module.launch_scenario_b_fixed_shared(input_tensor, output_fixed_shared, stride))
    ]
    
    results = []
    for name, func in methods:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            func()
        end.record()
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / iterations
        results.append((name, time_ms))
    
    print(f"{'Method':<25} {'Time (ms)':<12} {'Speedup'}")
    print("-" * 70)
    baseline = results[0][1]
    for name, time_ms in results:
        speedup = baseline / time_ms
        print(f"{name:<25} {time_ms:>10.4f}   {speedup:>6.2f}x")
    
    print()
    print("🔍 원인:")
    print("  - Strided 접근으로 인한 비효율적 메모리 트랜잭션")
    print("  - 같은 Warp의 Thread들이 멀리 떨어진 메모리 접근")
    print("  - Cache Line 낭비")
    print()
    print("💊 해결책:")
    print("  - Read는 Coalesced, Write는 어쩔 수 없음")
    print("  - Shared Memory를 중간 버퍼로 활용")
    print("  - 가능하면 메모리 레이아웃 재설계")
    print()


def benchmark_scenario_c(n: int = 10_000_000, iterations: int = 100) -> None:
    """Scenario C: Warp Divergence 문제"""
    print("=" * 70)
    print("🕵️ Scenario C: The Case of Warp Divergence")
    print("=" * 70)
    print(f"문제: Branch가 많은 커널이 느립니다. 어떻게 개선할까요?")
    print(f"데이터 크기: {n:,} elements")
    print()
    
    input_tensor = torch.randn(n, device='cuda', dtype=torch.float32)
    output_broken = torch.empty_like(input_tensor)
    output_fixed = torch.empty_like(input_tensor)
    
    # Warm-up
    for _ in range(10):
        detective_module.launch_scenario_c_broken(input_tensor, output_broken)
        detective_module.launch_scenario_c_fixed(input_tensor, output_fixed)
    torch.cuda.synchronize()
    
    # Benchmark Broken
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        detective_module.launch_scenario_c_broken(input_tensor, output_broken)
    end.record()
    torch.cuda.synchronize()
    time_broken = start.elapsed_time(end) / iterations
    
    # Benchmark Fixed
    start.record()
    for _ in range(iterations):
        detective_module.launch_scenario_c_fixed(input_tensor, output_fixed)
    end.record()
    torch.cuda.synchronize()
    time_fixed = start.elapsed_time(end) / iterations
    
    # Results
    speedup = time_broken / time_fixed
    print(f"❌ BROKEN: {time_broken:.4f} ms")
    print(f"✅ FIXED:  {time_fixed:.4f} ms")
    print(f"⚡ Speedup: {speedup:.2f}x")
    print()
    print("🔍 원인:")
    print("  - 과도한 if-else 분기로 Warp Divergence 발생")
    print("  - Warp 내 Thread들이 다른 경로 실행")
    print("  - Serial Execution → 성능 저하")
    print()
    print("💊 해결책:")
    print("  - Branch를 Predication으로 변환 (조건부 이동)")
    print("  - SIMD-friendly 코드 작성")
    print("  - 가능하면 데이터 기반 분기 회피")
    print()


def print_ncu_guide() -> None:
    """NCU Profiling 가이드"""
    print("=" * 70)
    print("🔬 NCU Profiling 가이드 - Detective Mode")
    print("=" * 70)
    print()
    print("📌 Scenario A (Low Occupancy) 진단:")
    print("  ncu --metrics achieved_occupancy,\\")
    print("               inst_per_warp,\\")
    print("               l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld \\")
    print("      python lecture5_review_week4_detective.py")
    print()
    print("📌 Scenario B (Memory Coalescing) 진단:")
    print("  ncu --metrics gld_efficiency,gst_efficiency,\\")
    print("               l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\\")
    print("               l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \\")
    print("      python lecture5_review_week4_detective.py")
    print()
    print("📌 Scenario C (Warp Divergence) 진단:")
    print("  ncu --metrics smsp__sass_branch_targets_threads_divergent.avg,\\")
    print("               smsp__sass_inst_executed_per_inst_issued \\")
    print("      python lecture5_review_week4_detective.py")
    print()


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🕵️ Week 4 복습: Performance Detective Game")
    print("=" * 70)
    print()
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("당신은 성능 탐정입니다. 느린 커널의 병목을 찾아 최적화하세요!")
    print()
    
    # Scenario A: Low Occupancy
    benchmark_scenario_a()
    
    # Scenario B: Poor Coalescing
    benchmark_scenario_b()
    
    # Scenario C: Warp Divergence
    benchmark_scenario_c()
    
    # NCU 가이드
    print_ncu_guide()
    
    print("=" * 70)
    print("✅ Performance Detective Game 완료!")
    print("=" * 70)
    print()
    print("🎓 학습 포인트:")
    print("  1. Occupancy: 레지스터 사용량이 성능에 미치는 영향")
    print("  2. Memory Coalescing: 접근 패턴 최적화의 중요성")
    print("  3. Warp Divergence: Branch 최소화 기법")
    print("  4. 실전 Profiling으로 병목 진단 능력 향상")
    print()
