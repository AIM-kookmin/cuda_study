"""
Week 2 과제 3: Warp Divergence 실험

목표: 분기(Branch)가 성능에 미치는 악영향 직접 확인

실험 내용:
- Kernel A (Bad): if (threadIdx.x % 2 == 0) - 짝수/홀수 분기 (Warp 내에서 발산)
- Kernel B (Good): if (threadIdx.x < 16) - Warp 단위 분기 (Warp 전체가 같은 경로)

예상 결과: Kernel A가 더 느림 (Warp Divergence로 인한 직렬화)
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"


# =============================================================================
# CUDA 커널 소스코드
# =============================================================================

cuda_source = """
#include <torch/extension.h>

// Kernel A (BAD): 짝수/홀수 분기 - Warp Divergence 발생!
// 한 Warp(32 threads) 내에서 절반은 A 경로, 절반은 B 경로
// → 직렬화로 인해 성능 저하
__global__ void divergent_bad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        if (threadIdx.x % 2 == 0) {
            // 짝수 스레드: 경로 A (복잡한 계산)
            float val = input[i];
            for (int j = 0; j < 100; j++) {
                val = val * 1.01f + 0.5f;
            }
            output[i] = val;
        } else {
            // 홀수 스레드: 경로 B (복잡한 계산)
            float val = input[i];
            for (int j = 0; j < 100; j++) {
                val = val * 0.99f - 0.5f;
            }
            output[i] = val;
        }
    }
}

// Kernel B (GOOD): Warp 단위 분기 - Divergence 없음!
// 각 Warp(32 threads)가 통째로 같은 경로를 선택
// Warp 0-15: 전부 경로 A, Warp 16-31: 전부 경로 B
__global__ void divergent_good_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        // threadIdx.x < 16이면 한 Warp의 절반이 같은 경로
        // 하지만 blockDim.x=32 가정 시, 이것도 문제
        // 더 나은 방법: threadIdx.x / 32 (Warp ID)로 분기
        int warp_id = threadIdx.x / 32;
        
        if (warp_id % 2 == 0) {
            // Warp 단위로 경로 A
            float val = input[i];
            for (int j = 0; j < 100; j++) {
                val = val * 1.01f + 0.5f;
            }
            output[i] = val;
        } else {
            // Warp 단위로 경로 B
            float val = input[i];
            for (int j = 0; j < 100; j++) {
                val = val * 0.99f - 0.5f;
            }
            output[i] = val;
        }
    }
}

// 간단한 버전 (더 명확한 비교용)
// Kernel A Simple: 짝수/홀수 분기
__global__ void divergent_bad_simple(
    float* data,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        if (threadIdx.x % 2 == 0) {
            // 짝수: 경로 A
            data[i] = data[i] * 2.0f;
        } else {
            // 홀수: 경로 B
            data[i] = data[i] * 3.0f;
        }
    }
}

// Kernel B Simple: Warp 단위 분기
__global__ void divergent_good_simple(
    float* data,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        int warp_id = threadIdx.x / 32;
        
        if (warp_id % 2 == 0) {
            // Warp 0, 2, 4, ...: 경로 A
            data[i] = data[i] * 2.0f;
        } else {
            // Warp 1, 3, 5, ...: 경로 B
            data[i] = data[i] * 3.0f;
        }
    }
}

// C++ Wrapper 함수들
void launch_bad(torch::Tensor input, torch::Tensor output, int n, int blocks, int threads) {
    divergent_bad_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
}

void launch_good(torch::Tensor input, torch::Tensor output, int n, int blocks, int threads) {
    divergent_good_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
}

void launch_bad_simple(torch::Tensor data, int n, int blocks, int threads) {
    divergent_bad_simple<<<blocks, threads>>>(data.data_ptr<float>(), n);
}

void launch_good_simple(torch::Tensor data, int n, int blocks, int threads) {
    divergent_good_simple<<<blocks, threads>>>(data.data_ptr<float>(), n);
}
"""

cpp_header = """
void launch_bad(torch::Tensor input, torch::Tensor output, int n, int blocks, int threads);
void launch_good(torch::Tensor input, torch::Tensor output, int n, int blocks, int threads);
void launch_bad_simple(torch::Tensor data, int n, int blocks, int threads);
void launch_good_simple(torch::Tensor data, int n, int blocks, int threads);
"""

# JIT 컴파일
print("=" * 60)
print("Compiling Warp Divergence Test Kernels...")
print("=" * 60)

module = load_inline(
    name='warp_divergence_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_bad', 'launch_good', 'launch_bad_simple', 'launch_good_simple'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# 실험 함수들
# =============================================================================

def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division"""
    return (n + divisor - 1) // divisor


def benchmark_complex_kernels(n: int, iterations: int = 1000) -> None:
    """복잡한 계산을 수행하는 커널 벤치마크"""
    print("=" * 60)
    print(f"실험 1: 복잡한 계산 (N={n:,}, iterations={iterations})")
    print("=" * 60)
    
    input_data = torch.rand(n, dtype=torch.float32, device='cuda')
    output_bad = torch.zeros_like(input_data)
    output_good = torch.zeros_like(input_data)
    
    threads_per_block = 256
    blocks = cdiv(n, threads_per_block)
    
    print(f"\n설정:")
    print(f"  • threads_per_block: {threads_per_block}")
    print(f"  • blocks_per_grid: {blocks}")
    
    # Warm-up
    for _ in range(10):
        module.launch_bad(input_data, output_bad, n, blocks, threads_per_block)
        module.launch_good(input_data, output_good, n, blocks, threads_per_block)
    torch.cuda.synchronize()
    
    # 시간 측정
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Kernel A (BAD): 짝수/홀수 분기
    start.record()
    for _ in range(iterations):
        module.launch_bad(input_data, output_bad, n, blocks, threads_per_block)
    end.record()
    torch.cuda.synchronize()
    bad_time = start.elapsed_time(end) / iterations
    
    # Kernel B (GOOD): Warp 단위 분기
    start.record()
    for _ in range(iterations):
        module.launch_good(input_data, output_good, n, blocks, threads_per_block)
    end.record()
    torch.cuda.synchronize()
    good_time = start.elapsed_time(end) / iterations
    
    # 결과 출력
    print("\n결과:")
    print(f"  Kernel A (BAD - 짝수/홀수 분기):  {bad_time:.4f} ms")
    print(f"  Kernel B (GOOD - Warp 단위 분기): {good_time:.4f} ms")
    print(f"  속도 차이: {bad_time / good_time:.2f}x")
    
    if bad_time > good_time:
        print(f"\n✅ 예상대로 Kernel A가 느립니다! (Warp Divergence 확인)")
        print(f"   → Kernel A는 각 Warp 내에서 스레드가 다른 경로를 타서 직렬화됨")
    else:
        print(f"\n⚠️  예상과 다른 결과입니다. 컴파일러 최적화가 개입했을 수 있습니다.")


def benchmark_simple_kernels(n: int, iterations: int = 1000) -> None:
    """간단한 계산을 수행하는 커널 벤치마크"""
    print("\n" + "=" * 60)
    print(f"실험 2: 간단한 계산 (N={n:,}, iterations={iterations})")
    print("=" * 60)
    
    data_bad = torch.ones(n, dtype=torch.float32, device='cuda')
    data_good = torch.ones(n, dtype=torch.float32, device='cuda')
    
    threads_per_block = 256
    blocks = cdiv(n, threads_per_block)
    
    # Warm-up
    for _ in range(10):
        module.launch_bad_simple(data_bad, n, blocks, threads_per_block)
        module.launch_good_simple(data_good, n, blocks, threads_per_block)
    torch.cuda.synchronize()
    
    # 시간 측정
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Simple Bad
    start.record()
    for _ in range(iterations):
        module.launch_bad_simple(data_bad, n, blocks, threads_per_block)
    end.record()
    torch.cuda.synchronize()
    bad_time = start.elapsed_time(end) / iterations
    
    # Simple Good
    start.record()
    for _ in range(iterations):
        module.launch_good_simple(data_good, n, blocks, threads_per_block)
    end.record()
    torch.cuda.synchronize()
    good_time = start.elapsed_time(end) / iterations
    
    # 결과 출력
    print("\n결과:")
    print(f"  Kernel A Simple (BAD):  {bad_time:.4f} ms")
    print(f"  Kernel B Simple (GOOD): {good_time:.4f} ms")
    print(f"  속도 차이: {bad_time / good_time:.2f}x")


def explain_warp_divergence() -> None:
    """Warp Divergence 설명"""
    print("\n" + "=" * 60)
    print("Warp Divergence란?")
    print("=" * 60)
    print("""
GPU는 32개의 스레드를 묶어서 "Warp"라는 단위로 실행합니다.
같은 Warp의 모든 스레드는 **동시에 같은 명령어**를 실행해야 합니다.

[BAD 패턴] - Warp 내에서 분기:
```
if (threadIdx.x % 2 == 0) {
    A();  // 짝수 스레드
} else {
    B();  // 홀수 스레드
}
```

Warp 0 (Thread 0-31):
  Thread 0, 2, 4, ... → A() 실행 (절반)
  Thread 1, 3, 5, ... → B() 실행 (절반)
  
→ Warp가 찢어집니다! (Divergence)
→ 하드웨어는 A()를 먼저 실행하고, B()를 나중에 실행 (직렬화)
→ 이론상 2배 느림

[GOOD 패턴] - Warp 단위로 분기:
```
int warp_id = threadIdx.x / 32;
if (warp_id % 2 == 0) {
    A();  // Warp 0, 2, 4, ...
} else {
    B();  // Warp 1, 3, 5, ...
}
```

Warp 0 전체 → A() 실행
Warp 1 전체 → B() 실행

→ 각 Warp 내에서는 모두 같은 경로!
→ Divergence 없음 → 빠름!
    """)


if __name__ == "__main__":
    print("\n" + "🚀 Week 2 과제 3: Warp Divergence 실험".center(60))
    print("=" * 60)
    
    # GPU 정보
    device_name = torch.cuda.get_device_name(0)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    warp_size = 32  # NVIDIA GPU는 항상 32
    
    print(f"\n🖥️  GPU: {device_name}")
    print(f"   SM 개수: {sm_count}")
    print(f"   Warp 크기: {warp_size}")
    
    # 설명
    explain_warp_divergence()
    
    # 실험 1: 복잡한 계산
    benchmark_complex_kernels(n=1_000_000, iterations=1000)
    
    # 실험 2: 간단한 계산
    benchmark_simple_kernels(n=10_000_000, iterations=1000)
    
    print("\n" + "=" * 60)
    print("🎉 Week 2 과제 3 완료!")
    print("=" * 60)
    print("""
핵심 교훈:
1. Warp 내에서 분기하지 마세요 (if (threadIdx.x % 2))
2. 분기가 필요하면 Warp 단위로 하세요 (if (threadIdx.x / 32))
3. 실제 성능 차이를 프로파일링으로 항상 확인하세요
    """)
