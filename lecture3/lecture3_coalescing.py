"""
Week 3: Memory Coalescing Experiment
메모리 접근 패턴에 따른 성능 차이 체험

목표:
1. Coalesced Access (연속 접근)의 높은 성능 확인
2. Strided Access (보폭 접근)의 성능 저하 체험  
3. Memory Bandwidth가 GPU 성능에 미치는 영향 이해
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

// ✅ GOOD: Coalesced Access (합성 접근)
// Warp 내 32개 스레드가 연속된 메모리 주소에 접근
// → 한 번의 Memory Transaction으로 처리
__global__ void coalesced_read_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        // Thread 0 → input[0], Thread 1 → input[1], ...
        // 연속된 주소 접근 → Memory Controller가 한 번에 처리
        output[i] = input[i] * 2.0f;
    }
}

// ❌ BAD: Strided Access (보폭 접근)  
// Warp 내 32개 스레드가 띄엄띄엄 떨어진 메모리 주소에 접근
// → 여러 번의 Memory Transaction 필요
__global__ void strided_read_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_idx = i * stride;
    
    if (actual_idx < n) {
        // Thread 0 → input[0], Thread 1 → input[stride], Thread 2 → input[2*stride], ...
        // 띄엄띄엄 접근 → Memory Controller가 여러 번 처리해야 함
        output[i] = input[actual_idx] * 2.0f;
    }
}

// 🔬 실험용: Random Access Pattern
// 완전히 랜덤한 패턴으로 메모리 접근 (최악의 경우)
__global__ void random_read_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ indices,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        // 완전히 랜덤한 인덱스로 접근
        int random_idx = indices[i] % n;
        output[i] = input[random_idx] * 2.0f;
    }
}

// C++ Wrapper 함수들
void launch_coalesced(
    torch::Tensor input,
    torch::Tensor output,
    int n,
    int blocks,
    int threads
) {
    coalesced_read_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
}

void launch_strided(
    torch::Tensor input,
    torch::Tensor output,
    int n,
    int stride,
    int blocks,
    int threads
) {
    strided_read_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        stride
    );
}

void launch_random(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor indices,
    int n,
    int blocks,
    int threads
) {
    random_read_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        indices.data_ptr<int>(),
        n
    );
}
"""

cpp_header = """
void launch_coalesced(torch::Tensor input, torch::Tensor output, int n, int blocks, int threads);
void launch_strided(torch::Tensor input, torch::Tensor output, int n, int stride, int blocks, int threads);
void launch_random(torch::Tensor input, torch::Tensor output, torch::Tensor indices, int n, int blocks, int threads);
"""

# JIT 컴파일
print("=" * 60)
print("Compiling Memory Coalescing Test Kernels...")
print("=" * 60)

module = load_inline(
    name='coalescing_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_coalesced', 'launch_strided', 'launch_random'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# 유틸리티 함수들
# =============================================================================

def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division"""
    return (n + divisor - 1) // divisor


def get_memory_bandwidth() -> float:
    """이론적 Memory Bandwidth 계산 (GB/s)"""
    props = torch.cuda.get_device_properties(0)
    # 실제 GPU에 따라 다르지만, 대략적인 값
    memory_clock_khz = getattr(props, 'memory_clock_rate', 7000000)  # 7 GHz 가정
    memory_bus_width = getattr(props, 'memory_bus_width', 256)  # 256-bit 가정
    
    # Bandwidth = Clock * Bus Width * 2 (DDR) / 8 (bits to bytes) / 1e9 (GB)
    theoretical_bw = (memory_clock_khz * 1000) * (memory_bus_width / 8) * 2 / 1e9
    return theoretical_bw


def calculate_achieved_bandwidth(data_size_bytes: int, time_ms: float) -> float:
    """실제 달성한 Memory Bandwidth 계산 (GB/s)"""
    # Read + Write이므로 2배
    total_bytes = data_size_bytes * 2
    time_seconds = time_ms / 1000.0
    return total_bytes / time_seconds / 1e9


# =============================================================================
# 실험 함수들
# =============================================================================

def experiment_coalesced_vs_strided(n: int, iterations: int = 100) -> None:
    """Coalesced vs Strided Access 비교"""
    print("=" * 70)
    print(f"실험 1: Coalesced vs Strided Access (N={n:,})")
    print("=" * 70)
    
    # 데이터 준비
    input_data = torch.rand(n, dtype=torch.float32, device='cuda')
    output_coalesced = torch.zeros(n, dtype=torch.float32, device='cuda')
    output_strided = torch.zeros(n // 4, dtype=torch.float32, device='cuda')  # Stride=4
    
    threads_per_block = 256
    blocks_coalesced = cdiv(n, threads_per_block)
    blocks_strided = cdiv(n // 4, threads_per_block)
    
    data_size_mb = n * 4 / 1024 / 1024  # float32 = 4 bytes
    
    print(f"데이터 크기: {data_size_mb:.1f} MB")
    print(f"스레드 설정: {threads_per_block} threads/block")
    
    # Warm-up
    for _ in range(10):
        module.launch_coalesced(input_data, output_coalesced, n, blocks_coalesced, threads_per_block)
        module.launch_strided(input_data, output_strided, n, 4, blocks_strided, threads_per_block)
    torch.cuda.synchronize()
    
    # 측정 시작
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Coalesced Access 측정
    start.record()
    for _ in range(iterations):
        module.launch_coalesced(input_data, output_coalesced, n, blocks_coalesced, threads_per_block)
    end.record()
    torch.cuda.synchronize()
    coalesced_time = start.elapsed_time(end) / iterations
    
    # Strided Access 측정 (stride = 4)
    start.record()
    for _ in range(iterations):
        module.launch_strided(input_data, output_strided, n, 4, blocks_strided, threads_per_block)
    end.record()
    torch.cuda.synchronize()
    strided_time = start.elapsed_time(end) / iterations
    
    # Bandwidth 계산
    coalesced_bw = calculate_achieved_bandwidth(n * 4, coalesced_time)
    strided_bw = calculate_achieved_bandwidth((n // 4) * 4, strided_time)
    theoretical_bw = get_memory_bandwidth()
    
    # 결과 출력
    print(f"\n결과:")
    print(f"  Coalesced Access:")
    print(f"    시간: {coalesced_time:.4f} ms")
    print(f"    Bandwidth: {coalesced_bw:.1f} GB/s ({coalesced_bw/theoretical_bw*100:.1f}% of theoretical)")
    print(f"  Strided Access (stride=4):")
    print(f"    시간: {strided_time:.4f} ms")
    print(f"    Bandwidth: {strided_bw:.1f} GB/s ({strided_bw/theoretical_bw*100:.1f}% of theoretical)")
    print(f"  속도 차이: {strided_time / coalesced_time:.1f}x slower")
    print(f"  이론적 최대 Bandwidth: {theoretical_bw:.1f} GB/s")


def experiment_different_strides(n: int, iterations: int = 100) -> None:
    """다양한 Stride 패턴 비교"""
    print("\n" + "=" * 70)
    print(f"실험 2: 다양한 Stride 패턴 (N={n:,})")
    print("=" * 70)
    
    input_data = torch.rand(n, dtype=torch.float32, device='cuda')
    threads_per_block = 256
    
    strides = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []
    
    for stride in strides:
        output_size = n // stride
        if output_size < 1000:  # 너무 작으면 스킵
            continue
            
        output = torch.zeros(output_size, dtype=torch.float32, device='cuda')
        blocks = cdiv(output_size, threads_per_block)
        
        # Warm-up
        for _ in range(5):
            module.launch_strided(input_data, output, n, stride, blocks, threads_per_block)
        torch.cuda.synchronize()
        
        # 측정
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            module.launch_strided(input_data, output, n, stride, blocks, threads_per_block)
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / iterations
        bandwidth = calculate_achieved_bandwidth(output_size * 4, time_ms)
        results.append((stride, time_ms, bandwidth))
    
    # 결과 출력
    print(f"\n{'Stride':<8} {'Time (ms)':<12} {'Bandwidth (GB/s)':<18} {'Efficiency (%)':<15}")
    print("-" * 60)
    
    theoretical_bw = get_memory_bandwidth()
    for stride, time_ms, bandwidth in results:
        efficiency = bandwidth / theoretical_bw * 100
        print(f"{stride:<8} {time_ms:<12.4f} {bandwidth:<18.1f} {efficiency:<15.1f}")


def experiment_random_access(n: int, iterations: int = 100) -> None:
    """Random Access Pattern 실험"""
    print("\n" + "=" * 70)
    print(f"실험 3: Random Access Pattern (N={n:,})")
    print("=" * 70)
    
    input_data = torch.rand(n, dtype=torch.float32, device='cuda')
    output = torch.zeros(n, dtype=torch.float32, device='cuda')
    
    # 랜덤 인덱스 생성 (최악의 경우 시뮬레이션)
    indices = torch.randint(0, n, (n,), dtype=torch.int32, device='cuda')
    
    threads_per_block = 256
    blocks = cdiv(n, threads_per_block)
    
    # Warm-up
    for _ in range(5):
        module.launch_random(input_data, output, indices, n, blocks, threads_per_block)
    torch.cuda.synchronize()
    
    # 측정
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        module.launch_random(input_data, output, indices, n, blocks, threads_per_block)
    end.record()
    torch.cuda.synchronize()
    
    random_time = start.elapsed_time(end) / iterations
    random_bw = calculate_achieved_bandwidth(n * 4, random_time)
    theoretical_bw = get_memory_bandwidth()
    
    print(f"\nRandom Access 결과:")
    print(f"  시간: {random_time:.4f} ms")
    print(f"  Bandwidth: {random_bw:.1f} GB/s ({random_bw/theoretical_bw*100:.1f}% of theoretical)")
    print(f"  → Cache Miss가 많이 발생하여 극도로 느림!")


def explain_memory_hierarchy() -> None:
    """Memory Hierarchy 설명"""
    print("\n" + "=" * 70)
    print("GPU Memory Hierarchy 이해하기")
    print("=" * 70)
    print("""
🏔️ GPU Memory Pyramid (위로 갈수록 빠름, 아래로 갈수록 큼)

    Register (32KB/SM)      ← 1 cycle, Thread 전용
         ↑
   Shared Memory (48KB)     ← 1~32 cycles, Block 공유  ⭐ 프로그래머 제어!
         ↑
   L1/L2 Cache (수 MB)     ← 32~200 cycles, 하드웨어 관리
         ↑
  Global Memory (수십 GB)   ← 200~800 cycles, 모든 Grid 공유  ⚠️ 병목!

핵심 원리:
1. 🚀 Coalesced Access: Warp의 32개 스레드가 연속된 주소에 접근
2. ⚡ Data Reuse: 자주 쓰는 데이터를 Shared Memory에 캐시
3. 🧠 Bandwidth vs Latency: GPU는 Bandwidth 머신, 지연시간은 숨김

Memory Coalescing 조건:
- 32개 스레드가 연속된 32×4 = 128 bytes에 접근
- 주소가 128 bytes로 정렬(aligned)되어 있어야 함
- Stride=1이 이상적, Stride > 32면 매우 비효율적
    """)


if __name__ == "__main__":
    print("\n" + "🚀 Week 3: Memory Coalescing Experiment".center(70))
    print("=" * 70)
    
    # GPU 정보
    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    global_mem_gb = props.total_memory / 1024**3
    
    print(f"\n🖥️  GPU: {device_name}")
    print(f"   Global Memory: {global_mem_gb:.1f} GB")
    print(f"   이론적 Memory Bandwidth: ~{get_memory_bandwidth():.0f} GB/s")
    
    # Memory Hierarchy 설명
    explain_memory_hierarchy()
    
    # 실험 1: Coalesced vs Strided
    experiment_coalesced_vs_strided(n=10_000_000, iterations=100)
    
    # 실험 2: 다양한 Stride 패턴
    experiment_different_strides(n=20_000_000, iterations=50)
    
    # 실험 3: Random Access (최악의 경우)
    experiment_random_access(n=1_000_000, iterations=50)
    
    print("\n" + "=" * 70)
    print("🎓 핵심 교훈")
    print("=" * 70)
    print("""
1. 💡 Memory가 GPU의 가장 큰 병목이다
2. 🎯 Coalesced Access가 성능의 핵심이다
3. 📊 Stride가 클수록 성능이 급격히 떨어진다  
4. 🔥 Random Access는 최악 - 피해야 한다
5. ⚡ 다음 시간에 배울 Shared Memory로 이 문제를 해결한다!
    """)
    
    print("\n" + "=" * 70)
    print("🎉 Week 3-1: Memory Coalescing 실험 완료!")
    print("=" * 70 + "\n")
