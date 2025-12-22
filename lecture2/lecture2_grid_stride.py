"""
Week 2 실습 2: Grid-Stride Loop (확장 가능한 커널 패턴)

이 스크립트는 두 가지 Vector Addition 구현을 비교합니다:
1. Monolithic 방식 (Week 1): 데이터 1개당 스레드 1개
2. Grid-Stride Loop 방식 (Week 2): 스레드가 여러 데이터를 처리

Grid-Stride Loop의 장점:
- 데이터 크기에 상관없이 동작 (확장성)
- 고정된 Grid 크기로 최적 성능 유지
- 디버깅 용이 (Grid 크기를 1로 줄여서 순차 실행 가능)
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"


def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division (올림 나눗셈)"""
    return (n + divisor - 1) // divisor


# =============================================================================
# CUDA 커널 소스코드
# =============================================================================

cuda_source = """
#include <torch/extension.h>

// -----------------------------------------------------------------------------
// 방식 1: Monolithic (Week 1 스타일)
// - 데이터 1개당 스레드 1개
// - Grid 크기 = ceil(N / block_size)
// - 문제: N이 매우 크면 Grid 크기 제한에 걸릴 수 있음
// -----------------------------------------------------------------------------
__global__ void vector_add_monolithic(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    // 글로벌 인덱스 계산
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 범위 체크 (Boundary Check)
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// -----------------------------------------------------------------------------
// 방식 2: Grid-Stride Loop (Week 2 스타일) ⭐
// - 스레드가 stride만큼 건너뛰며 여러 데이터 처리
// - Grid 크기를 고정해도 됨 (예: SM 개수 × 4)
// - 데이터 크기에 상관없이 동작 (확장성)
// -----------------------------------------------------------------------------
__global__ void vector_add_grid_stride(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    // 시작 인덱스: 이 스레드의 글로벌 ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // stride: 전체 스레드 수 (Grid의 모든 스레드)
    int stride = blockDim.x * gridDim.x;
    
    // stride만큼 건너뛰면서 끝까지 처리
    // 예: 스레드 0은 index 0, stride, 2*stride, ... 를 처리
    for (int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// -----------------------------------------------------------------------------
// Python에서 호출할 Wrapper 함수들
// -----------------------------------------------------------------------------

void launch_monolithic(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    int n,
    int blocks,
    int threads
) {
    vector_add_monolithic<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
}

void launch_grid_stride(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    int n,
    int blocks,
    int threads
) {
    vector_add_grid_stride<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
}
"""

cpp_header = """
void launch_monolithic(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    int n,
    int blocks,
    int threads
);

void launch_grid_stride(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    int n,
    int blocks,
    int threads
);
"""

# JIT 컴파일
print("=" * 60)
print("Compiling CUDA kernels...")
print("=" * 60)

module = load_inline(
    name='vector_add_comparison',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_monolithic', 'launch_grid_stride'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# Python Wrapper 함수들
# =============================================================================

def vector_add_monolithic(a: torch.Tensor, b: torch.Tensor, threads_per_block: int = 256) -> torch.Tensor:
    """
    Monolithic 방식의 Vector Addition
    Grid 크기 = ceil(N / threads_per_block)
    """
    n = a.numel()
    c = torch.zeros_like(a)
    
    blocks = cdiv(n, threads_per_block)
    module.launch_monolithic(a, b, c, n, blocks, threads_per_block)
    
    return c


def vector_add_grid_stride(
    a: torch.Tensor, 
    b: torch.Tensor, 
    threads_per_block: int = 256,
    blocks_per_grid: int = None
) -> torch.Tensor:
    """
    Grid-Stride Loop 방식의 Vector Addition
    Grid 크기를 고정 (기본값: SM 개수 × 4)
    """
    n = a.numel()
    c = torch.zeros_like(a)
    
    # Grid 크기 설정 (기본값: SM 개수 × 4)
    if blocks_per_grid is None:
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        blocks_per_grid = sm_count * 4
    
    module.launch_grid_stride(a, b, c, n, blocks_per_grid, threads_per_block)
    
    return c


def verify_correctness(a: torch.Tensor, b: torch.Tensor) -> None:
    """두 방식의 결과가 PyTorch와 일치하는지 검증"""
    print("=" * 60)
    print("정확성 검증")
    print("=" * 60)
    
    # 정답 (PyTorch)
    expected = a + b
    
    # Monolithic 방식
    result_mono = vector_add_monolithic(a, b)
    torch.cuda.synchronize()
    is_correct_mono = torch.allclose(result_mono, expected)
    
    # Grid-Stride 방식
    result_stride = vector_add_grid_stride(a, b)
    torch.cuda.synchronize()
    is_correct_stride = torch.allclose(result_stride, expected)
    
    print(f"Monolithic 방식: {'✅ 정확' if is_correct_mono else '❌ 오류'}")
    print(f"Grid-Stride 방식: {'✅ 정확' if is_correct_stride else '❌ 오류'}")
    
    if is_correct_mono and is_correct_stride:
        print("\n🎉 두 방식 모두 정확합니다!")
    else:
        print("\n⚠️ 결과가 일치하지 않습니다!")


def benchmark_comparison(n: int, iterations: int = 100) -> None:
    """두 방식의 성능 비교"""
    print("\n" + "=" * 60)
    print(f"성능 벤치마크 (N = {n:,}, {iterations} iterations)")
    print("=" * 60)
    
    # 데이터 생성
    a = torch.rand(n, dtype=torch.float32, device='cuda')
    b = torch.rand(n, dtype=torch.float32, device='cuda')
    
    # GPU 정보
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    threads_per_block = 256
    
    print(f"\nSM 개수: {sm_count}")
    print(f"Block당 스레드: {threads_per_block}")
    
    # Warm-up
    for _ in range(10):
        _ = vector_add_monolithic(a, b)
        _ = vector_add_grid_stride(a, b)
    torch.cuda.synchronize()
    
    # CUDA 이벤트 생성
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    results = {}
    
    # 1. Monolithic 방식
    blocks_mono = cdiv(n, threads_per_block)
    start.record()
    for _ in range(iterations):
        c = torch.zeros_like(a)
        module.launch_monolithic(a, b, c, n, blocks_mono, threads_per_block)
    end.record()
    torch.cuda.synchronize()
    mono_time = start.elapsed_time(end) / iterations
    results['Monolithic'] = (mono_time, blocks_mono)
    
    # 2. Grid-Stride 방식 (다양한 Grid 크기 테스트)
    grid_multipliers = [1, 2, 4, 8]
    
    for mult in grid_multipliers:
        blocks_stride = sm_count * mult
        start.record()
        for _ in range(iterations):
            c = torch.zeros_like(a)
            module.launch_grid_stride(a, b, c, n, blocks_stride, threads_per_block)
        end.record()
        torch.cuda.synchronize()
        stride_time = start.elapsed_time(end) / iterations
        results[f'Grid-Stride (SM×{mult})'] = (stride_time, blocks_stride)
    
    # 3. PyTorch 기본 연산 (비교용)
    start.record()
    for _ in range(iterations):
        _ = a + b
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / iterations
    results['PyTorch (a + b)'] = (pytorch_time, 'N/A')
    
    # 결과 출력
    print("\n" + "-" * 60)
    print(f"{'방식':<25} {'시간 (ms)':<12} {'Grid 크기':<12} {'상대 속도':<10}")
    print("-" * 60)
    
    baseline = results['Monolithic'][0]
    for name, (time_ms, grid_size) in results.items():
        speedup = baseline / time_ms if time_ms > 0 else float('inf')
        grid_str = str(grid_size) if grid_size != 'N/A' else 'N/A'
        print(f"{name:<25} {time_ms:<12.4f} {grid_str:<12} {speedup:<10.2f}x")
    
    print("-" * 60)


def demonstrate_scalability() -> None:
    """Grid-Stride Loop의 확장성 시연"""
    print("\n" + "=" * 60)
    print("확장성 시연: 다양한 데이터 크기에서의 동작")
    print("=" * 60)
    
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    threads_per_block = 256
    fixed_grid = sm_count * 4  # 고정된 Grid 크기
    
    print(f"\n고정된 설정:")
    print(f"  • threads_per_block = {threads_per_block}")
    print(f"  • blocks_per_grid = {fixed_grid} (SM × 4)")
    print(f"  • 총 스레드 수 = {threads_per_block * fixed_grid:,}")
    
    data_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    
    print("\n" + "-" * 60)
    print(f"{'데이터 크기':<15} {'스레드당 작업량':<18} {'검증':<10}")
    print("-" * 60)
    
    for n in data_sizes:
        total_threads = threads_per_block * fixed_grid
        work_per_thread = cdiv(n, total_threads)
        
        # 실제로 실행해서 검증
        a = torch.rand(n, dtype=torch.float32, device='cuda')
        b = torch.rand(n, dtype=torch.float32, device='cuda')
        
        result = vector_add_grid_stride(a, b, threads_per_block, fixed_grid)
        torch.cuda.synchronize()
        
        expected = a + b
        is_correct = torch.allclose(result, expected)
        
        status = "✅" if is_correct else "❌"
        print(f"{n:>13,} {work_per_thread:>15,}개 {status:>10}")
        
        # 메모리 정리
        del a, b, result, expected
        torch.cuda.empty_cache()
    
    print("-" * 60)
    print("\n💡 핵심: Grid 크기가 고정되어도 모든 데이터 크기에서 동작합니다!")


def explain_grid_stride_visually() -> None:
    """Grid-Stride Loop의 동작을 시각적으로 설명"""
    print("\n" + "=" * 60)
    print("Grid-Stride Loop 동작 원리 (시각화)")
    print("=" * 60)
    
    # 작은 예시로 설명
    n = 12  # 데이터 크기
    threads = 4  # 총 스레드 수 (1 block × 4 threads)
    stride = threads
    
    print(f"\n예시: 데이터 {n}개, 스레드 {threads}개")
    print(f"stride = {stride} (전체 스레드 수)\n")
    
    print("데이터 인덱스: ", end="")
    print(" ".join(f"[{i:2d}]" for i in range(n)))
    print()
    
    for thread_id in range(threads):
        indices = list(range(thread_id, n, stride))
        print(f"Thread {thread_id}: ", end="")
        
        visualization = []
        for i in range(n):
            if i in indices:
                visualization.append(f"[{i:2d}]")
            else:
                visualization.append("    ")
        print(" ".join(visualization))
        print(f"         → 처리: {indices}")
    
    print("\n" + "-" * 60)
    print("각 스레드가 stride만큼 건너뛰며 자기 몫의 데이터를 처리합니다.")
    print("-" * 60)


if __name__ == "__main__":
    print("\n" + "🚀 Week 2: Grid-Stride Loop 실습".center(60))
    print("=" * 60)
    
    # GPU 정보 출력
    device_name = torch.cuda.get_device_name(0)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n🖥️  GPU: {device_name}")
    print(f"   SM 개수: {sm_count}")
    
    # 1. 동작 원리 시각화
    explain_grid_stride_visually()
    
    # 2. 정확성 검증
    n = 1_000_000
    a = torch.rand(n, dtype=torch.float32, device='cuda')
    b = torch.rand(n, dtype=torch.float32, device='cuda')
    verify_correctness(a, b)
    
    # 3. 성능 벤치마크
    benchmark_comparison(n=10_000_000, iterations=100)
    
    # 4. 확장성 시연
    demonstrate_scalability()
    
    print("\n" + "=" * 60)
    print("🎉 Week 2 실습 완료!")
    print("=" * 60 + "\n")
