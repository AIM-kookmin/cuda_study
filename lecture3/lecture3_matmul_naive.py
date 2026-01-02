"""
Week 3: Naive Matrix Multiplication (기본 행렬 곱셈)

목표:
1. 기본적인 행렬 곱셈 커널 구현
2. Global Memory만 사용할 때의 성능 한계 체험
3. 다음 시간에 배울 Tiled 버전과의 비교를 위한 Baseline 제공

수식: C[i][j] = Σ(k=0 to K-1) A[i][k] * B[k][j]
"""
import os
import math
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

// Naive Matrix Multiplication (기본 행렬 곱셈)
// C = A × B, 모든 데이터를 Global Memory에서 직접 읽음
__global__ void matmul_naive_kernel(
    const float* __restrict__ A,  // A[M][K]
    const float* __restrict__ B,  // B[K][N]  
    float* __restrict__ C,        // C[M][N]
    int M, int N, int K
) {
    // 2D Grid 사용: (N, M) 크기
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C의 열 (j)
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C의 행 (i)
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Dot Product: A의 row번째 행과 B의 col번째 열의 내적
        for (int k = 0; k < K; k++) {
            // A[row][k] * B[k][col]
            // ⚠️ Global Memory에서 매번 읽음 (비효율!)
            float a_val = A[row * K + k];      // A[row][k]
            float b_val = B[k * N + col];      // B[k][col] - Strided Access!
            sum += a_val * b_val;
        }
        
        // 결과를 Global Memory에 저장
        C[row * N + col] = sum;
    }
}

// 1D Grid 버전 (Grid-Stride Loop)
__global__ void matmul_naive_1d_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 1D 인덱싱으로 2D 좌표 계산
    int total_threads = blockDim.x * gridDim.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-Stride Loop
    for (int idx = thread_id; idx < M * N; idx += total_threads) {
        int row = idx / N;
        int col = idx % N;
        
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// C++ Wrapper 함수들
void launch_matmul_naive_2d(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    int M, int N, int K,
    dim3 grid, dim3 block
) {
    matmul_naive_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}

void launch_matmul_naive_1d(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    int M, int N, int K,
    int blocks, int threads
) {
    matmul_naive_1d_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}
"""

cpp_header = """
void launch_matmul_naive_2d(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    int M, int N, int K, dim3 grid, dim3 block
);
void launch_matmul_naive_1d(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    int M, int N, int K, int blocks, int threads
);
"""

# JIT 컴파일
print("=" * 60)
print("Compiling Naive Matrix Multiplication Kernels...")
print("=" * 60)

module = load_inline(
    name='matmul_naive_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_matmul_naive_2d', 'launch_matmul_naive_1d'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# 유틸리티 함수들
# =============================================================================

def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division"""
    return (n + divisor - 1) // divisor


def calculate_gflops(M: int, N: int, K: int, time_ms: float) -> float:
    """GFLOPS 계산 (Giga FLoating-point Operations Per Second)"""
    # 행렬 곱셈 연산량: 2*M*N*K (곱셈 M*N*K번, 덧셈 M*N*K번)
    total_ops = 2 * M * N * K
    time_seconds = time_ms / 1000.0
    gflops = total_ops / time_seconds / 1e9
    return gflops


def get_theoretical_peak_gflops() -> float:
    """이론적 Peak Performance 계산"""
    props = torch.cuda.get_device_properties(0)
    
    # SM 개수 * Cores/SM * Base Clock
    # 실제 값은 GPU마다 다름 (대략적인 추정)
    sm_count = props.multi_processor_count
    
    # 추정치 (실제 하드웨어에 따라 다름)
    if "RTX" in props.name or "GeForce" in props.name:
        cores_per_sm = 128  # Ampere/Ada 아키텍처 기준
        base_clock_ghz = 1.5  # 추정
    else:
        cores_per_sm = 64  # 보수적 추정
        base_clock_ghz = 1.0
    
    peak_gflops = sm_count * cores_per_sm * base_clock_ghz * 2  # FMA 명령어
    return peak_gflops


# =============================================================================
# Matrix Multiplication 구현들
# =============================================================================

def matmul_naive_2d(A: torch.Tensor, B: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """
    2D Grid를 사용한 Naive Matrix Multiplication
    
    Args:
        A: [M, K] 행렬
        B: [K, N] 행렬  
        block_size: Thread Block 크기 (기본값: 16)
    
    Returns:
        C: [M, N] 행렬
    """
    assert A.dim() == 2 and B.dim() == 2, "2D 행렬만 지원"
    assert A.shape[1] == B.shape[0], f"행렬 차원 불일치: A={A.shape}, B={B.shape}"
    
    M, K = A.shape
    K2, N = B.shape
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    # 2D Grid 설정
    block = (block_size, block_size)
    grid = (cdiv(N, block_size), cdiv(M, block_size))
    
    # 커널 실행
    module.launch_matmul_naive_2d(A, B, C, M, N, K, grid, block)
    
    return C


def matmul_naive_1d(A: torch.Tensor, B: torch.Tensor, 
                   threads_per_block: int = 256) -> torch.Tensor:
    """
    1D Grid를 사용한 Naive Matrix Multiplication (Grid-Stride Loop)
    
    Args:
        A: [M, K] 행렬
        B: [K, N] 행렬
        threads_per_block: 블록당 스레드 수
    
    Returns:
        C: [M, N] 행렬
    """
    assert A.dim() == 2 and B.dim() == 2, "2D 행렬만 지원"
    assert A.shape[1] == B.shape[0], f"행렬 차원 불일치: A={A.shape}, B={B.shape}"
    
    M, K = A.shape
    K2, N = B.shape
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    # 1D Grid 설정 (Grid-Stride Loop)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    blocks = sm_count * 4  # SM당 4개 블록
    
    # 커널 실행
    module.launch_matmul_naive_1d(A, B, C, M, N, K, blocks, threads_per_block)
    
    return C


def matmul_pytorch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch 기본 행렬 곱셈 (cuBLAS 사용)"""
    return torch.mm(A, B)


# =============================================================================
# 벤치마크 함수들
# =============================================================================

def verify_correctness(M: int, N: int, K: int) -> None:
    """정확성 검증"""
    print("=" * 60)
    print(f"정확성 검증 (M={M}, N={N}, K={K})")
    print("=" * 60)
    
    # 작은 크기 행렬로 테스트
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    
    # 각 구현의 결과
    C_pytorch = matmul_pytorch(A, B)
    C_naive_2d = matmul_naive_2d(A, B)
    C_naive_1d = matmul_naive_1d(A, B)
    
    # 비교
    error_2d = (C_pytorch - C_naive_2d).abs().max().item()
    error_1d = (C_pytorch - C_naive_1d).abs().max().item()
    
    print(f"최대 오차:")
    print(f"  Naive 2D vs PyTorch: {error_2d:.2e}")
    print(f"  Naive 1D vs PyTorch: {error_1d:.2e}")
    
    if error_2d < 1e-4 and error_1d < 1e-4:
        print("✅ 모든 구현이 정확합니다!")
    else:
        print("❌ 오차가 큽니다. 구현을 확인해주세요.")


def benchmark_different_sizes() -> None:
    """다양한 행렬 크기에서 성능 측정"""
    print("\n" + "=" * 70)
    print("다양한 행렬 크기 성능 비교")
    print("=" * 70)
    
    sizes = [128, 256, 512, 1024]
    iterations = 50
    
    print(f"{'Size':<8} {'Naive 2D (ms)':<15} {'Naive 1D (ms)':<15} {'PyTorch (ms)':<15} {'GFLOPS':<10}")
    print("-" * 70)
    
    for size in sizes:
        M = N = K = size
        
        # 데이터 생성
        A = torch.randn(M, K, dtype=torch.float32, device='cuda')
        B = torch.randn(K, N, dtype=torch.float32, device='cuda')
        
        # Warm-up
        for _ in range(5):
            _ = matmul_naive_2d(A, B)
            _ = matmul_naive_1d(A, B)
            _ = matmul_pytorch(A, B)
        torch.cuda.synchronize()
        
        # 시간 측정
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Naive 2D 측정
        start.record()
        for _ in range(iterations):
            C = matmul_naive_2d(A, B)
        end.record()
        torch.cuda.synchronize()
        time_2d = start.elapsed_time(end) / iterations
        
        # Naive 1D 측정
        start.record()
        for _ in range(iterations):
            C = matmul_naive_1d(A, B)
        end.record()
        torch.cuda.synchronize()
        time_1d = start.elapsed_time(end) / iterations
        
        # PyTorch 측정
        start.record()
        for _ in range(iterations):
            C = matmul_pytorch(A, B)
        end.record()
        torch.cuda.synchronize()
        time_pytorch = start.elapsed_time(end) / iterations
        
        # GFLOPS 계산 (Naive 2D 기준)
        gflops = calculate_gflops(M, N, K, time_2d)
        
        print(f"{size:<8} {time_2d:<15.4f} {time_1d:<15.4f} {time_pytorch:<15.4f} {gflops:<10.1f}")


def analyze_memory_pattern() -> None:
    """메모리 접근 패턴 분석"""
    print("\n" + "=" * 70)
    print("메모리 접근 패턴 분석")
    print("=" * 70)
    print("""
Naive Matrix Multiplication의 문제점:

1. 🐌 반복적인 Global Memory 접근:
   - 각 C[i][j] 계산에 A의 i번째 행과 B의 j번째 열을 모두 읽음
   - A[i][k]는 같은 행의 다른 원소들과 함께 여러 번 읽힘
   - B[k][j]는 같은 열의 다른 원소들과 함께 여러 번 읽힘

2. 💥 B 행렬의 Strided Access:
   - B[k][j] 접근 시 stride = N인 접근 패턴
   - N이 클수록 Cache Miss 증가, Coalescing 깨짐

3. 📊 Memory Bandwidth vs Compute 비율:
   - 각 원소당 2번의 메모리 읽기, 1번의 FMA 연산
   - Arithmetic Intensity가 낮음 (메모리 바운드)

해결책 (다음 시간에 배울 내용):
- 🚀 Tiling: 데이터를 Shared Memory에 캐시하여 재사용
- ⚡ Coalescing: 메모리 접근 패턴 최적화  
- 🧠 Data Reuse: 같은 데이터를 여러 번 사용
    """)


if __name__ == "__main__":
    print("\n" + "🚀 Week 3: Naive Matrix Multiplication".center(70))
    print("=" * 70)
    
    # GPU 정보
    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    
    print(f"\n🖥️  GPU: {device_name}")
    print(f"   SM 개수: {sm_count}")
    print(f"   이론적 Peak Performance: ~{get_theoretical_peak_gflops():.0f} GFLOPS")
    
    # 정확성 검증
    verify_correctness(M=32, N=32, K=32)
    
    # 성능 벤치마크
    benchmark_different_sizes()
    
    # 메모리 패턴 분석
    analyze_memory_pattern()
    
    print("\n" + "=" * 70)
    print("📊 관찰 포인트")
    print("=" * 70)
    print("""
1. 행렬 크기가 커질수록 성능이 급격히 떨어짐 (O(N³) 복잡도)
2. PyTorch(cuBLAS)가 압도적으로 빠름 (고도 최적화)
3. Naive 구현은 GPU의 잠재력을 제대로 활용하지 못함
4. Memory Bandwidth가 주요 병목 지점

다음 시간 예고:
🔥 Shared Memory + Tiling으로 5~10배 성능 향상 달성!
    """)
    
    print("\n" + "=" * 70)
    print("🎉 Week 3-2: Naive Matrix Multiplication 완료!")
    print("=" * 70 + "\n")
