"""
Week 5 Day 3-4: Week 3 복습
Matrix Multiplication 단계별 최적화

목표:
1. Naive 구현으로 Baseline 설정
2. Shared Memory Tiling으로 대폭 성능 향상
3. Bank Conflict 최적화
4. GFLOPS 계산 및 cuBLAS 비교

학습 포인트:
- Tiling이 성능에 미치는 영향 정량화
- __syncthreads()의 올바른 사용
- Memory Bandwidth vs Compute 이해
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
# CUDA 12.0은 compute capability 9.0까지만 지원
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;9.0"

TILE_SIZE = 16


def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division"""
    return (n + divisor - 1) // divisor


def calculate_gflops(M: int, N: int, K: int, time_ms: float) -> float:
    """GFLOPS 계산"""
    total_ops = 2 * M * N * K  # FMA 연산
    time_seconds = time_ms / 1000.0
    return total_ops / time_seconds / 1e9


# =============================================================================
# CUDA Kernel 소스코드
# =============================================================================

cuda_source = f"""
#include <torch/extension.h>

#define TILE_SIZE {TILE_SIZE}

// Level 1: Naive Matrix Multiplication
__global__ void matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {{
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {{
            sum += A[row * K + k] * B[k * N + col];
        }}
        C[row * N + col] = sum;
    }}
}}

// Level 2: Tiled Matrix Multiplication (Basic)
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        // Load A tile
        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K) {{
            As[ty][tx] = A[row * K + a_col];
        }} else {{
            As[ty][tx] = 0.0f;
        }}
        
        // Load B tile
        int b_row = tile * TILE_SIZE + ty;
        if (b_row < K && col < N) {{
            Bs[ty][tx] = B[b_row * N + col];
        }} else {{
            Bs[ty][tx] = 0.0f;
        }}
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum += As[ty][k] * Bs[k][tx];
        }}
        
        __syncthreads();
    }}
    
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}

// Level 3: Tiled with Bank Conflict Optimization
__global__ void matmul_tiled_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {{
    // Bank Conflict 회피를 위한 +1 padding
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        // Load A tile
        int a_col = tile * TILE_SIZE + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        
        // Load B tile
        int b_row = tile * TILE_SIZE + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute with unrolling hint
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum += As[ty][k] * Bs[k][tx];
        }}
        
        __syncthreads();
    }}
    
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}

// Launcher functions
void launch_matmul_naive(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    int M, int N, int K
) {{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    matmul_naive_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}}

void launch_matmul_tiled(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    int M, int N, int K
) {{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}}

void launch_matmul_tiled_optimized(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    int M, int N, int K
) {{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_tiled_optimized_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}}
"""

cpp_header = """
void launch_matmul_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K);
void launch_matmul_tiled(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K);
void launch_matmul_tiled_optimized(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K);
"""

print("=" * 60)
print(f"Compiling Matrix Multiply Kernels (TILE_SIZE={TILE_SIZE})...")
print("=" * 60)

module = load_inline(
    name='matmul_review_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_matmul_naive', 'launch_matmul_tiled', 'launch_matmul_tiled_optimized'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# Python Wrapper 함수들
# =============================================================================

def matmul_naive(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Naive Matrix Multiplication"""
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape[1] == B.shape[0]
    
    M, K = A.shape
    K2, N = B.shape
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    module.launch_matmul_naive(A, B, C, M, N, K)
    return C


def matmul_tiled(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Tiled Matrix Multiplication"""
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape[1] == B.shape[0]
    
    M, K = A.shape
    K2, N = B.shape
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    module.launch_matmul_tiled(A, B, C, M, N, K)
    return C


def matmul_tiled_optimized(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Tiled + Bank Conflict Optimized"""
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape[1] == B.shape[0]
    
    M, K = A.shape
    K2, N = B.shape
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    module.launch_matmul_tiled_optimized(A, B, C, M, N, K)
    return C


# =============================================================================
# 검증 및 벤치마크
# =============================================================================

def verify_correctness():
    print("=" * 70)
    print("정확성 검증")
    print("=" * 70)
    
    M, N, K = 128, 128, 128
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    
    C_pytorch = torch.mm(A, B)
    C_naive = matmul_naive(A, B)
    C_tiled = matmul_tiled(A, B)
    C_tiled_opt = matmul_tiled_optimized(A, B)
    
    error_naive = (C_pytorch - C_naive).abs().max().item()
    error_tiled = (C_pytorch - C_tiled).abs().max().item()
    error_tiled_opt = (C_pytorch - C_tiled_opt).abs().max().item()
    
    print(f"행렬 크기: {M}x{K} × {K}x{N}")
    print(f"최대 오차:")
    print(f"  Naive: {error_naive:.2e} {'✅' if error_naive < 1e-3 else '❌'}")
    print(f"  Tiled: {error_tiled:.2e} {'✅' if error_tiled < 1e-3 else '❌'}")
    print(f"  Tiled Opt: {error_tiled_opt:.2e} {'✅' if error_tiled_opt < 1e-3 else '❌'}")
    print()


def benchmark_progressive():
    print("=" * 70)
    print("단계별 성능 비교")
    print("=" * 70)
    
    sizes = [256, 512, 1024, 2048]
    iterations = 20
    
    print(f"{'Size':<8} {'PyTorch':<12} {'Naive':<12} {'Tiled':<12} {'Tiled+Opt':<12} {'Best GFLOPS':<12}")
    print("-" * 80)
    
    for size in sizes:
        M = N = K = size
        
        A = torch.randn(M, K, dtype=torch.float32, device='cuda')
        B = torch.randn(K, N, dtype=torch.float32, device='cuda')
        
        # Warm-up
        for _ in range(3):
            _ = torch.mm(A, B)
            if size <= 1024:  # Naive는 느리므로 큰 크기는 스킵
                _ = matmul_naive(A, B)
            _ = matmul_tiled(A, B)
            _ = matmul_tiled_optimized(A, B)
        torch.cuda.synchronize()
        
        # 측정
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # PyTorch (cuBLAS)
        start.record()
        for _ in range(iterations):
            C = torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        time_pytorch = start.elapsed_time(end) / iterations
        
        # Naive (작은 크기만)
        if size <= 1024:
            start.record()
            for _ in range(iterations):
                C = matmul_naive(A, B)
            end.record()
            torch.cuda.synchronize()
            time_naive = start.elapsed_time(end) / iterations
        else:
            time_naive = -1
        
        # Tiled
        start.record()
        for _ in range(iterations):
            C = matmul_tiled(A, B)
        end.record()
        torch.cuda.synchronize()
        time_tiled = start.elapsed_time(end) / iterations
        
        # Tiled Optimized
        start.record()
        for _ in range(iterations):
            C = matmul_tiled_optimized(A, B)
        end.record()
        torch.cuda.synchronize()
        time_tiled_opt = start.elapsed_time(end) / iterations
        
        # GFLOPS 계산
        gflops_best = calculate_gflops(M, N, K, time_tiled_opt)
        
        naive_str = f"{time_naive:.2f}" if time_naive > 0 else "N/A"
        
        print(f"{size:<8} {time_pytorch:<12.4f} {naive_str:<12} {time_tiled:<12.4f} "
              f"{time_tiled_opt:<12.4f} {gflops_best:<12.1f}")


def analyze_optimization_impact():
    print("\n" + "=" * 70)
    print("최적화 효과 분석")
    print("=" * 70)
    
    size = 1024
    M = N = K = size
    
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    
    iterations = 50
    
    # Naive
    for _ in range(10):
        _ = matmul_naive(A, B)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        _ = matmul_naive(A, B)
    end.record()
    torch.cuda.synchronize()
    time_naive = start.elapsed_time(end) / iterations
    
    # Tiled
    for _ in range(10):
        _ = matmul_tiled(A, B)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(iterations):
        _ = matmul_tiled(A, B)
    end.record()
    torch.cuda.synchronize()
    time_tiled = start.elapsed_time(end) / iterations
    
    # Tiled Optimized
    for _ in range(10):
        _ = matmul_tiled_optimized(A, B)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(iterations):
        _ = matmul_tiled_optimized(A, B)
    end.record()
    torch.cuda.synchronize()
    time_tiled_opt = start.elapsed_time(end) / iterations
    
    # cuBLAS
    for _ in range(10):
        _ = torch.mm(A, B)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(iterations):
        _ = torch.mm(A, B)
    end.record()
    torch.cuda.synchronize()
    time_cublas = start.elapsed_time(end) / iterations
    
    print(f"행렬 크기: {M}x{N}x{K}")
    print(f"\n실행 시간:")
    print(f"  Naive:          {time_naive:.4f} ms (Baseline)")
    print(f"  Tiled:          {time_tiled:.4f} ms ({time_naive/time_tiled:.1f}x faster)")
    print(f"  Tiled+Opt:      {time_tiled_opt:.4f} ms ({time_naive/time_tiled_opt:.1f}x faster)")
    print(f"  cuBLAS:         {time_cublas:.4f} ms ({time_naive/time_cublas:.1f}x faster)")
    
    print(f"\nGFLOPS:")
    print(f"  Naive:          {calculate_gflops(M, N, K, time_naive):.1f}")
    print(f"  Tiled:          {calculate_gflops(M, N, K, time_tiled):.1f}")
    print(f"  Tiled+Opt:      {calculate_gflops(M, N, K, time_tiled_opt):.1f}")
    print(f"  cuBLAS:         {calculate_gflops(M, N, K, time_cublas):.1f}")
    
    print(f"\ncuBLAS 대비 효율:")
    print(f"  Tiled+Opt:      {(time_cublas/time_tiled_opt)*100:.1f}%")


if __name__ == "__main__":
    print("\n" + "🚀 Week 5 Day 3-4: Matrix Multiply Mastery".center(70))
    print("=" * 70)
    
    device_name = torch.cuda.get_device_name(0)
    print(f"\n🖥️  GPU: {device_name}")
    print(f"   Tile Size: {TILE_SIZE}x{TILE_SIZE}")
    
    # 정확성 검증
    verify_correctness()
    
    # 단계별 성능 비교
    benchmark_progressive()
    
    # 최적화 효과 분석
    analyze_optimization_impact()
    
    print("\n" + "=" * 70)
    print("🎉 Matrix Multiply 복습 완료!")
    print("=" * 70)
    print("\n핵심 학습:")
    print("  ✅ Tiling으로 5-10배 성능 향상")
    print("  ✅ Bank Conflict 최적화의 미세한 개선")
    print("  ✅ cuBLAS는 여전히 더 빠름 (고도 최적화)")
