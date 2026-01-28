"""
Week 3 복습: Matrix Transpose 최적화 여정 🔄

학습 목표:
1. Naive Transpose의 메모리 접근 패턴 이해
2. Shared Memory Tiling을 통한 Coalescing 개선
3. Bank Conflict 회피 (+1 Padding 기법)
4. Memory Bandwidth 계산 및 분석

최적화 단계:
- Level 0: PyTorch 기본 구현 (참조용)
- Level 1: Naive CUDA (비효율적 메모리 접근)
- Level 2: Shared Memory Tiling (Coalescing 개선)
- Level 3: Bank Conflict 회피 (최종 최적화)
"""
import os
import math
import torch
from torch.utils.cpp_extension import load_inline

# Compiler 설정 (PyTorch 호환성)
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
# CUDA 12.0은 compute capability 9.0까지만 지원
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;9.0"

# 상수 정의
TILE_SIZE = 32  # Transpose에서 일반적으로 사용하는 타일 크기

# ============================================================
# CUDA Kernel Source Code
# ============================================================

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Level 1: Naive Transpose
// 문제점: Write 시 strided access로 인한 비효율적 메모리 접근
__global__ void transpose_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        // Read: Coalesced (연속적)
        // Write: Strided (비연속적) - 성능 저하!
        output[col * M + row] = input[row * N + col];
    }
}

// Level 2: Shared Memory Tiling
// 개선점: Shared Memory를 중간 버퍼로 사용하여 Global Memory 접근 최적화
__global__ void transpose_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    // Shared Memory: 타일 단위 버퍼
    __shared__ float tile[32][32];
    
    // 전역 인덱스 계산
    int global_row = blockIdx.y * 32 + threadIdx.y;
    int global_col = blockIdx.x * 32 + threadIdx.x;
    
    // 1단계: Global Memory → Shared Memory (Coalesced Read)
    if (global_row < M && global_col < N) {
        tile[threadIdx.y][threadIdx.x] = input[global_row * N + global_col];
    }
    __syncthreads();
    
    // 2단계: Shared Memory → Global Memory (Transposed, Coalesced Write)
    // 주의: Block 좌표도 Transpose 해야 함
    global_row = blockIdx.x * 32 + threadIdx.y;
    global_col = blockIdx.y * 32 + threadIdx.x;
    
    if (global_row < N && global_col < M) {
        output[global_row * M + global_col] = tile[threadIdx.x][threadIdx.y];
    }
}

// Level 3: Bank Conflict 회피
// 개선점: +1 Padding으로 Bank Conflict 제거
__global__ void transpose_tiled_padded_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    // Shared Memory: +1 Padding으로 Bank Conflict 회피
    // [32][32] → [32][33]
    __shared__ float tile[32][33];
    
    int global_row = blockIdx.y * 32 + threadIdx.y;
    int global_col = blockIdx.x * 32 + threadIdx.x;
    
    // Global → Shared (Coalesced Read)
    if (global_row < M && global_col < N) {
        tile[threadIdx.y][threadIdx.x] = input[global_row * N + global_col];
    }
    __syncthreads();
    
    // Shared → Global (Transposed, Coalesced Write, No Bank Conflicts)
    global_row = blockIdx.x * 32 + threadIdx.y;
    global_col = blockIdx.y * 32 + threadIdx.x;
    
    if (global_row < N && global_col < M) {
        output[global_row * M + global_col] = tile[threadIdx.x][threadIdx.y];
    }
}

// Launcher Functions
torch::Tensor transpose_naive_cuda(torch::Tensor input) {
    const int M = input.size(0);
    const int N = input.size(1);
    
    auto output = torch::empty({N, M}, input.options());
    
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    
    transpose_naive_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    
    return output;
}

torch::Tensor transpose_tiled_cuda(torch::Tensor input) {
    const int M = input.size(0);
    const int N = input.size(1);
    
    auto output = torch::empty({N, M}, input.options());
    
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    
    transpose_tiled_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    
    return output;
}

torch::Tensor transpose_tiled_padded_cuda(torch::Tensor input) {
    const int M = input.size(0);
    const int N = input.size(1);
    
    auto output = torch::empty({N, M}, input.options());
    
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    
    transpose_tiled_padded_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor transpose_naive_cuda(torch::Tensor input);
torch::Tensor transpose_tiled_cuda(torch::Tensor input);
torch::Tensor transpose_tiled_padded_cuda(torch::Tensor input);
"""

# JIT 컴파일
transpose_module = load_inline(
    name='transpose_module',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['transpose_naive_cuda', 'transpose_tiled_cuda', 'transpose_tiled_padded_cuda'],
    verbose=False
)

# ============================================================
# Python Wrapper Functions
# ============================================================

def transpose_naive(input_tensor: torch.Tensor) -> torch.Tensor:
    """Level 1: Naive Transpose (비효율적)"""
    assert input_tensor.dim() == 2, f"Expected 2D tensor, got shape {input_tensor.shape}"
    assert input_tensor.is_cuda, "Input must be on CUDA device"
    assert input_tensor.dtype == torch.float32, "Only float32 supported"
    input_tensor = input_tensor.contiguous()
    return transpose_module.transpose_naive_cuda(input_tensor)


def transpose_tiled(input_tensor: torch.Tensor) -> torch.Tensor:
    """Level 2: Shared Memory Tiling (Coalescing 개선)"""
    assert input_tensor.dim() == 2, f"Expected 2D tensor, got shape {input_tensor.shape}"
    assert input_tensor.is_cuda, "Input must be on CUDA device"
    assert input_tensor.dtype == torch.float32, "Only float32 supported"
    input_tensor = input_tensor.contiguous()
    return transpose_module.transpose_tiled_cuda(input_tensor)


def transpose_tiled_padded(input_tensor: torch.Tensor) -> torch.Tensor:
    """Level 3: Bank Conflict 회피 (+1 Padding)"""
    assert input_tensor.dim() == 2, f"Expected 2D tensor, got shape {input_tensor.shape}"
    assert input_tensor.is_cuda, "Input must be on CUDA device"
    assert input_tensor.dtype == torch.float32, "Only float32 supported"
    input_tensor = input_tensor.contiguous()
    return transpose_module.transpose_tiled_padded_cuda(input_tensor)


# ============================================================
# Verification Functions
# ============================================================

def verify_transpose(M: int, N: int) -> None:
    """각 레벨의 Transpose 정확성 검증"""
    print("=" * 70)
    print(f"🔍 정확성 검증 - Matrix Size: {M} x {N}")
    print("=" * 70)
    
    # 테스트 데이터 생성
    input_tensor = torch.randn(M, N, device='cuda', dtype=torch.float32)
    reference = input_tensor.t().contiguous()  # PyTorch 참조
    
    # Level 1: Naive
    result_naive = transpose_naive(input_tensor)
    is_correct_naive = torch.allclose(result_naive, reference, rtol=1e-5, atol=1e-5)
    max_diff_naive = (result_naive - reference).abs().max().item()
    
    # Level 2: Tiled
    result_tiled = transpose_tiled(input_tensor)
    is_correct_tiled = torch.allclose(result_tiled, reference, rtol=1e-5, atol=1e-5)
    max_diff_tiled = (result_tiled - reference).abs().max().item()
    
    # Level 3: Tiled + Padded
    result_padded = transpose_tiled_padded(input_tensor)
    is_correct_padded = torch.allclose(result_padded, reference, rtol=1e-5, atol=1e-5)
    max_diff_padded = (result_padded - reference).abs().max().item()
    
    # 결과 출력
    print(f"✅ Level 1 (Naive):  {'PASS' if is_correct_naive else 'FAIL'} (max error: {max_diff_naive:.2e})")
    print(f"✅ Level 2 (Tiled):  {'PASS' if is_correct_tiled else 'FAIL'} (max error: {max_diff_tiled:.2e})")
    print(f"✅ Level 3 (Padded): {'PASS' if is_correct_padded else 'FAIL'} (max error: {max_diff_padded:.2e})")
    print()


# ============================================================
# Benchmark Functions
# ============================================================

def calculate_bandwidth_gbps(M: int, N: int, time_ms: float) -> float:
    """Memory Bandwidth 계산 (GB/s)"""
    total_bytes = 2 * M * N * 4  # Read + Write, float32 (4 bytes)
    time_seconds = time_ms / 1000.0
    return (total_bytes / time_seconds) / 1e9


def benchmark_transpose_methods(M: int = 4096, N: int = 4096, iterations: int = 100) -> None:
    """모든 Transpose 방법 벤치마크"""
    print("=" * 70)
    print(f"⚡ 성능 벤치마크 - Matrix Size: {M} x {N}")
    print(f"반복 횟수: {iterations}회")
    print("=" * 70)
    
    input_tensor = torch.randn(M, N, device='cuda', dtype=torch.float32)
    
    # Warm-up
    for _ in range(10):
        _ = input_tensor.t().contiguous()
        _ = transpose_naive(input_tensor)
        _ = transpose_tiled(input_tensor)
        _ = transpose_tiled_padded(input_tensor)
    torch.cuda.synchronize()
    
    methods = [
        ("PyTorch (참조)", lambda: input_tensor.t().contiguous()),
        ("Level 1: Naive", lambda: transpose_naive(input_tensor)),
        ("Level 2: Tiled", lambda: transpose_tiled(input_tensor)),
        ("Level 3: Padded", lambda: transpose_tiled_padded(input_tensor))
    ]
    
    results = []
    
    for name, func in methods:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            _ = func()
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / iterations
        bandwidth = calculate_bandwidth_gbps(M, N, time_ms)
        results.append((name, time_ms, bandwidth))
    
    # 결과 출력
    print(f"{'Method':<25} {'Time (ms)':<12} {'Bandwidth (GB/s)':<18} {'Speedup'}")
    print("-" * 70)
    
    baseline_time = results[0][1]
    for name, time_ms, bandwidth in results:
        speedup = baseline_time / time_ms
        print(f"{name:<25} {time_ms:>10.4f}   {bandwidth:>15.2f}      {speedup:>6.2f}x")
    
    print()
    
    # 분석
    print("📊 분석:")
    print(f"  - Naive → Tiled 개선율: {results[1][1] / results[2][1]:.2f}x")
    print(f"  - Tiled → Padded 개선율: {results[2][1] / results[3][1]:.2f}x")
    print(f"  - 전체 개선율 (Naive → Padded): {results[1][1] / results[3][1]:.2f}x")
    print()


def benchmark_different_sizes() -> None:
    """다양한 행렬 크기에서 성능 비교"""
    print("=" * 70)
    print("📏 다양한 크기에서의 성능 비교")
    print("=" * 70)
    
    sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192)
    ]
    
    print(f"{'Size':<15} {'Naive (ms)':<12} {'Tiled (ms)':<12} {'Padded (ms)':<13} {'Best BW (GB/s)'}")
    print("-" * 70)
    
    for M, N in sizes:
        input_tensor = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Warm-up
        for _ in range(5):
            _ = transpose_naive(input_tensor)
            _ = transpose_tiled(input_tensor)
            _ = transpose_tiled_padded(input_tensor)
        torch.cuda.synchronize()
        
        # Timing
        iterations = 50
        times = []
        
        for func in [transpose_naive, transpose_tiled, transpose_tiled_padded]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(iterations):
                _ = func(input_tensor)
            end.record()
            torch.cuda.synchronize()
            
            time_ms = start.elapsed_time(end) / iterations
            times.append(time_ms)
        
        best_bandwidth = calculate_bandwidth_gbps(M, N, times[2])
        print(f"{M}x{N:<9} {times[0]:>10.4f}   {times[1]:>10.4f}   {times[2]:>11.4f}   {best_bandwidth:>13.2f}")
    
    print()


# ============================================================
# NCU Profiling 가이드
# ============================================================

def print_profiling_guide() -> None:
    """NCU Profiling 사용법 안내"""
    print("=" * 70)
    print("🔬 NCU Profiling 가이드")
    print("=" * 70)
    print()
    print("📌 기본 프로파일링:")
    print("  ncu --set full python lecture5_review_week3_transpose.py")
    print()
    print("📌 Memory 관련 메트릭만 수집:")
    print("  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\\")
    print("               l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\\")
    print("               smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\\")
    print("               smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \\")
    print("      python lecture5_review_week3_transpose.py")
    print()
    print("📌 Bank Conflict 확인:")
    print("  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\\")
    print("               l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \\")
    print("      python lecture5_review_week3_transpose.py")
    print()
    print("💡 주목할 메트릭:")
    print("  - Global Load/Store Efficiency: Coalescing 효율")
    print("  - Shared Memory Bank Conflicts: Bank Conflict 횟수")
    print("  - Achieved Occupancy: 점유율")
    print()


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔄 Week 3 복습: Matrix Transpose 최적화 여정")
    print("=" * 70)
    print()
    
    # GPU 정보
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 정확성 검증
    verify_transpose(1024, 2048)
    
    # 성능 벤치마크
    benchmark_transpose_methods(4096, 4096, iterations=100)
    
    # 다양한 크기 비교
    benchmark_different_sizes()
    
    # Profiling 가이드
    print_profiling_guide()
    
    print("=" * 70)
    print("✅ Week 3 Transpose 복습 완료!")
    print("=" * 70)
    print()
    print("🎓 학습 포인트:")
    print("  1. Naive: Write 시 Strided Access로 인한 성능 저하")
    print("  2. Tiled: Shared Memory로 Coalescing 개선")
    print("  3. Padded: +1 Padding으로 Bank Conflict 제거")
    print("  4. 메모리 접근 패턴이 성능에 미치는 영향 체감")
    print()
