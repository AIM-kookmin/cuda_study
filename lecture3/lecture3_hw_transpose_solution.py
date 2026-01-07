"""
Week 3 과제 해답: Matrix Transpose Optimization (행렬 전치 최적화)

완전한 해답 코드 with 상세한 설명과 성능 분석

해결된 문제들:
1. ✅ Memory Coalescing 최적화 (Read Coalesced → Write Coalesced)
2. ✅ Shared Memory Bank Conflicts 완전 제거
3. ✅ 경계 조건 처리 (non-square matrices)
4. ✅ 성능 측정 및 분석 자동화
5. ✅ PyTorch baseline과의 정확성 검증

성능 목표: Naive 대비 3-5배 성능 향상
"""
import os
import math
import torch
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"

# 최적화된 타일 크기들
TILE_SIZE = 32  # 기본 타일 크기
TILE_SIZE_OPTIMIZED = 32  # Bank conflict 해결용


# =============================================================================
# CUDA 커널 소스코드 - 완전한 해답
# =============================================================================

cuda_source = f"""
#include <torch/extension.h>

#define TILE_SIZE {TILE_SIZE}
#define TILE_SIZE_OPT {TILE_SIZE_OPTIMIZED}

// ❌ Naive Matrix Transpose (참고용 - 비효율적)
__global__ void transpose_naive_kernel(
    const float* __restrict__ input,   // [M][N]
    float* __restrict__ output,        // [N][M]
    int M, int N
) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {{
        // Read: Coalesced ✅, Write: Strided ❌
        output[col * M + row] = input[row * N + col];
    }}
}}

// ✅ SOLUTION 1: Shared Memory Tiled Transpose
// 핵심 아이디어: Read와 Write 모두 Coalesced하게 만들기
__global__ void transpose_tiled_kernel(
    const float* __restrict__ input,   // [M][N]
    float* __restrict__ output,        // [N][M]  
    int M, int N
) {{
    // Shared Memory 타일 (정사각형)
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    // Global 좌표 계산
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Phase 1: Global → Shared (Coalesced Read ✅)
    if (y < M && x < N) {{
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }} else {{
        tile[threadIdx.y][threadIdx.x] = 0.0f;  // Padding
    }}
    
    __syncthreads();  // 모든 스레드가 로딩 완료까지 대기
    
    // Phase 2: Shared → Global (Coalesced Write ✅)  
    // 주의: 출력 좌표를 transpose해서 계산!
    int out_x = blockIdx.y * TILE_SIZE + threadIdx.x;  // ← blockIdx 바뀜!
    int out_y = blockIdx.x * TILE_SIZE + threadIdx.y;  // ← blockIdx 바뀜!
    
    if (out_y < N && out_x < M) {{
        // transpose된 인덱스로 읽기
        output[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];  // ← threadIdx 바뀜!
    }}
}}

// ✅ SOLUTION 2: Bank Conflict Free Transpose (최고 성능)
// 핵심 개선: +1 Padding으로 Bank Conflicts 완전 제거
__global__ void transpose_optimized_kernel(
    const float* __restrict__ input,   // [M][N]
    float* __restrict__ output,        // [N][M]
    int M, int N
) {{
    // Bank Conflict 방지: [TILE_SIZE][TILE_SIZE + 1] 패딩
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // ← +1 패딩!
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Phase 1: Coalesced Read
    if (y < M && x < N) {{
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }} else {{
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }}
    
    __syncthreads();
    
    // Phase 2: Coalesced Write + Bank Conflict Free
    int out_x = blockIdx.y * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (out_y < N && out_x < M) {{
        // 패딩 덕분에 Bank Conflicts 없음!
        output[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];
    }}
}}

// ✅ SOLUTION 3: Advanced Rectangular Transpose
// 비정방 행렬(M≠N)에 최적화된 버전
__global__ void transpose_rectangular_kernel(
    const float* __restrict__ input,   // [M][N]  
    float* __restrict__ output,        // [N][M]
    int M, int N
) {{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    // 더 복잡한 인덱스 계산 (비정방 행렬 대응)
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Multiple loads per thread for efficiency (if needed)
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {{
        int curr_y = y + i;
        if (curr_y < M && x < N) {{
            tile[threadIdx.y + i][threadIdx.x] = input[curr_y * N + x];
        }}
    }}
    
    __syncthreads();
    
    // Transpose write
    int out_x = blockIdx.y * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    #pragma unroll  
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {{
        int curr_out_y = out_y + i;
        if (curr_out_y < N && out_x < M) {{
            output[curr_out_y * M + out_x] = tile[threadIdx.x][threadIdx.y + i];
        }}
    }}
}}

// Python 인터페이스
torch::Tensor transpose_naive_cuda(torch::Tensor input) {{
    const int M = input.size(0);
    const int N = input.size(1);
    
    auto output = torch::zeros({{N, M}}, input.options());
    
    const dim3 blockSize(16, 16);  // 16x16 = 256 threads
    const dim3 gridSize(
        (N + blockSize.x - 1) / blockSize.x,
        (M + blockSize.y - 1) / blockSize.y
    );
    
    transpose_naive_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    
    return output;
}}

torch::Tensor transpose_tiled_cuda(torch::Tensor input) {{
    const int M = input.size(0);
    const int N = input.size(1);
    
    auto output = torch::zeros({{N, M}}, input.options());
    
    const dim3 blockSize(TILE_SIZE, TILE_SIZE);  // 32x32 = 1024 threads
    const dim3 gridSize(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    transpose_tiled_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    
    return output;
}}

torch::Tensor transpose_optimized_cuda(torch::Tensor input) {{
    const int M = input.size(0);
    const int N = input.size(1);
    
    auto output = torch::zeros({{N, M}}, input.options());
    
    const dim3 blockSize(TILE_SIZE, TILE_SIZE);
    const dim3 gridSize(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    transpose_optimized_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    
    return output;
}}

torch::Tensor transpose_rectangular_cuda(torch::Tensor input) {{
    const int M = input.size(0);
    const int N = input.size(1);
    
    auto output = torch::zeros({{N, M}}, input.options());
    
    // 더 작은 블록 크기 (비정방 행렬에 적합)
    const dim3 blockSize(TILE_SIZE, 8);  // 32x8 = 256 threads
    const dim3 gridSize(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + 8 - 1) / 8
    );
    
    transpose_rectangular_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    
    return output;
}}
"""

# C++ wrapper
cpp_source = """
torch::Tensor transpose_naive_cuda(torch::Tensor input);
torch::Tensor transpose_tiled_cuda(torch::Tensor input);
torch::Tensor transpose_optimized_cuda(torch::Tensor input);  
torch::Tensor transpose_rectangular_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_naive", &transpose_naive_cuda, "Naive Matrix Transpose");
    m.def("transpose_tiled", &transpose_tiled_cuda, "Tiled Matrix Transpose");
    m.def("transpose_optimized", &transpose_optimized_cuda, "Optimized Matrix Transpose");
    m.def("transpose_rectangular", &transpose_rectangular_cuda, "Rectangular Matrix Transpose");
}
"""

# JIT 컴파일
print("🔨 JIT 컴파일 중... (30초 정도 소요)")
module = load_inline(
    name='transpose_solution_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    verbose=False
)

print("✅ 컴파일 완료!")


# =============================================================================
# 해답 검증 및 성능 분석
# =============================================================================

def verify_correctness(input_tensor: torch.Tensor) -> bool:
    """모든 구현의 정확성을 PyTorch baseline과 검증"""
    
    # PyTorch 정답
    expected = input_tensor.t().contiguous()
    
    # 모든 CUDA 구현 테스트
    implementations = [
        ("Naive", module.transpose_naive),
        ("Tiled", module.transpose_tiled), 
        ("Optimized", module.transpose_optimized),
        ("Rectangular", module.transpose_rectangular),
    ]
    
    print("🧪 정확성 검증:")
    all_correct = True
    
    for name, func in implementations:
        try:
            result = func(input_tensor)
            is_correct = torch.allclose(result, expected, atol=1e-5)
            status = "✅" if is_correct else "❌"
            print(f"  {status} {name}: {is_correct}")
            
            if not is_correct:
                error = torch.max(torch.abs(result - expected)).item()
                print(f"     Max error: {error:.2e}")
                all_correct = False
                
        except Exception as e:
            print(f"  ❌ {name}: ERROR - {str(e)}")
            all_correct = False
    
    return all_correct


def benchmark_all_methods(M: int, N: int, iterations: int = 100) -> None:
    """모든 transpose 방법의 성능을 비교 분석"""
    
    print(f"\n📊 성능 벤치마크 [{M}×{N} 행렬, {iterations}회 평균]")
    print("=" * 70)
    
    # 테스트 데이터 생성
    device = torch.device('cuda')
    input_tensor = torch.randn(M, N, device=device, dtype=torch.float32)
    
    # 정확성 검증
    if not verify_correctness(input_tensor):
        print("❌ 정확성 검증 실패! 벤치마크를 중단합니다.")
        return
    
    print("\n⚡ 성능 측정:")
    
    # PyTorch 기준선
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        pytorch_result = input_tensor.t().contiguous()
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / iterations
    
    # 메모리 대역폭 계산
    bytes_transferred = M * N * 4 * 2  # float32 * (read + write)
    theoretical_bandwidth = bytes_transferred / (pytorch_time * 1e-3) / 1e9  # GB/s
    
    print(f"  PyTorch baseline:     {pytorch_time:.3f} ms")
    print(f"  Memory bandwidth:     {theoretical_bandwidth:.1f} GB/s")
    print()
    
    # CUDA 구현들
    implementations = [
        ("Naive CUDA", module.transpose_naive),
        ("Tiled CUDA", module.transpose_tiled),
        ("Optimized CUDA", module.transpose_optimized), 
        ("Rectangular CUDA", module.transpose_rectangular),
    ]
    
    results = []
    
    for name, func in implementations:
        # Warm-up
        for _ in range(10):
            _ = func(input_tensor)
        
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            result = func(input_tensor)
        end.record()
        torch.cuda.synchronize()
        
        cuda_time = start.elapsed_time(end) / iterations
        speedup = pytorch_time / cuda_time
        bandwidth = bytes_transferred / (cuda_time * 1e-3) / 1e9
        
        results.append((name, cuda_time, speedup, bandwidth))
        
        status = "🚀" if speedup > 1.0 else "🐌"
        print(f"  {status} {name:15s}: {cuda_time:6.3f} ms  "
              f"({speedup:4.1f}x)  {bandwidth:6.1f} GB/s")
    
    # 최고 성능 분석
    best = max(results, key=lambda x: x[2])  # speedup 기준
    print(f"\n🏆 최고 성능: {best[0]} - {best[2]:.1f}배 향상!")
    
    # 성능 분석
    print(f"\n🔍 성능 분석:")
    print(f"  이론적 최대 대역폭: ~1000 GB/s (RTX 4090)")
    print(f"  달성된 최대 대역폭:  {best[3]:.1f} GB/s ({best[3]/1000*100:.1f}%)")
    
    if best[3] / 1000 < 0.6:
        print(f"  📝 개선 여지: Bank conflicts, Launch overhead 추가 최적화 가능")
    else:
        print(f"  ✨ 우수한 성능: 하드웨어 한계에 근접한 성능!")


def analyze_memory_patterns() -> None:
    """메모리 접근 패턴 분석 및 교육적 설명"""
    
    print("\n🧠 메모리 접근 패턴 분석:")
    print("=" * 50)
    
    print("""
📖 Transpose의 핵심 문제:
   Original: A[row][col] → Transposed: A_T[col][row]
   
   ❌ Naive 접근:
   - Read:  A[row][col]     → Coalesced ✅ (연속 메모리)  
   - Write: A_T[col][row]   → Strided ❌ (띄엄띄엄)
   
   ✅ Tiled 해결책:
   1. Shared Memory에 타일 단위로 저장 (Coalesced Read)
   2. Transpose된 순서로 출력 (Coalesced Write)
   3. 두 단계 모두 Coalesced 달성!

🔧 Bank Conflicts 해결:
   - 문제: tile[threadIdx.x][threadIdx.y] 접근 시 동일 Bank 충돌
   - 해결: tile[TILE_SIZE][TILE_SIZE + 1] 패딩으로 Bank 분산
   - 효과: 추가 15-30% 성능 향상
   
🎯 최적화 포인트:
   1. Memory Coalescing (가장 중요)
   2. Shared Memory Bank Conflicts
   3. Launch Configuration  
   4. Register Usage & Occupancy
""")


def test_various_sizes() -> None:
    """다양한 행렬 크기에서 성능 테스트"""
    
    print("\n📏 다양한 크기에서의 성능 테스트:")
    print("=" * 60)
    
    test_sizes = [
        (512, 512),      # 정방행렬 - 소형
        (1024, 1024),    # 정방행렬 - 중형  
        (2048, 2048),    # 정방행렬 - 대형
        (1000, 2000),    # 비정방행렬 - 세로 긴
        (2000, 1000),    # 비정방행렬 - 가로 긴
    ]
    
    for M, N in test_sizes:
        print(f"\n▶ 행렬 크기: {M} × {N}")
        benchmark_all_methods(M, N, iterations=50)


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """해답 데모 실행"""
    
    print("=" * 70)
    print("🎓 Week 3 과제 해답: Matrix Transpose 최적화")
    print("=" * 70)
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("❌ CUDA가 사용 불가능합니다.")
        return
        
    print(f"📱 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 메모리 접근 패턴 설명
    analyze_memory_patterns()
    
    # 기본 성능 테스트
    benchmark_all_methods(1024, 1024, iterations=100)
    
    # 추가 크기 테스트 (선택적)
    user_input = input("\n🤔 다양한 크기에서 추가 테스트하시겠습니까? (y/N): ")
    if user_input.lower() == 'y':
        test_various_sizes()
    
    print("\n" + "=" * 70)
    print("🎉 Matrix Transpose 최적화 완료!")
    print("✨ 핵심 학습: Shared Memory + Coalescing + Bank Conflict 해결")
    print("📚 다음: Week 4 - Profiling & Performance Analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()