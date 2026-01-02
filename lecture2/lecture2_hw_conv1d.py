"""
Week 2 과제 2: 1D Convolution with Grid-Stride Loop

목표: Vector Add보다 복잡한 커널을 Grid-Stride Loop로 구현
설명: 입력 벡터 x에 대해 3-tap 필터 [0.2, 0.6, 0.2]를 적용하는 1D Convolution

수식: y[i] = 0.2*x[i-1] + 0.6*x[i] + 0.2*x[i+1]

주의사항:
- i=0일 때 x[-1]은 0으로 처리 (경계 조건)
- i=N-1일 때 x[N]은 0으로 처리 (경계 조건)
- Grid-Stride Loop 패턴 반드시 적용
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

// 1D Convolution with Grid-Stride Loop
// Filter: [0.2, 0.6, 0.2]
__global__ void conv1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n
) {
    // Grid-Stride Loop 패턴
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = index; i < n; i += stride) {
        float left = (i > 0) ? x[i - 1] : 0.0f;      // 경계 체크: i=0이면 0
        float center = x[i];
        float right = (i < n - 1) ? x[i + 1] : 0.0f; // 경계 체크: i=N-1이면 0
        
        // 3-tap 필터 적용: [0.2, 0.6, 0.2]
        y[i] = 0.2f * left + 0.6f * center + 0.2f * right;
    }
}

void conv1d_launch(
    torch::Tensor x,
    torch::Tensor y,
    int n,
    int blocks,
    int threads
) {
    conv1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
}
"""

cpp_header = """
void conv1d_launch(
    torch::Tensor x,
    torch::Tensor y,
    int n,
    int blocks,
    int threads
);
"""

# JIT 컴파일
print("=" * 60)
print("Compiling 1D Convolution CUDA kernel...")
print("=" * 60)

module = load_inline(
    name='conv1d_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['conv1d_launch'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# Python Wrapper 함수
# =============================================================================

def conv1d_cuda(
    x: torch.Tensor,
    threads_per_block: int = 256,
    blocks_per_grid: int = None
) -> torch.Tensor:
    """
    1D Convolution with Grid-Stride Loop
    Filter: [0.2, 0.6, 0.2]
    
    Args:
        x: 입력 벡터 (1D Tensor, CUDA)
        threads_per_block: 블록당 스레드 수
        blocks_per_grid: Grid 크기 (None이면 SM×4 사용)
    
    Returns:
        y: 출력 벡터 (1D Tensor, CUDA)
    """
    assert x.dim() == 1, f"입력은 1D 텐서여야 합니다. 현재: {x.shape}"
    assert x.is_cuda, "입력 텐서는 CUDA 텐서여야 합니다."
    assert x.dtype == torch.float32, "입력 텐서는 float32여야 합니다."
    
    n = x.numel()
    y = torch.zeros_like(x)
    
    # Grid 크기 설정 (Grid-Stride Loop이므로 고정값 사용)
    if blocks_per_grid is None:
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        blocks_per_grid = sm_count * 4
    
    module.conv1d_launch(x, y, n, blocks_per_grid, threads_per_block)
    
    return y


def conv1d_pytorch(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 구현 (검증용)"""
    n = x.numel()
    y = torch.zeros_like(x)
    
    for i in range(n):
        left = x[i - 1] if i > 0 else 0.0
        center = x[i]
        right = x[i + 1] if i < n - 1 else 0.0
        y[i] = 0.2 * left + 0.6 * center + 0.2 * right
    
    return y


def verify_correctness(x: torch.Tensor) -> None:
    """정확성 검증"""
    print("=" * 60)
    print("정확성 검증")
    print("=" * 60)
    
    # CUDA 결과
    y_cuda = conv1d_cuda(x)
    torch.cuda.synchronize()
    
    # PyTorch 결과
    y_pytorch = conv1d_pytorch(x)
    
    # 비교
    is_correct = torch.allclose(y_cuda, y_pytorch, rtol=1e-5, atol=1e-5)
    max_diff = (y_cuda - y_pytorch).abs().max().item()
    
    print(f"CUDA 결과: {y_cuda[:10].cpu().numpy()}")
    print(f"PyTorch 결과: {y_pytorch[:10].cpu().numpy()}")
    print(f"최대 오차: {max_diff:.2e}")
    
    if is_correct:
        print("✅ 검증 통과!")
    else:
        print("❌ 검증 실패!")


def benchmark(n: int, iterations: int = 100) -> None:
    """성능 벤치마크"""
    print("\n" + "=" * 60)
    print(f"성능 벤치마크 (N = {n:,}, {iterations} iterations)")
    print("=" * 60)
    
    x = torch.rand(n, dtype=torch.float32, device='cuda')
    
    # Warm-up
    for _ in range(10):
        _ = conv1d_cuda(x)
    torch.cuda.synchronize()
    
    # CUDA 커널 시간 측정
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        y = conv1d_cuda(x)
    end.record()
    torch.cuda.synchronize()
    
    cuda_time = start.elapsed_time(end) / iterations
    print(f"CUDA 커널 평균 시간: {cuda_time:.4f} ms")
    
    # Grid-Stride 설정 정보
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    blocks = sm_count * 4
    threads = 256
    print(f"\nGrid-Stride 설정:")
    print(f"  • blocks_per_grid: {blocks} (SM × 4)")
    print(f"  • threads_per_block: {threads}")
    print(f"  • 총 스레드 수: {blocks * threads:,}")
    print(f"  • 스레드당 작업량: ~{cdiv(n, blocks * threads):,}개")


def test_edge_cases() -> None:
    """경계 조건 테스트"""
    print("\n" + "=" * 60)
    print("경계 조건 테스트")
    print("=" * 60)
    
    # 작은 배열 테스트
    x_small = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    y_cuda = conv1d_cuda(x_small)
    y_expected = torch.tensor([
        0.2*0 + 0.6*1 + 0.2*2,  # i=0: left=0
        0.2*1 + 0.6*2 + 0.2*3,  # i=1
        0.2*2 + 0.6*3 + 0.2*4,  # i=2
        0.2*3 + 0.6*4 + 0.2*5,  # i=3
        0.2*4 + 0.6*5 + 0.2*0,  # i=4: right=0
    ], device='cuda')
    
    print(f"입력: {x_small.cpu().numpy()}")
    print(f"CUDA 출력: {y_cuda.cpu().numpy()}")
    print(f"예상 출력: {y_expected.cpu().numpy()}")
    
    if torch.allclose(y_cuda, y_expected, rtol=1e-5):
        print("✅ 경계 조건 처리 정확!")
    else:
        print("❌ 경계 조건 처리 오류!")


if __name__ == "__main__":
    print("\n" + "🚀 Week 2 과제 2: 1D Convolution".center(60))
    print("=" * 60)
    
    # GPU 정보
    device_name = torch.cuda.get_device_name(0)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n🖥️  GPU: {device_name}")
    print(f"   SM 개수: {sm_count}")
    
    # 경계 조건 테스트
    test_edge_cases()
    
    # 정확성 검증 (큰 배열)
    n = 1_000_000
    x = torch.rand(n, dtype=torch.float32, device='cuda')
    verify_correctness(x)
    
    # 성능 벤치마크
    benchmark(n=10_000_000, iterations=100)
    
    print("\n" + "=" * 60)
    print("🎉 Week 2 과제 2 완료!")
    print("=" * 60 + "\n")
