"""
Week 1 과제 3: RGB to Grayscale CUDA 커널 구현

입력: [H, W, 3] 크기의 RGB 이미지 텐서
출력: [H, W] 크기의 Grayscale 이미지 텐서
공식: gray = 0.21 * R + 0.72 * G + 0.07 * B

핵심: 2D 인덱싱
    row = blockIdx.y * blockDim.y + threadIdx.y
    col = blockIdx.x * blockDim.x + threadIdx.x
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


# CUDA 커널 소스코드
cuda_source = """
#include <torch/extension.h>

__global__ void rgb_to_grayscale_kernel(
    const float* __restrict__ input,   // [H, W, 3] RGB 이미지
    float* __restrict__ output,         // [H, W] Grayscale 이미지
    int height,
    int width
) {
    // 2D 인덱싱: 각 스레드가 하나의 픽셀을 처리
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 범위 체크 (Boundary Check) - 이미지 크기를 벗어나면 리턴
    if (row >= height || col >= width) {
        return;
    }
    
    // 입력 텐서에서 RGB 값 읽기
    // 메모리 레이아웃: [H, W, 3] -> index = (row * width + col) * 3 + channel
    int pixel_idx = (row * width + col) * 3;
    float r = input[pixel_idx + 0];
    float g = input[pixel_idx + 1];
    float b = input[pixel_idx + 2];
    
    // Grayscale 변환 공식: 0.21*R + 0.72*G + 0.07*B
    float gray = 0.21f * r + 0.72f * g + 0.07f * b;
    
    // 출력 텐서에 저장
    // 메모리 레이아웃: [H, W] -> index = row * width + col
    output[row * width + col] = gray;
}

void rgb_to_grayscale(
    torch::Tensor input,
    torch::Tensor output,
    int height,
    int width,
    int block_x,
    int block_y,
    int grid_x,
    int grid_y
) {
    // 2D Grid/Block 설정
    dim3 block(block_x, block_y);
    dim3 grid(grid_x, grid_y);
    
    // 커널 실행
    rgb_to_grayscale_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width
    );
    
    // 에러 체크를 위한 동기화
    cudaDeviceSynchronize();
}
"""

# C++ 헤더 (함수 선언)
cpp_header = """
void rgb_to_grayscale(
    torch::Tensor input,
    torch::Tensor output,
    int height,
    int width,
    int block_x,
    int block_y,
    int grid_x,
    int grid_y
);
"""

# JIT 컴파일
print("Compiling CUDA kernel...")
module = load_inline(
    name='rgb_to_grayscale_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['rgb_to_grayscale'],
    verbose=True
)
print("Compilation done!\n")


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
    RGB 이미지를 Grayscale로 변환하는 Python wrapper 함수
    
    Args:
        image: [H, W, 3] 크기의 RGB 이미지 (float32, CUDA)
    
    Returns:
        [H, W] 크기의 Grayscale 이미지 (float32, CUDA)
    """
    assert image.dim() == 3 and image.shape[2] == 3, \
        f"입력은 [H, W, 3] 형태여야 합니다. 현재: {image.shape}"
    assert image.is_cuda, "입력 텐서는 CUDA 텐서여야 합니다."
    assert image.dtype == torch.float32, "입력 텐서는 float32여야 합니다."
    
    # 이미지가 메모리에서 연속적인지 확인
    image = image.contiguous()
    
    height, width = image.shape[0], image.shape[1]
    
    # 출력 텐서 생성
    output = torch.zeros(height, width, dtype=torch.float32, device='cuda')
    
    # Block/Grid 크기 설정 (16x16 = 256 threads per block)
    block_x, block_y = 16, 16
    grid_x = cdiv(width, block_x)
    grid_y = cdiv(height, block_y)
    
    # 커널 실행
    module.rgb_to_grayscale(
        image, output,
        height, width,
        block_x, block_y,
        grid_x, grid_y
    )
    
    return output


def verify_with_pytorch(image: torch.Tensor, cuda_result: torch.Tensor) -> tuple:
    """PyTorch 구현과 비교하여 검증"""
    # PyTorch로 동일한 연산 수행
    pytorch_result = 0.21 * image[:, :, 0] + 0.72 * image[:, :, 1] + 0.07 * image[:, :, 2]
    
    # 결과 비교 (부동소수점 오차 허용)
    is_close = torch.allclose(cuda_result, pytorch_result, rtol=1e-5, atol=1e-5)
    max_diff = (cuda_result - pytorch_result).abs().max().item()
    
    return is_close, max_diff


if __name__ == "__main__":
    print("=" * 60)
    print("RGB to Grayscale CUDA 커널 테스트")
    print("=" * 60)
    
    # 테스트 이미지 생성 (랜덤 RGB 이미지)
    H, W = 1080, 1920  # Full HD 크기
    print(f"\n테스트 이미지 크기: {H} x {W} x 3 (RGB)")
    
    # 랜덤 RGB 이미지 생성 (0~1 범위)
    image = torch.rand(H, W, 3, dtype=torch.float32, device='cuda')
    print(f"입력 텐서: shape={image.shape}, dtype={image.dtype}, device={image.device}")
    
    # Grayscale 변환 실행
    print("\nCUDA 커널 실행 중...")
    gray = rgb_to_grayscale(image)
    print(f"출력 텐서: shape={gray.shape}, dtype={gray.dtype}, device={gray.device}")
    
    # PyTorch 결과와 비교 검증
    print("\n검증 중 (PyTorch 결과와 비교)...")
    is_correct, max_diff = verify_with_pytorch(image, gray)
    
    if is_correct:
        print(f"✅ 검증 통과! (최대 오차: {max_diff:.2e})")
    else:
        print(f"❌ 검증 실패! (최대 오차: {max_diff:.2e})")
    
    # 성능 벤치마크
    print("\n" + "=" * 60)
    print("성능 벤치마크")
    print("=" * 60)
    
    # Warm-up
    for _ in range(3):
        _ = rgb_to_grayscale(image)
    torch.cuda.synchronize()
    
    # CUDA 커널 시간 측정
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    iterations = 100
    start_event.record()
    for _ in range(iterations):
        _ = rgb_to_grayscale(image)
    end_event.record()
    torch.cuda.synchronize()
    
    cuda_time = start_event.elapsed_time(end_event) / iterations
    print(f"CUDA 커널 평균 시간: {cuda_time:.4f} ms")
    
    # PyTorch 시간 측정 (비교용)
    start_event.record()
    for _ in range(iterations):
        _ = 0.21 * image[:, :, 0] + 0.72 * image[:, :, 1] + 0.07 * image[:, :, 2]
    end_event.record()
    torch.cuda.synchronize()
    
    pytorch_time = start_event.elapsed_time(end_event) / iterations
    print(f"PyTorch 평균 시간: {pytorch_time:.4f} ms")
    
    speedup = pytorch_time / cuda_time if cuda_time > 0 else float('inf')
    print(f"속도 비교: {speedup:.2f}x {'(CUDA 더 빠름)' if speedup > 1 else '(PyTorch 더 빠름)'}")
    
    print("\n" + "=" * 60)
    print("🎉 Week 1 과제 3 완료!")
    print("=" * 60)
