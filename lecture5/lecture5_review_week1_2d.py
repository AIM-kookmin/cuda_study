"""
Week 5 Day 2: 2D Kernels 복습
이미지 처리 커널 실습

목표:
1. 2D Thread Indexing 완전 이해
2. 경계 조건 처리 숙달
3. Memory Coalescing 고려
4. 실제 이미지 처리 파이프라인 구현

구현할 커널들:
- RGB to Grayscale: gray = 0.21*R + 0.72*G + 0.07*B
- Image Transpose: output[j][i] = input[i][j]
- Box Blur (3x3): 주변 픽셀 평균
- Sobel Edge Detection: 엣지 검출
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
# CUDA 12.0은 compute capability 9.0까지만 지원
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;9.0"


def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division"""
    return (n + divisor - 1) // divisor


# =============================================================================
# CUDA Kernel 소스코드
# =============================================================================

cuda_source = """
#include <torch/extension.h>

// 1. RGB to Grayscale
__global__ void rgb_to_gray_kernel(
    const float* __restrict__ input,   // [H, W, 3]
    float* __restrict__ output,        // [H, W]
    int height,
    int width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int pixel_idx = (row * width + col) * 3;
        float r = input[pixel_idx + 0];
        float g = input[pixel_idx + 1];
        float b = input[pixel_idx + 2];
        
        output[row * width + col] = 0.21f * r + 0.72f * g + 0.07f * b;
    }
}

// 2. Image Transpose
__global__ void transpose_kernel(
    const float* __restrict__ input,   // [H, W]
    float* __restrict__ output,        // [W, H]
    int height,
    int width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        output[col * height + row] = input[row * width + col];
    }
}

// 3. Box Blur (3x3)
__global__ void box_blur_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        float sum = 0.0f;
        int count = 0;
        
        // 3x3 윈도우
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = row + dy;
                int nx = col + dx;
                
                // 경계 체크 (clamp)
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    sum += input[ny * width + nx];
                    count++;
                }
            }
        }
        
        output[row * width + col] = sum / count;
    }
}

// 4. Sobel Edge Detection
__global__ void sobel_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        // Sobel X kernel
        float gx = 
            -1.0f * input[(row-1) * width + (col-1)] +
            -2.0f * input[(row  ) * width + (col-1)] +
            -1.0f * input[(row+1) * width + (col-1)] +
             1.0f * input[(row-1) * width + (col+1)] +
             2.0f * input[(row  ) * width + (col+1)] +
             1.0f * input[(row+1) * width + (col+1)];
        
        // Sobel Y kernel
        float gy = 
            -1.0f * input[(row-1) * width + (col-1)] +
            -2.0f * input[(row-1) * width + (col  )] +
            -1.0f * input[(row-1) * width + (col+1)] +
             1.0f * input[(row+1) * width + (col-1)] +
             2.0f * input[(row+1) * width + (col  )] +
             1.0f * input[(row+1) * width + (col+1)];
        
        // Gradient magnitude
        output[row * width + col] = sqrtf(gx * gx + gy * gy);
    } else if (row < height && col < width) {
        output[row * width + col] = 0.0f;
    }
}

// Launcher functions
void launch_rgb_to_gray(
    torch::Tensor input, torch::Tensor output,
    int height, int width
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    rgb_to_gray_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height, width
    );
}

void launch_transpose(
    torch::Tensor input, torch::Tensor output,
    int height, int width
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    transpose_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height, width
    );
}

void launch_box_blur(
    torch::Tensor input, torch::Tensor output,
    int height, int width
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    box_blur_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height, width
    );
}

void launch_sobel(
    torch::Tensor input, torch::Tensor output,
    int height, int width
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    sobel_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height, width
    );
}
"""

cpp_header = """
void launch_rgb_to_gray(torch::Tensor input, torch::Tensor output, int height, int width);
void launch_transpose(torch::Tensor input, torch::Tensor output, int height, int width);
void launch_box_blur(torch::Tensor input, torch::Tensor output, int height, int width);
void launch_sobel(torch::Tensor input, torch::Tensor output, int height, int width);
"""

print("=" * 60)
print("Compiling 2D Image Processing Kernels...")
print("=" * 60)

module = load_inline(
    name='image_ops_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_rgb_to_gray', 'launch_transpose', 'launch_box_blur', 'launch_sobel'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# Python Wrapper 함수들
# =============================================================================

def rgb_to_gray_cuda(image: torch.Tensor) -> torch.Tensor:
    """RGB to Grayscale: [H, W, 3] -> [H, W]"""
    assert image.dim() == 3 and image.shape[2] == 3
    assert image.is_cuda
    
    height, width = image.shape[0], image.shape[1]
    output = torch.zeros(height, width, dtype=torch.float32, device='cuda')
    
    module.launch_rgb_to_gray(image.contiguous(), output, height, width)
    return output


def transpose_cuda(image: torch.Tensor) -> torch.Tensor:
    """Image Transpose: [H, W] -> [W, H]"""
    assert image.dim() == 2
    assert image.is_cuda
    
    height, width = image.shape
    output = torch.zeros(width, height, dtype=torch.float32, device='cuda')
    
    module.launch_transpose(image.contiguous(), output, height, width)
    return output


def box_blur_cuda(image: torch.Tensor) -> torch.Tensor:
    """Box Blur (3x3): [H, W] -> [H, W]"""
    assert image.dim() == 2
    assert image.is_cuda
    
    height, width = image.shape
    output = torch.zeros_like(image)
    
    module.launch_box_blur(image.contiguous(), output, height, width)
    return output


def sobel_cuda(image: torch.Tensor) -> torch.Tensor:
    """Sobel Edge Detection: [H, W] -> [H, W]"""
    assert image.dim() == 2
    assert image.is_cuda
    
    height, width = image.shape
    output = torch.zeros_like(image)
    
    module.launch_sobel(image.contiguous(), output, height, width)
    return output


# =============================================================================
# 검증 함수들
# =============================================================================

def verify_rgb_to_gray():
    print("=" * 60)
    print("1. RGB to Grayscale 검증")
    print("=" * 60)
    
    H, W = 512, 512
    image = torch.rand(H, W, 3, dtype=torch.float32, device='cuda')
    
    gray_cuda = rgb_to_gray_cuda(image)
    gray_pytorch = 0.21 * image[:, :, 0] + 0.72 * image[:, :, 1] + 0.07 * image[:, :, 2]
    
    is_correct = torch.allclose(gray_cuda, gray_pytorch, rtol=1e-5)
    max_diff = (gray_cuda - gray_pytorch).abs().max().item()
    
    print(f"이미지 크기: {H}x{W}")
    print(f"결과: {'✅ 정확' if is_correct else '❌ 오류'}")
    print(f"최대 오차: {max_diff:.2e}\n")


def verify_transpose():
    print("=" * 60)
    print("2. Image Transpose 검증")
    print("=" * 60)
    
    H, W = 256, 512
    image = torch.randn(H, W, dtype=torch.float32, device='cuda')
    
    transposed_cuda = transpose_cuda(image)
    transposed_pytorch = image.t()
    
    is_correct = torch.allclose(transposed_cuda, transposed_pytorch, rtol=1e-5)
    max_diff = (transposed_cuda - transposed_pytorch).abs().max().item()
    
    print(f"입력 크기: {H}x{W} -> 출력 크기: {W}x{H}")
    print(f"결과: {'✅ 정확' if is_correct else '❌ 오류'}")
    print(f"최대 오차: {max_diff:.2e}\n")


def verify_box_blur():
    print("=" * 60)
    print("3. Box Blur 검증")
    print("=" * 60)
    
    H, W = 128, 128
    image = torch.randn(H, W, dtype=torch.float32, device='cuda')
    
    blurred_cuda = box_blur_cuda(image)
    
    # PyTorch로 검증 (단순 구현)
    padded = torch.nn.functional.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    kernel = torch.ones(1, 1, 3, 3, device='cuda') / 9.0
    blurred_pytorch = torch.nn.functional.conv2d(padded, kernel).squeeze()
    
    # 경계 처리 방식이 다르므로 내부 영역만 비교
    inner_cuda = blurred_cuda[1:-1, 1:-1]
    inner_pytorch = blurred_pytorch[1:-1, 1:-1]
    
    max_diff = (inner_cuda - inner_pytorch).abs().max().item()
    
    print(f"이미지 크기: {H}x{W}")
    print(f"최대 오차 (내부 영역): {max_diff:.2e}")
    print(f"결과: {'✅ 합리적' if max_diff < 0.1 else '❌ 확인 필요'}\n")


def verify_sobel():
    print("=" * 60)
    print("4. Sobel Edge Detection 검증")
    print("=" * 60)
    
    H, W = 128, 128
    # 간단한 패턴 생성 (흰색 사각형)
    image = torch.zeros(H, W, dtype=torch.float32, device='cuda')
    image[32:96, 32:96] = 1.0
    
    edges_cuda = sobel_cuda(image)
    
    # 엣지가 검출되었는지 확인
    edge_strength = edges_cuda.max().item()
    non_zero = (edges_cuda > 0.1).sum().item()
    
    print(f"이미지 크기: {H}x{W}")
    print(f"최대 엣지 강도: {edge_strength:.4f}")
    print(f"엣지 픽셀 수: {non_zero}")
    print(f"결과: {'✅ 엣지 검출됨' if edge_strength > 1.0 else '❌ 확인 필요'}\n")


# =============================================================================
# 벤치마크
# =============================================================================

def benchmark_all():
    print("=" * 60)
    print("성능 벤치마크")
    print("=" * 60)
    
    sizes = [(512, 512), (1024, 1024), (1920, 1080)]
    iterations = 100
    
    for H, W in sizes:
        print(f"\n이미지 크기: {H}x{W}")
        print("-" * 60)
        
        # RGB to Gray
        image_rgb = torch.rand(H, W, 3, dtype=torch.float32, device='cuda')
        for _ in range(10):
            _ = rgb_to_gray_cuda(image_rgb)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            _ = rgb_to_gray_cuda(image_rgb)
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / iterations
        print(f"RGB to Gray: {time_ms:.4f} ms")
        
        # Box Blur
        image_2d = torch.rand(H, W, dtype=torch.float32, device='cuda')
        for _ in range(10):
            _ = box_blur_cuda(image_2d)
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(iterations):
            _ = box_blur_cuda(image_2d)
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / iterations
        print(f"Box Blur: {time_ms:.4f} ms")


if __name__ == "__main__":
    print("\n" + "🚀 Week 5 Day 2: 2D Image Processing Review".center(60))
    print("=" * 60)
    
    device_name = torch.cuda.get_device_name(0)
    print(f"\n🖥️  GPU: {device_name}")
    
    # 검증
    verify_rgb_to_gray()
    verify_transpose()
    verify_box_blur()
    verify_sobel()
    
    # 벤치마크
    benchmark_all()
    
    print("\n" + "=" * 60)
    print("🎉 2D Kernels 복습 완료!")
    print("=" * 60)
    print("\nDay 3으로 계속...")
