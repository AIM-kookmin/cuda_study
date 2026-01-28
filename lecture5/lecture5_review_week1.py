"""
Week 5 Day 1-2: Week 1-2 기초 복습
Vector Operations 종합 실습

목표:
1. Grid-Stride Loop 패턴 완전 숙달
2. 1D Thread Indexing 마스터
3. Boundary Check 완벽 처리
4. PyTorch 결과와 정확도 검증

구현할 커널들:
- Vector Add: c[i] = a[i] + b[i]
- Vector Scale: b[i] = alpha * a[i]  
- AXPY: y[i] = alpha * x[i] + y[i]
- Dot Product: sum(a[i] * b[i]) (Reduction 패턴)
- L2 Norm: sqrt(sum(x[i]^2))
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
# TORCH_CUDA_ARCH_LIST 제거 - GPU 아키텍처 자동 감지 사용


def cdiv(n: int, divisor: int) -> int:
    """Ceiling Division"""
    return (n + divisor - 1) // divisor


# =============================================================================
# CUDA Kernel 소스코드
# =============================================================================

cuda_source = """
#include <torch/extension.h>

// 1. Vector Add: c[i] = a[i] + b[i]
__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// 2. Vector Scale: b[i] = alpha * a[i]
__global__ void vector_scale_kernel(
    const float* __restrict__ a,
    float* __restrict__ b,
    float alpha,
    int n
) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = index; i < n; i += stride) {
        b[i] = alpha * a[i];
    }
}

// 3. AXPY: y[i] = alpha * x[i] + y[i]
__global__ void axpy_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    float alpha,
    int n
) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = index; i < n; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

// 4. Dot Product (Reduction 패턴 - 간단 버전)
__global__ void dot_product_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ partial_sums,
    int n
) {
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 각 스레드가 자기 몫 계산
    float sum = 0.0f;
    for (int i = index; i < n; i += stride) {
        sum += a[i] * b[i];
    }
    
    // Shared Memory에 저장
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Block 내 Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Block 대표가 결과 저장
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

// 5. L2 Norm: sqrt(sum(x[i]^2))
__global__ void l2_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ partial_sums,
    int n
) {
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 각 스레드가 제곱합 계산
    float sum = 0.0f;
    for (int i = index; i < n; i += stride) {
        float val = x[i];
        sum += val * val;
    }
    
    // Shared Memory에 저장
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Block 내 Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Block 대표가 결과 저장
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

// Python 인터페이스
void launch_vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c, int n) {
    int sm_count = 16;  // 기본값, 실제로는 cudaGetDeviceProperties로 가져와야 함
    int threads = 256;
    int blocks = sm_count * 4;
    
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
}

void launch_vector_scale(torch::Tensor a, torch::Tensor b, float alpha, int n) {
    int sm_count = 16;
    int threads = 256;
    int blocks = sm_count * 4;
    
    vector_scale_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        alpha,
        n
    );
}

void launch_axpy(torch::Tensor x, torch::Tensor y, float alpha, int n) {
    int sm_count = 16;
    int threads = 256;
    int blocks = sm_count * 4;
    
    axpy_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        alpha,
        n
    );
}

void launch_dot_product(torch::Tensor a, torch::Tensor b, torch::Tensor partial_sums, int n) {
    int threads = 256;
    int blocks = 64;  // Reduction이므로 적당한 수
    
    dot_product_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        n
    );
}

void launch_l2_norm(torch::Tensor x, torch::Tensor partial_sums, int n) {
    int threads = 256;
    int blocks = 64;
    
    l2_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        n
    );
}
"""

cpp_header = """
void launch_vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c, int n);
void launch_vector_scale(torch::Tensor a, torch::Tensor b, float alpha, int n);
void launch_axpy(torch::Tensor x, torch::Tensor y, float alpha, int n);
void launch_dot_product(torch::Tensor a, torch::Tensor b, torch::Tensor partial_sums, int n);
void launch_l2_norm(torch::Tensor x, torch::Tensor partial_sums, int n);
"""

# JIT 컴파일
print("=" * 60)
print("Compiling Vector Operations Kernels...")
print("=" * 60)

module = load_inline(
    name='vector_ops_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_vector_add', 'launch_vector_scale', 'launch_axpy',
               'launch_dot_product', 'launch_l2_norm'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# Python Wrapper 함수들
# =============================================================================

def vector_add_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector Add: c = a + b"""
    assert a.shape == b.shape
    assert a.is_cuda and b.is_cuda
    
    n = a.numel()
    c = torch.zeros_like(a)
    
    module.launch_vector_add(a, b, c, n)
    return c


def vector_scale_cuda(a: torch.Tensor, alpha: float) -> torch.Tensor:
    """Vector Scale: b = alpha * a"""
    assert a.is_cuda
    
    n = a.numel()
    b = torch.zeros_like(a)
    
    module.launch_vector_scale(a, b, alpha, n)
    return b


def axpy_cuda(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """AXPY: y = alpha * x + y"""
    assert x.shape == y.shape
    assert x.is_cuda and y.is_cuda
    
    n = x.numel()
    y_result = y.clone()
    
    module.launch_axpy(x, y_result, alpha, n)
    return y_result


def dot_product_cuda(a: torch.Tensor, b: torch.Tensor) -> float:
    """Dot Product: sum(a[i] * b[i])"""
    assert a.shape == b.shape
    assert a.is_cuda and b.is_cuda
    
    n = a.numel()
    partial_sums = torch.zeros(64, dtype=torch.float32, device='cuda')
    
    module.launch_dot_product(a, b, partial_sums, n)
    
    # CPU에서 최종 합산
    return partial_sums.sum().item()


def l2_norm_cuda(x: torch.Tensor) -> float:
    """L2 Norm: sqrt(sum(x[i]^2))"""
    assert x.is_cuda
    
    n = x.numel()
    partial_sums = torch.zeros(64, dtype=torch.float32, device='cuda')
    
    module.launch_l2_norm(x, partial_sums, n)
    
    # CPU에서 최종 제곱근
    sum_of_squares = partial_sums.sum().item()
    return (sum_of_squares ** 0.5)


# =============================================================================
# 검증 함수들
# =============================================================================

def verify_vector_add():
    print("=" * 60)
    print("1. Vector Add 검증")
    print("=" * 60)
    
    n = 1000000
    a = torch.randn(n, dtype=torch.float32, device='cuda')
    b = torch.randn(n, dtype=torch.float32, device='cuda')
    
    c_cuda = vector_add_cuda(a, b)
    c_pytorch = a + b
    
    is_correct = torch.allclose(c_cuda, c_pytorch, rtol=1e-5)
    max_diff = (c_cuda - c_pytorch).abs().max().item()
    
    print(f"결과: {'✅ 정확' if is_correct else '❌ 오류'}")
    print(f"최대 오차: {max_diff:.2e}\n")


def verify_vector_scale():
    print("=" * 60)
    print("2. Vector Scale 검증")
    print("=" * 60)
    
    n = 1000000
    alpha = 3.14
    a = torch.randn(n, dtype=torch.float32, device='cuda')
    
    b_cuda = vector_scale_cuda(a, alpha)
    b_pytorch = alpha * a
    
    is_correct = torch.allclose(b_cuda, b_pytorch, rtol=1e-5)
    max_diff = (b_cuda - b_pytorch).abs().max().item()
    
    print(f"결과: {'✅ 정확' if is_correct else '❌ 오류'}")
    print(f"최대 오차: {max_diff:.2e}\n")


def verify_axpy():
    print("=" * 60)
    print("3. AXPY 검증")
    print("=" * 60)
    
    n = 1000000
    alpha = 2.5
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    y = torch.randn(n, dtype=torch.float32, device='cuda')
    
    y_cuda = axpy_cuda(x, y, alpha)
    y_pytorch = alpha * x + y
    
    is_correct = torch.allclose(y_cuda, y_pytorch, rtol=1e-5)
    max_diff = (y_cuda - y_pytorch).abs().max().item()
    
    print(f"결과: {'✅ 정확' if is_correct else '❌ 오류'}")
    print(f"최대 오차: {max_diff:.2e}\n")


def verify_dot_product():
    print("=" * 60)
    print("4. Dot Product 검증")
    print("=" * 60)
    
    n = 1000000
    a = torch.randn(n, dtype=torch.float32, device='cuda')
    b = torch.randn(n, dtype=torch.float32, device='cuda')
    
    result_cuda = dot_product_cuda(a, b)
    result_pytorch = torch.dot(a, b).item()
    
    error = abs(result_cuda - result_pytorch) / abs(result_pytorch)
    
    print(f"CUDA 결과: {result_cuda:.6f}")
    print(f"PyTorch 결과: {result_pytorch:.6f}")
    print(f"상대 오차: {error:.2e}")
    print(f"결과: {'✅ 정확' if error < 1e-4 else '❌ 오류'}\n")


def verify_l2_norm():
    print("=" * 60)
    print("5. L2 Norm 검증")
    print("=" * 60)
    
    n = 1000000
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    
    result_cuda = l2_norm_cuda(x)
    result_pytorch = torch.norm(x, p=2).item()
    
    error = abs(result_cuda - result_pytorch) / abs(result_pytorch)
    
    print(f"CUDA 결과: {result_cuda:.6f}")
    print(f"PyTorch 결과: {result_pytorch:.6f}")
    print(f"상대 오차: {error:.2e}")
    print(f"결과: {'✅ 정확' if error < 1e-4 else '❌ 오류'}\n")


# =============================================================================
# 벤치마크 함수
# =============================================================================

def benchmark_all():
    print("\n" + "=" * 60)
    print("성능 벤치마크")
    print("=" * 60)
    
    sizes = [1000, 100000, 10000000]
    iterations = 100
    
    for n in sizes:
        print(f"\n데이터 크기: {n:,}")
        print("-" * 60)
        
        a = torch.randn(n, dtype=torch.float32, device='cuda')
        b = torch.randn(n, dtype=torch.float32, device='cuda')
        
        # Warm-up
        for _ in range(10):
            _ = vector_add_cuda(a, b)
        torch.cuda.synchronize()
        
        # 측정
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            _ = vector_add_cuda(a, b)
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / iterations
        bandwidth = (3 * n * 4) / (time_ms / 1000) / 1e9  # GB/s
        
        print(f"Vector Add: {time_ms:.4f} ms ({bandwidth:.1f} GB/s)")


if __name__ == "__main__":
    print("\n" + "🚀 Week 5 Day 1-2: Vector Operations Review".center(60))
    print("=" * 60)
    
    # GPU 정보
    device_name = torch.cuda.get_device_name(0)
    print(f"\n🖥️  GPU: {device_name}")
    
    # 검증
    verify_vector_add()
    verify_vector_scale()
    verify_axpy()
    verify_dot_product()
    verify_l2_norm()
    
    # 벤치마크
    benchmark_all()
    
    print("\n" + "=" * 60)
    print("🎉 Week 1-2 복습 완료!")
    print("=" * 60)
    print("\n다음: lecture5_review_week1_2d.py (2D Kernels)")
