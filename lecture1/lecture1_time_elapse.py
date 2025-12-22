import os
import torch
from torch.utils.cpp_extension import load_inline

# 1. CUDA 소스 (구현부)
cuda_source = """
#include <torch/extension.h>
#include <cstdio>

__global__ void hello_kernel() {
    printf("🚀 Hello from CUDA Kernel! Block %d Thread %d\\n", blockIdx.x, threadIdx.x);
}

void hello(int blocks, int threads) {
    hello_kernel<<<blocks, threads>>>();
    cudaDeviceSynchronize();
}
"""

# 2. C++ 헤더 (선언부) - 중요!
# 컴파일러에게 "hello라는 함수가 있어"라고 미리 알려줍니다.
cpp_header = "void hello(int blocks, int threads);"

os.environ["CC"]= "gcc-10"
os.environ["CXX"]= "g++-10"

# JIT 컴파일
my_module = load_inline(
    name='hello_extension',
    cpp_sources=[cpp_header],    # 선언부
    cuda_sources=[cuda_source],  # 구현부
    functions=['hello'],         # 'hello_kernel'은 지워야 합니다!
    verbose=True
)

def benchmark_kernal():
    print("\n=== Benchmarking CUDA Kernel ===")
    blocks = 2
    threads = 2

    print("Warming up...")
    my_module.hello(blocks, threads)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    my_module.hello(blocks, threads)    
    end_event.record()

    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Elapsed Time: {elapsed_time:.3f} ms")
    print("=== Benchmark Finished ===")

if __name__ == "__main__":
    benchmark_kernal()