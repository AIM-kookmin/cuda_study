"""
Week 3 심화: Bank Conflicts 실전 체험

목표:
1. Bank Conflicts 현상을 직접 측정하고 체험
2. Padding과 Swizzling 기법의 효과 정량적 확인
3. 다양한 접근 패턴에서의 성능 차이 분석
4. NVIDIA Profiler 없이도 Bank Conflicts를 감지하는 방법 학습

기반: CUDA C Programming Guide, NVIDIA Developer Documentation
"""
import os
import math
import torch
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"

BLOCK_SIZE = 32  # 32x32 = 1024 threads (Warp size와 일치)


# =============================================================================
# Bank Conflicts 측정 CUDA 커널
# =============================================================================

cuda_source = f"""
#include <torch/extension.h>

#define BLOCK_SIZE {BLOCK_SIZE}
#define WARP_SIZE 32

// 🔴 32-way Bank Conflict (최악의 경우)
__global__ void bank_conflict_32way(
    float* output,
    int iterations
) {{
    __shared__ float data[BLOCK_SIZE][BLOCK_SIZE];  // No padding
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % WARP_SIZE;
    
    // 데이터 초기화
    data[threadIdx.y][threadIdx.x] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    
    // 모든 Thread가 같은 column 접근 → 32-way conflict
    for (int i = 0; i < iterations; i++) {{
        sum += data[lane_id][0];  // Column 0에 모든 Thread 몰림!
    }}
    
    output[tid] = sum;
}}

// 🟡 2-way Bank Conflict (중간 수준)
__global__ void bank_conflict_2way(
    float* output,
    int iterations
) {{
    __shared__ float data[BLOCK_SIZE][BLOCK_SIZE];
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % WARP_SIZE;
    
    data[threadIdx.y][threadIdx.x] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    
    // 2개 Thread씩 같은 Bank → 2-way conflict
    for (int i = 0; i < iterations; i++) {{
        sum += data[lane_id % 16][0];  // 16개 위치 → 2-way conflict
    }}
    
    output[tid] = sum;
}}

// 🟢 No Bank Conflict (최적)
__global__ void no_bank_conflict(
    float* output,
    int iterations
) {{
    __shared__ float data[BLOCK_SIZE][BLOCK_SIZE];
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % WARP_SIZE;
    
    data[threadIdx.y][threadIdx.x] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    
    // 각 Thread가 다른 Bank 접근 → No conflict
    for (int i = 0; i < iterations; i++) {{
        sum += data[0][lane_id];  // Row 0의 다른 column → 각각 다른 Bank
    }}
    
    output[tid] = sum;
}}

// ⭐ Broadcast (예외: 빠름)
__global__ void broadcast_access(
    float* output,
    int iterations
) {{
    __shared__ float data[BLOCK_SIZE][BLOCK_SIZE];
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    data[threadIdx.y][threadIdx.x] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    
    // 모든 Thread가 정확히 같은 위치 접근 → Broadcast (빠름!)
    for (int i = 0; i < iterations; i++) {{
        sum += data[0][0];  // 모든 Thread가 같은 주소
    }}
    
    output[tid] = sum;
}}

// 🛠️ Padding으로 Bank Conflict 해결
__global__ void padded_no_conflict(
    float* output,
    int iterations
) {{
    __shared__ float data[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 Padding!
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % WARP_SIZE;
    
    data[threadIdx.y][threadIdx.x] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    
    // 패딩으로 인해 Column 접근도 Bank Conflict 없음
    for (int i = 0; i < iterations; i++) {{
        sum += data[lane_id][0];  // 이제 각각 다른 Bank!
    }}
    
    output[tid] = sum;
}}

// 🔄 Swizzling으로 Bank Conflict 해결
__global__ void swizzled_no_conflict(
    float* output,
    int iterations
) {{
    __shared__ float data[BLOCK_SIZE][BLOCK_SIZE];
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % WARP_SIZE;
    
    // Swizzling: XOR 연산으로 Bank 분산
    int swizzled_x = threadIdx.x ^ threadIdx.y;
    data[threadIdx.y][swizzled_x] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    
    for (int i = 0; i < iterations; i++) {{
        int access_x = lane_id ^ (i % BLOCK_SIZE);  // Dynamic swizzling
        sum += data[i % BLOCK_SIZE][access_x % BLOCK_SIZE];
    }}
    
    output[tid] = sum;
}}

// C++ Wrapper 함수들
void launch_bank_conflict_32way(torch::Tensor output, int iterations) {{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1, 1);
    bank_conflict_32way<<<grid, block>>>(output.data_ptr<float>(), iterations);
}}

void launch_bank_conflict_2way(torch::Tensor output, int iterations) {{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1, 1);
    bank_conflict_2way<<<grid, block>>>(output.data_ptr<float>(), iterations);
}}

void launch_no_bank_conflict(torch::Tensor output, int iterations) {{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1, 1);
    no_bank_conflict<<<grid, block>>>(output.data_ptr<float>(), iterations);
}}

void launch_broadcast_access(torch::Tensor output, int iterations) {{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1, 1);
    broadcast_access<<<grid, block>>>(output.data_ptr<float>(), iterations);
}}

void launch_padded_no_conflict(torch::Tensor output, int iterations) {{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1, 1);
    padded_no_conflict<<<grid, block>>>(output.data_ptr<float>(), iterations);
}}

void launch_swizzled_no_conflict(torch::Tensor output, int iterations) {{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1, 1);
    swizzled_no_conflict<<<grid, block>>>(output.data_ptr<float>(), iterations);
}}
"""

cpp_header = """
void launch_bank_conflict_32way(torch::Tensor output, int iterations);
void launch_bank_conflict_2way(torch::Tensor output, int iterations);
void launch_no_bank_conflict(torch::Tensor output, int iterations);
void launch_broadcast_access(torch::Tensor output, int iterations);
void launch_padded_no_conflict(torch::Tensor output, int iterations);
void launch_swizzled_no_conflict(torch::Tensor output, int iterations);
"""

# JIT 컴파일
print("=" * 70)
print("Compiling Bank Conflicts Demonstration Kernels...")
print("=" * 70)

module = load_inline(
    name='bank_conflicts_extension',
    cpp_sources=[cpp_header],
    cuda_sources=[cuda_source],
    functions=['launch_bank_conflict_32way', 'launch_bank_conflict_2way', 
               'launch_no_bank_conflict', 'launch_broadcast_access',
               'launch_padded_no_conflict', 'launch_swizzled_no_conflict'],
    verbose=True
)

print("Compilation done!\n")


# =============================================================================
# 벤치마크 및 분석 함수들
# =============================================================================

def benchmark_bank_conflicts(iterations: int = 10000) -> None:
    """다양한 Bank Conflict 패턴 성능 측정"""
    print("=" * 80)
    print("Bank Conflicts 성능 비교 실험")
    print("=" * 80)
    
    # 출력 텐서 준비
    output_size = BLOCK_SIZE * BLOCK_SIZE
    output = torch.zeros(output_size, dtype=torch.float32, device='cuda')
    
    kernels = [
        ("32-way Conflict (최악)", module.launch_bank_conflict_32way, "🔴"),
        ("2-way Conflict (중간)", module.launch_bank_conflict_2way, "🟡"),
        ("No Conflict (최적)", module.launch_no_bank_conflict, "🟢"),
        ("Broadcast (특수)", module.launch_broadcast_access, "⭐"),
        ("Padded (최적화)", module.launch_padded_no_conflict, "🛠️"),
        ("Swizzled (최적화)", module.launch_swizzled_no_conflict, "🔄"),
    ]
    
    results = []
    
    print(f"측정 설정: {BLOCK_SIZE}×{BLOCK_SIZE} = {output_size} threads, {iterations:,} iterations\n")
    print(f"{'Pattern':<25} {'Time (ms)':<12} {'Relative':<10} {'Bandwidth':<12}")
    print("-" * 70)
    
    for name, kernel_func, icon in kernels:
        # Warm-up
        for _ in range(3):
            kernel_func(output, 100)
        torch.cuda.synchronize()
        
        # 측정
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        kernel_func(output, iterations)
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end)
        results.append((name, time_ms, icon))
        
        # 첫 번째를 기준으로 상대적 성능 계산
        if len(results) == 1:
            baseline_time = time_ms
            relative = 1.0
        else:
            relative = baseline_time / time_ms  # 높을수록 좋음
        
        # 가상의 Bandwidth 계산 (참고용)
        # 실제로는 Shared Memory Access이므로 Global Memory Bandwidth와는 다름
        operations = output_size * iterations
        bandwidth = operations * 4 / (time_ms / 1000) / 1e9  # GB/s (가상)
        
        print(f"{icon} {name:<22} {time_ms:<12.4f} {relative:<10.2f} {bandwidth:<12.1f}")
    
    return results


def analyze_bank_mapping() -> None:
    """Bank 매핑 원리 설명 및 시각화"""
    print("\n" + "=" * 80)
    print("Bank Mapping 원리 분석")
    print("=" * 80)
    
    print("Bank 매핑 공식: bank_id = (address / 4) % 32")
    print("여기서 address는 float 배열의 바이트 주소")
    
    # 2D 배열의 Bank 매핑 예시
    print(f"\n2D 배열 shared[{BLOCK_SIZE}][{BLOCK_SIZE}]의 Bank 매핑:")
    print("(첫 번째 Warp의 접근 패턴 분석)")
    print()
    
    # Row-wise 접근 (Good)
    print("✅ Row-wise 접근: shared[0][threadIdx.x]")
    print("Thread → Bank 매핑:")
    for i in range(min(16, BLOCK_SIZE)):  # 처음 16개만 출력
        bank_id = i % 32
        print(f"  Thread {i:2d} → shared[0][{i:2d}] → Bank {bank_id:2d}")
    if BLOCK_SIZE > 16:
        print("  ... (각 Thread가 서로 다른 Bank)")
    
    print()
    
    # Column-wise 접근 (Bad)
    print("❌ Column-wise 접근: shared[threadIdx.x][0]")
    print("Thread → Bank 매핑:")
    for i in range(min(16, BLOCK_SIZE)):
        # shared[i][0]의 주소 = i * BLOCK_SIZE * 4 + 0 * 4
        address = i * BLOCK_SIZE * 4
        bank_id = (address // 4) % 32
        print(f"  Thread {i:2d} → shared[{i:2d}][0] → Bank {bank_id:2d}")
    if BLOCK_SIZE > 16:
        print("  ... (모든 Thread가 같은 Bank 0!)")
    
    print()
    
    # Padded 접근 (Fixed)
    padded_width = BLOCK_SIZE + 1
    print(f"✅ Padded 접근: shared[threadIdx.x][0] (width = {padded_width})")
    print("Thread → Bank 매핑:")
    for i in range(min(16, BLOCK_SIZE)):
        # shared[i][0]의 주소 = i * padded_width * 4 + 0 * 4
        address = i * padded_width * 4
        bank_id = (address // 4) % 32
        print(f"  Thread {i:2d} → shared[{i:2d}][0] → Bank {bank_id:2d}")
    if BLOCK_SIZE > 16:
        print("  ... (각 Thread가 서로 다른 Bank로 분산!)")


def demonstrate_padding_effect() -> None:
    """Padding 효과 상세 분석"""
    print("\n" + "=" * 80)
    print("Padding 효과 상세 분석")
    print("=" * 80)
    
    print("Matrix Transpose에서 Padding의 효과:")
    print()
    
    # Without Padding
    print("🔴 Without Padding (Bank Conflicts):")
    print("```cuda")
    print("__shared__ float tile[32][32];")
    print("// Read: tile[threadIdx.y][threadIdx.x] → No conflict (row-wise)")
    print("// Write: output[...] = tile[threadIdx.x][threadIdx.y] → 32-way conflict!")
    print("```")
    print("문제: tile[0][threadIdx.y], tile[1][threadIdx.y], ... 모두 같은 Bank")
    print()
    
    # With Padding
    print("✅ With Padding (No Conflicts):")
    print("```cuda")
    print("__shared__ float tile[32][33];  // +1 padding")
    print("// Read: tile[threadIdx.y][threadIdx.x] → No conflict")
    print("// Write: output[...] = tile[threadIdx.x][threadIdx.y] → No conflict!")
    print("```")
    print("해결: 각 행이 33개 원소 → 행마다 Bank 오프셋이 1씩 증가")
    print()
    
    # 수학적 설명
    print("수학적 원리:")
    print("- 원래: bank_id = (row * 32 + col) % 32 = col % 32 (row 무시)")
    print("- 패딩: bank_id = (row * 33 + col) % 32 = (row + col) % 32 (row 고려)")
    print()
    print("결과: tile[i][0]의 Bank ID = i % 32 → 모두 다른 Bank!")


def profile_memory_patterns() -> None:
    """메모리 접근 패턴별 프로파일링 가이드"""
    print("\n" + "=" * 80)
    print("Memory Access Pattern 프로파일링 가이드")
    print("=" * 80)
    
    print("NVIDIA Profiler를 이용한 Bank Conflict 측정:")
    print()
    
    print("1. Nsight Compute (권장):")
    print("```bash")
    print("ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.avg your_program")
    print("```")
    print("- 값이 1.0에 가까우면 Bank Conflict 없음")
    print("- 값이 클수록 심각한 Bank Conflict")
    print()
    
    print("2. Legacy nvprof:")
    print("```bash")
    print("nvprof --metrics shared_load_transactions_per_request your_program")
    print("```")
    print("- 이상적 값: 1.0 (각 요청당 1 transaction)")
    print("- 값이 클수록 Bank Conflict로 인한 추가 transaction")
    print()
    
    print("3. 수동 측정 (이 프로그램처럼):")
    print("- 같은 작업량에서 시간 차이 측정")
    print("- Bank Conflict가 심할수록 실행 시간 증가")
    print("- 32-way conflict는 이론상 32배까지 느려질 수 있음")


if __name__ == "__main__":
    print("\n" + "🔬 Week 3 심화: Bank Conflicts 실전 체험".center(80))
    print("=" * 80)
    
    # GPU 정보
    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    shared_mem_per_block = props.shared_memory_per_block
    
    print(f"\n🖥️  GPU: {device_name}")
    print(f"   SM Count: {sm_count}")
    print(f"   Shared Memory per Block: {shared_mem_per_block/1024:.0f} KB")
    print(f"   Block Size: {BLOCK_SIZE}×{BLOCK_SIZE} = {BLOCK_SIZE*BLOCK_SIZE} threads")
    
    # 메인 실험: Bank Conflicts 성능 비교
    results = benchmark_bank_conflicts(iterations=50000)
    
    # Bank 매핑 원리 분석
    analyze_bank_mapping()
    
    # Padding 효과 설명
    demonstrate_padding_effect()
    
    # 프로파일링 가이드
    profile_memory_patterns()
    
    print("\n" + "=" * 80)
    print("📊 실험 결과 요약")
    print("=" * 80)
    
    if len(results) >= 4:
        conflict_32_time = results[0][1]
        no_conflict_time = results[2][1]
        padded_time = results[4][1] if len(results) > 4 else results[2][1]
        
        print(f"🔴 32-way Conflict: {conflict_32_time:.4f} ms (기준)")
        print(f"🟢 No Conflict: {no_conflict_time:.4f} ms ({conflict_32_time/no_conflict_time:.1f}x 빠름)")
        print(f"🛠️  Padded: {padded_time:.4f} ms ({conflict_32_time/padded_time:.1f}x 빠름)")
        print()
        print("✨ 핵심 교훈:")
        print("1. Bank Conflict는 실제로 성능을 심각하게 저하시킨다!")
        print("2. 단순한 Padding만으로도 극적인 성능 향상이 가능하다")  
        print("3. Memory 접근 패턴이 GPU 성능의 핵심이다")
        print("4. 'No-cost' 최적화 - 알고리즘 변경 없이 패딩만으로 해결")
    
    print("\n" + "=" * 80)
    print("🎉 Bank Conflicts 마스터 완료!")
    print("=" * 80 + "\n")
