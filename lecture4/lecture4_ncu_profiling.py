"""
Week 4 실습 2: NVIDIA Nsight Compute (ncu) 프로파일링 자동화

목표:
1. ncu 명령어를 Python에서 자동 실행
2. 주요 성능 메트릭 수집 및 분석
3. CSV 출력 파싱으로 정량적 성능 분석
4. Before/After 최적화 효과 측정

핵심 기능:
- 자동 ncu 프로파일링 실행
- 메트릭 데이터 파싱 및 시각화
- 병목 유형 자동 진단
- 최적화 전후 성능 비교
"""
import os
import subprocess
import csv
import json
import tempfile
import shutil
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"


# =============================================================================
# 테스트용 CUDA 커널들
# =============================================================================

cuda_source = """
#include <torch/extension.h>

// 최적화 전: Bank Conflicts가 있는 Matrix Transpose
__global__ void transpose_with_conflicts(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    __shared__ float tile[32][32];  // No padding - Bank conflicts 발생
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (y < M && x < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    
    __syncthreads();
    
    int out_x = blockIdx.y * 32 + threadIdx.x;
    int out_y = blockIdx.x * 32 + threadIdx.y;
    
    if (out_y < N && out_x < M) {
        output[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];  // Bank conflicts!
    }
}

// 최적화 후: Bank Conflicts 해결된 Matrix Transpose
__global__ void transpose_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    __shared__ float tile[32][33];  // +1 padding - Bank conflicts 해결
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (y < M && x < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    
    __syncthreads();
    
    int out_x = blockIdx.y * 32 + threadIdx.x;
    int out_y = blockIdx.x * 32 + threadIdx.y;
    
    if (out_y < N && out_x < M) {
        output[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];  // No conflicts!
    }
}

// 낮은 Occupancy 커널 (많은 레지스터 사용)
__global__ void low_occupancy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 많은 지역 변수로 레지스터 압박
        float r1=input[i], r2=input[i], r3=input[i], r4=input[i];
        float r5=input[i], r6=input[i], r7=input[i], r8=input[i];
        float r9=input[i], r10=input[i], r11=input[i], r12=input[i];
        float r13=input[i], r14=input[i], r15=input[i], r16=input[i];
        
        for (int j = 0; j < 100; j++) {
            r1 += r2*1.1f; r2 += r3*1.1f; r3 += r4*1.1f; r4 += r5*1.1f;
            r5 += r6*1.1f; r6 += r7*1.1f; r7 += r8*1.1f; r8 += r9*1.1f;
            r9 += r10*1.1f; r10 += r11*1.1f; r11 += r12*1.1f; r12 += r13*1.1f;
            r13 += r14*1.1f; r14 += r15*1.1f; r15 += r16*1.1f; r16 += r1*1.1f;
        }
        
        output[i] = r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+r16;
    }
}

// 높은 Occupancy 커널 (적은 레지스터 사용)
__global__ void high_occupancy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < 1000; j++) {
            sum += input[i] * 1.001f;  // 간단한 연산
        }
        output[i] = sum;
    }
}

// Python 인터페이스
torch::Tensor transpose_with_conflicts_cuda(torch::Tensor input) {
    const int M = input.size(0);
    const int N = input.size(1);
    auto output = torch::zeros({N, M}, input.options());
    
    const dim3 blockSize(32, 32);
    const dim3 gridSize((N + 31) / 32, (M + 31) / 32);
    
    transpose_with_conflicts<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), M, N
    );
    
    return output;
}

torch::Tensor transpose_optimized_cuda(torch::Tensor input) {
    const int M = input.size(0);
    const int N = input.size(1);
    auto output = torch::zeros({N, M}, input.options());
    
    const dim3 blockSize(32, 32);
    const dim3 gridSize((N + 31) / 32, (M + 31) / 32);
    
    transpose_optimized<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), M, N
    );
    
    return output;
}

torch::Tensor low_occupancy_cuda(torch::Tensor input) {
    const int n = input.size(0);
    auto output = torch::zeros_like(input);
    
    const dim3 blockSize(64);  // 작은 블록
    const dim3 gridSize((n + 63) / 64);
    
    low_occupancy_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    
    return output;
}

torch::Tensor high_occupancy_cuda(torch::Tensor input) {
    const int n = input.size(0);
    auto output = torch::zeros_like(input);
    
    const dim3 blockSize(256);  // 큰 블록
    const dim3 gridSize((n + 255) / 256);
    
    high_occupancy_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor transpose_with_conflicts_cuda(torch::Tensor input);
torch::Tensor transpose_optimized_cuda(torch::Tensor input);
torch::Tensor low_occupancy_cuda(torch::Tensor input);
torch::Tensor high_occupancy_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_conflicts", &transpose_with_conflicts_cuda, "Transpose with Bank Conflicts");
    m.def("transpose_optimized", &transpose_optimized_cuda, "Optimized Transpose");
    m.def("low_occupancy", &low_occupancy_cuda, "Low Occupancy Kernel");
    m.def("high_occupancy", &high_occupancy_cuda, "High Occupancy Kernel");
}
"""

print("🔨 JIT 컴파일 중... (NCU 프로파일링용 커널들)")
module = load_inline(
    name='ncu_profiling_kernels',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    verbose=False
)
print("✅ 컴파일 완료!")


# =============================================================================
# NCU 프로파일링 자동화 클래스
# =============================================================================

class NCUProfiler:
    """NVIDIA Nsight Compute 자동 프로파일링 클래스"""
    
    def __init__(self):
        self.ncu_available = self._check_ncu_availability()
        self.temp_dir = tempfile.mkdtemp(prefix="ncu_profile_")
        
    def _check_ncu_availability(self):
        """ncu 명령어 사용 가능성 확인"""
        try:
            result = subprocess.run(['ncu', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ NVIDIA Nsight Compute 발견: {result.stdout.split()[2]}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        print("⚠️ ncu 명령어를 찾을 수 없습니다.")
        print("   설치 가이드: https://developer.nvidia.com/nsight-compute")
        print("   시뮬레이션 모드로 계속 진행합니다.")
        return False
    
    def profile_kernel(self, python_script_content, kernel_name=None, 
                      metrics=None, launch_skip=5, launch_count=3):
        """커널을 프로파일링하고 결과 반환"""
        
        if not self.ncu_available:
            return self._simulate_ncu_output(kernel_name or "unknown_kernel")
            
        # 임시 Python 스크립트 생성
        script_path = Path(self.temp_dir) / "profile_script.py"
        with open(script_path, 'w') as f:
            f.write(python_script_content)
        
        # 기본 메트릭 설정
        if metrics is None:
            metrics = [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed", 
                "l1tex__t_throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
            ]
        
        # ncu 명령어 구성
        csv_output = Path(self.temp_dir) / "ncu_output.csv"
        ncu_cmd = [
            'ncu', '--csv', '--log-file', str(csv_output),
            '--metrics', ','.join(metrics),
            '--launch-skip', str(launch_skip),
            '--launch-count', str(launch_count),
        ]
        
        if kernel_name:
            ncu_cmd.extend(['--kernel-name', kernel_name])
            
        ncu_cmd.extend(['python', str(script_path)])
        
        try:
            print(f"🔍 NCU 프로파일링 시작: {kernel_name or 'all kernels'}")
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, 
                                  timeout=120, cwd=self.temp_dir)
            
            if result.returncode != 0:
                print(f"❌ NCU 실행 실패: {result.stderr}")
                return None
                
            # CSV 결과 파싱
            return self._parse_ncu_csv(csv_output)
            
        except subprocess.TimeoutExpired:
            print("⏰ NCU 프로파일링 타임아웃")
            return None
        except Exception as e:
            print(f"❌ NCU 프로파일링 오류: {e}")
            return None
    
    def _parse_ncu_csv(self, csv_path):
        """NCU CSV 출력 파싱"""
        
        if not csv_path.exists():
            return None
            
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                results = []
                
                for row in reader:
                    # 주요 메트릭 추출
                    parsed_row = {}
                    for key, value in row.items():
                        if 'throughput' in key.lower() or 'occupancy' in key.lower():
                            try:
                                parsed_row[key] = float(value) if value else 0.0
                            except ValueError:
                                parsed_row[key] = 0.0
                        else:
                            parsed_row[key] = value
                    
                    results.append(parsed_row)
                    
                return results
                
        except Exception as e:
            print(f"❌ CSV 파싱 오류: {e}")
            return None
    
    def _simulate_ncu_output(self, kernel_name):
        """ncu가 없을 때 시뮬레이션 데이터"""
        
        # 커널별로 다른 시뮬레이션 값
        if "conflicts" in kernel_name.lower():
            sm_throughput = 35.2  # Bank conflicts로 인한 낮은 성능
            memory_throughput = 68.5
            occupancy = 87.3
            coalescing = 3.2  # 나쁜 coalescing
        elif "optimized" in kernel_name.lower():
            sm_throughput = 78.1  # 최적화된 성능
            memory_throughput = 89.7
            occupancy = 91.2
            coalescing = 1.1  # 좋은 coalescing
        elif "low_occupancy" in kernel_name.lower():
            sm_throughput = 42.3  # 낮은 occupancy
            memory_throughput = 45.1
            occupancy = 35.8  # 매우 낮음
            coalescing = 1.8
        elif "high_occupancy" in kernel_name.lower():
            sm_throughput = 85.7  # 높은 occupancy
            memory_throughput = 71.2
            occupancy = 89.4  # 높음
            coalescing = 1.2
        else:
            # 기본값
            sm_throughput = 60.0
            memory_throughput = 55.0
            occupancy = 70.0
            coalescing = 2.0
            
        return [{
            "Kernel Name": kernel_name,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": sm_throughput,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": memory_throughput,
            "sm__warps_active.avg.pct_of_peak_sustained_active": occupancy,
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": coalescing,
            "Duration": f"{kernel_name}_duration_ms",
        }]
    
    def analyze_results(self, results):
        """NCU 결과 자동 분석 및 병목 진단"""
        
        if not results:
            return "❌ 분석할 데이터가 없습니다."
            
        analysis = []
        
        for result in results:
            kernel_name = result.get("Kernel Name", "Unknown")
            
            # 메트릭 추출
            sm_throughput = result.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
            memory_throughput = result.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)
            occupancy = result.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)
            coalescing = result.get("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 1)
            
            # 병목 진단
            bottlenecks = []
            
            if sm_throughput < 50:
                bottlenecks.append("🔴 낮은 SM 활용도")
            if memory_throughput < 60:
                bottlenecks.append("🔴 낮은 메모리 대역폭")
            if occupancy < 60:
                bottlenecks.append("🟡 낮은 Occupancy")
            if coalescing > 2.0:
                bottlenecks.append("🔴 나쁜 Memory Coalescing")
                
            # 최적화 제안
            suggestions = []
            
            if occupancy < 60:
                suggestions.append("• Block size 증가 또는 레지스터 사용량 감소")
            if coalescing > 2.0:
                suggestions.append("• Memory access 패턴 개선 (stride 감소)")
            if sm_throughput < 50 and memory_throughput > 70:
                suggestions.append("• Arithmetic intensity 증가 (더 많은 연산)")
            if memory_throughput < 60 and sm_throughput > 70:
                suggestions.append("• Shared memory 활용 또는 data reuse 증가")
                
            analysis.append({
                "kernel": kernel_name,
                "metrics": {
                    "sm_throughput": sm_throughput,
                    "memory_throughput": memory_throughput,
                    "occupancy": occupancy,
                    "coalescing": coalescing
                },
                "bottlenecks": bottlenecks,
                "suggestions": suggestions
            })
            
        return analysis
    
    def cleanup(self):
        """임시 파일 정리"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"⚠️ 임시 파일 정리 실패: {e}")


# =============================================================================
# 실제 프로파일링 실험들
# =============================================================================

def create_profiling_script(kernel_func_name, test_data_size=(1024, 1024)):
    """프로파일링할 Python 스크립트 생성"""
    
    script = f'''
import torch
import sys
import os

# Add current directory to path for module import
sys.path.insert(0, os.getcwd())

# Import the compiled module
try:
    from ncu_profiling_kernels import *
except ImportError:
    print("Module not found, creating placeholder...")
    # Create placeholder for simulation
    def {kernel_func_name}(x):
        return torch.zeros_like(x)

def main():
    device = torch.device('cuda')
    
    # Test data
    if "{kernel_func_name}" == "transpose_conflicts" or "{kernel_func_name}" == "transpose_optimized":
        input_tensor = torch.randn{test_data_size}, device=device, dtype=torch.float32)
    else:
        input_tensor = torch.randn({test_data_size[0] * test_data_size[1]}, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        result = {kernel_func_name}(input_tensor)
    
    # Actual profiling target
    for _ in range(10):
        result = {kernel_func_name}(input_tensor)
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()
'''
    
    return script


def experiment_bank_conflicts():
    """Bank Conflicts 최적화 전후 비교"""
    
    print("\n" + "="*70)
    print("🧪 NCU 실험 1: Bank Conflicts 최적화 효과")
    print("="*70)
    
    profiler = NCUProfiler()
    
    # Bank Conflicts 있는 버전 프로파일링
    script_conflicts = create_profiling_script("transpose_conflicts")
    results_conflicts = profiler.profile_kernel(
        script_conflicts, 
        kernel_name="transpose_with_conflicts"
    )
    
    # 최적화된 버전 프로파일링  
    script_optimized = create_profiling_script("transpose_optimized")
    results_optimized = profiler.profile_kernel(
        script_optimized,
        kernel_name="transpose_optimized"
    )
    
    # 결과 분석
    print("\n📊 Bank Conflicts 실험 결과:")
    
    if results_conflicts and results_optimized:
        analysis_conflicts = profiler.analyze_results(results_conflicts)
        analysis_optimized = profiler.analyze_results(results_optimized)
        
        print("\n🔴 Bank Conflicts 있는 버전:")
        display_analysis(analysis_conflicts[0])
        
        print("\n🟢 최적화된 버전:")
        display_analysis(analysis_optimized[0])
        
        # 성능 향상 계산
        sm_improvement = (analysis_optimized[0]["metrics"]["sm_throughput"] / 
                         analysis_conflicts[0]["metrics"]["sm_throughput"] - 1) * 100
        
        print(f"\n🚀 성능 향상: SM Throughput {sm_improvement:+.1f}%")
        
    profiler.cleanup()


def experiment_occupancy():
    """Occupancy 차이에 따른 성능 영향"""
    
    print("\n" + "="*70)
    print("🧪 NCU 실험 2: Occupancy가 성능에 미치는 영향")
    print("="*70)
    
    profiler = NCUProfiler()
    
    # 낮은 Occupancy 버전
    script_low = create_profiling_script("low_occupancy", (1000000,))
    results_low = profiler.profile_kernel(
        script_low,
        kernel_name="low_occupancy_kernel"
    )
    
    # 높은 Occupancy 버전
    script_high = create_profiling_script("high_occupancy", (1000000,))  
    results_high = profiler.profile_kernel(
        script_high,
        kernel_name="high_occupancy_kernel"
    )
    
    # 결과 분석
    print("\n📊 Occupancy 실험 결과:")
    
    if results_low and results_high:
        analysis_low = profiler.analyze_results(results_low)
        analysis_high = profiler.analyze_results(results_high)
        
        print("\n🔴 낮은 Occupancy (많은 레지스터):")
        display_analysis(analysis_low[0])
        
        print("\n🟢 높은 Occupancy (적은 레지스터):")
        display_analysis(analysis_high[0])
        
    profiler.cleanup()


def display_analysis(analysis):
    """분석 결과 시각적 출력"""
    
    metrics = analysis["metrics"]
    
    print(f"  📈 SM Throughput:      {metrics['sm_throughput']:5.1f}%")
    print(f"  💾 Memory Throughput:  {metrics['memory_throughput']:5.1f}%")
    print(f"  👥 Occupancy:         {metrics['occupancy']:5.1f}%")
    print(f"  🔗 Coalescing Ratio:  {metrics['coalescing']:5.2f}")
    
    if analysis["bottlenecks"]:
        print(f"  🚨 병목 요소:")
        for bottleneck in analysis["bottlenecks"]:
            print(f"     {bottleneck}")
    else:
        print(f"  ✅ 주요 병목 없음")
        
    if analysis["suggestions"]:
        print(f"  💡 최적화 제안:")
        for suggestion in analysis["suggestions"]:
            print(f"     {suggestion}")


def experiment_automated_analysis():
    """여러 커널을 자동으로 프로파일링하고 비교"""
    
    print("\n" + "="*70) 
    print("🧪 NCU 실험 3: 자동 성능 분석 및 비교")
    print("="*70)
    
    profiler = NCUProfiler()
    
    test_kernels = [
        ("transpose_conflicts", "transpose_with_conflicts"),
        ("transpose_optimized", "transpose_optimized"),
        ("low_occupancy", "low_occupancy_kernel"),
        ("high_occupancy", "high_occupancy_kernel"),
    ]
    
    all_results = []
    
    for func_name, kernel_name in test_kernels:
        print(f"\n🔍 프로파일링: {kernel_name}")
        
        if "transpose" in func_name:
            script = create_profiling_script(func_name, (512, 512))
        else:
            script = create_profiling_script(func_name, (1000000,))
            
        results = profiler.profile_kernel(script, kernel_name=kernel_name)
        
        if results:
            analysis = profiler.analyze_results(results)
            all_results.extend(analysis)
            
    # 전체 결과 요약
    print("\n" + "="*70)
    print("📊 전체 커널 성능 요약")
    print("="*70)
    
    print(f"{'Kernel':<25} {'SM%':<8} {'Mem%':<8} {'Occ%':<8} {'Coal':<6} {'Status':<15}")
    print("-" * 70)
    
    for analysis in all_results:
        metrics = analysis["metrics"]
        status = "🟢 Good" if not analysis["bottlenecks"] else f"🔴 {len(analysis['bottlenecks'])} issues"
        
        print(f"{analysis['kernel'][:24]:<25} "
              f"{metrics['sm_throughput']:<8.1f} "
              f"{metrics['memory_throughput']:<8.1f} "
              f"{metrics['occupancy']:<8.1f} "
              f"{metrics['coalescing']:<6.2f} "
              f"{status:<15}")
    
    # 최고/최저 성능 커널
    best_kernel = max(all_results, key=lambda x: x["metrics"]["sm_throughput"])
    worst_kernel = min(all_results, key=lambda x: x["metrics"]["sm_throughput"])
    
    print(f"\n🏆 최고 성능: {best_kernel['kernel']} ({best_kernel['metrics']['sm_throughput']:.1f}%)")
    print(f"🐌 최저 성능: {worst_kernel['kernel']} ({worst_kernel['metrics']['sm_throughput']:.1f}%)")
    
    profiler.cleanup()


def ncu_command_examples():
    """실무에서 사용하는 NCU 명령어 예시들"""
    
    print("\n" + "="*70)
    print("📚 실무 NCU 명령어 예시집")
    print("="*70)
    
    commands = [
        {
            "name": "기본 전체 프로파일링",
            "cmd": "ncu --set full ./my_program",
            "description": "모든 메트릭으로 프로파일링"
        },
        {
            "name": "특정 커널만 프로파일링",
            "cmd": "ncu --kernel-name my_kernel ./my_program",
            "description": "특정 커널만 선택적 분석"
        },
        {
            "name": "CSV 출력으로 자동화",
            "cmd": "ncu --csv --log-file results.csv --set roofline ./my_program",
            "description": "Roofline 메트릭을 CSV로 저장"
        },
        {
            "name": "Warmup 건너뛰기",
            "cmd": "ncu --launch-skip 10 --launch-count 5 ./my_program", 
            "description": "처음 10회 건너뛰고 5회만 측정"
        },
        {
            "name": "메모리 분석 중심",
            "cmd": "ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_throughput.avg.pct_of_peak_sustained_elapsed ./my_program",
            "description": "메모리 관련 메트릭만 수집"
        },
        {
            "name": "Python 프로그램 프로파일링",
            "cmd": "ncu --set roofline python my_script.py",
            "description": "Python CUDA 스크립트 분석"
        }
    ]
    
    for i, cmd_info in enumerate(commands, 1):
        print(f"\n{i}. {cmd_info['name']}")
        print(f"   명령어: {cmd_info['cmd']}")
        print(f"   설명: {cmd_info['description']}")
    
    print(f"\n💡 NCU 사용 팁:")
    print(f"   • --set full은 느리지만 완전한 분석 제공")
    print(f"   • --set roofline은 빠르고 병목 파악에 적합")
    print(f"   • Python 프로그램은 JIT 컴파일 후 warmup 필요")
    print(f"   • CSV 출력으로 CI/CD 파이프라인 통합 가능")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """NCU 프로파일링 실험 실행"""
    
    print("="*70)
    print("🔬 NVIDIA Nsight Compute (NCU) 프로파일링 실습")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA가 사용 불가능합니다.")
        return
        
    print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
    
    # NCU 실험들
    experiment_bank_conflicts()
    experiment_occupancy()
    experiment_automated_analysis()
    ncu_command_examples()
    
    print("\n" + "="*70)
    print("🎓 NCU 프로파일링 실습 완료!")
    print("="*70)
    print("📚 핵심 학습:")
    print("  1. NCU로 정확한 병목 진단 가능")
    print("  2. 최적화 전후 정량적 성능 비교")
    print("  3. SM/Memory Throughput으로 병목 유형 구분")
    print("  4. Occupancy와 Coalescing 메트릭의 실무 활용")
    print("  5. 자동화된 성능 분석 파이프라인 구축")
    print("\n🔍 다음: Nsight Systems로 전체 애플리케이션 타임라인 분석!")


if __name__ == "__main__":
    main()