"""
Week 4 실습 3: NVIDIA Nsight Systems (nsys) 타임라인 분석

목표:
1. 전체 애플리케이션의 CPU-GPU 상호작용 시각화
2. Memory transfer 병목 지점 식별
3. Kernel launch overhead 및 동기화 문제 진단
4. NVTX 마커를 이용한 커스텀 구간 분석

핵심 기능:
- nsys 자동 실행 및 결과 분석
- CPU-GPU 타임라인 해석
- Memory transfer 패턴 분석
- 최적화 포인트 자동 식별
"""
import os
import subprocess
import tempfile
import json
import time
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

# 컴파일러 설정
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"


# =============================================================================
# 다양한 시나리오의 CUDA 커널들
# =============================================================================

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// 시나리오 1: CPU-GPU 동기화 문제가 있는 커널
__global__ void sync_problem_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 간단한 연산
        data[i] = data[i] * 2.0f + 1.0f;
    }
}

// 시나리오 2: 작은 커널들 (Launch overhead 지배적)
__global__ void small_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += 1.0f;  // 매우 간단한 연산
    }
}

// 시나리오 3: 메모리 집약적 커널 (Transfer bound)
__global__ void memory_intensive_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 많은 메모리 접근, 적은 연산
        output[i] = input[i] + input[i+1] + input[i+2] + input[i+3];
    }
}

// 시나리오 4: 계산 집약적 커널 (Compute bound)
__global__ void compute_intensive_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float result = data[i];
        // 많은 연산
        for (int j = 0; j < 1000; j++) {
            result = sinf(result) * cosf(result) + sqrtf(fabsf(result) + 0.001f);
        }
        data[i] = result;
    }
}

// 시나리오 5: Shared Memory를 사용하는 커널
__global__ void shared_memory_kernel(float* data, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // Shared memory로 데이터 로드
    if (i < n) {
        sdata[tid] = data[i];
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Reduction
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0 && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 결과 저장
    if (tid == 0 && i < n) {
        data[blockIdx.x] = sdata[0];
    }
}

// Python 인터페이스 함수들
void sync_problem_cuda(torch::Tensor data) {
    int n = data.size(0);
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    sync_problem_kernel<<<grid_size, block_size>>>(data.data_ptr<float>(), n);
    // 의도적으로 동기화하지 않음 (문제 상황 시뮬레이션)
}

void small_kernel_cuda(torch::Tensor data) {
    int n = data.size(0);
    int block_size = 32;  // 작은 블록
    int grid_size = (n + block_size - 1) / block_size;
    
    small_kernel<<<grid_size, block_size>>>(data.data_ptr<float>(), n);
    cudaDeviceSynchronize();
}

torch::Tensor memory_intensive_cuda(torch::Tensor input) {
    int n = input.size(0) - 3;  // 경계 조건
    auto output = torch::zeros({n}, input.options());
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    memory_intensive_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    cudaDeviceSynchronize();
    
    return output;
}

void compute_intensive_cuda(torch::Tensor data) {
    int n = data.size(0);
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    compute_intensive_kernel<<<grid_size, block_size>>>(data.data_ptr<float>(), n);
    cudaDeviceSynchronize();
}

void shared_memory_cuda(torch::Tensor data) {
    int n = data.size(0);
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    shared_memory_kernel<<<grid_size, block_size>>>(data.data_ptr<float>(), n);
    cudaDeviceSynchronize();
}
"""

cpp_source = """
void sync_problem_cuda(torch::Tensor data);
void small_kernel_cuda(torch::Tensor data);
torch::Tensor memory_intensive_cuda(torch::Tensor input);
void compute_intensive_cuda(torch::Tensor data);
void shared_memory_cuda(torch::Tensor data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sync_problem", &sync_problem_cuda, "Sync Problem Kernel");
    m.def("small_kernel", &small_kernel_cuda, "Small Kernel");
    m.def("memory_intensive", &memory_intensive_cuda, "Memory Intensive Kernel");
    m.def("compute_intensive", &compute_intensive_cuda, "Compute Intensive Kernel");
    m.def("shared_memory", &shared_memory_cuda, "Shared Memory Kernel");
}
"""

print("🔨 JIT 컴파일 중... (NSYS 프로파일링용 커널들)")
module = load_inline(
    name='nsys_timeline_kernels',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    verbose=False
)
print("✅ 컴파일 완료!")


# =============================================================================
# NSYS 프로파일링 자동화 클래스
# =============================================================================

class NSYSProfiler:
    """NVIDIA Nsight Systems 자동 프로파일링 클래스"""
    
    def __init__(self):
        self.nsys_available = self._check_nsys_availability()
        self.temp_dir = tempfile.mkdtemp(prefix="nsys_profile_")
        
    def _check_nsys_availability(self):
        """nsys 명령어 사용 가능성 확인"""
        try:
            result = subprocess.run(['nsys', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ NVIDIA Nsight Systems 발견")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        print("⚠️ nsys 명령어를 찾을 수 없습니다.")
        print("   설치 가이드: https://developer.nvidia.com/nsight-systems")
        print("   시뮬레이션 모드로 계속 진행합니다.")
        return False
    
    def profile_application(self, python_script_content, duration=10, 
                          trace_apis=None, output_name="timeline"):
        """전체 애플리케이션을 프로파일링하고 타임라인 생성"""
        
        if not self.nsys_available:
            return self._simulate_timeline_analysis()
            
        # 임시 Python 스크립트 생성
        script_path = Path(self.temp_dir) / "profile_app.py"
        with open(script_path, 'w') as f:
            f.write(python_script_content)
        
        # 기본 trace API 설정
        if trace_apis is None:
            trace_apis = ["cuda", "nvtx", "osrt", "cublas"]
        
        # nsys 명령어 구성
        output_path = Path(self.temp_dir) / f"{output_name}.nsys-rep"
        nsys_cmd = [
            'nsys', 'profile',
            '--trace', ','.join(trace_apis),
            '--duration', str(duration),
            '--output', str(output_path),
            '--force-overwrite', 'true',
            '--stats', 'true',
            'python', str(script_path)
        ]
        
        try:
            print(f"🔍 NSYS 프로파일링 시작 ({duration}초)")
            result = subprocess.run(nsys_cmd, capture_output=True, text=True, 
                                  timeout=duration + 30, cwd=self.temp_dir)
            
            if result.returncode != 0:
                print(f"❌ NSYS 실행 실패: {result.stderr}")
                return None
                
            # 통계 정보 추출
            stats = self._parse_nsys_stats(result.stdout)
            return {
                "report_path": output_path,
                "stats": stats,
                "stdout": result.stdout
            }
            
        except subprocess.TimeoutExpired:
            print("⏰ NSYS 프로파일링 타임아웃")
            return None
        except Exception as e:
            print(f"❌ NSYS 프로파일링 오류: {e}")
            return None
    
    def _parse_nsys_stats(self, stdout):
        """NSYS 통계 출력 파싱"""
        
        stats = {
            "gpu_utilization": 0.0,
            "kernel_count": 0,
            "memory_transfers": 0,
            "total_gpu_time": 0.0,
            "total_cpu_time": 0.0
        }
        
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # GPU 활용도 추출
            if "GPU activities" in line and "%" in line:
                try:
                    percent_pos = line.find('%')
                    if percent_pos > 0:
                        # 숫자 부분 추출
                        start = percent_pos - 1
                        while start > 0 and (line[start-1].isdigit() or line[start-1] == '.'):
                            start -= 1
                        stats["gpu_utilization"] = float(line[start:percent_pos])
                except ValueError:
                    pass
            
            # 커널 개수 추출
            elif "kernel" in line.lower() and ("launched" in line or "calls" in line):
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    stats["kernel_count"] = numbers[0]
            
            # 메모리 전송 추출
            elif "memcpy" in line.lower() or "transfer" in line.lower():
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    stats["memory_transfers"] = numbers[0]
        
        return stats
    
    def _simulate_timeline_analysis(self):
        """nsys가 없을 때 시뮬레이션 분석"""
        
        return {
            "report_path": None,
            "stats": {
                "gpu_utilization": 75.3,
                "kernel_count": 1250,
                "memory_transfers": 45,
                "total_gpu_time": 8.7,
                "total_cpu_time": 2.1
            },
            "stdout": "Simulated NSYS output - real profiling not available",
            "simulation": True
        }
    
    def analyze_timeline(self, profile_results):
        """타임라인 분석 및 병목 진단"""
        
        if not profile_results:
            return "❌ 분석할 데이터가 없습니다."
            
        stats = profile_results["stats"]
        issues = []
        suggestions = []
        
        # GPU 활용도 분석
        gpu_util = stats.get("gpu_utilization", 0)
        if gpu_util < 50:
            issues.append(f"🔴 낮은 GPU 활용도 ({gpu_util:.1f}%)")
            suggestions.append("• CPU 병목 또는 launch overhead 확인 필요")
            suggestions.append("• Kernel 실행 시간 vs Transfer 시간 비율 분석")
        elif gpu_util < 80:
            issues.append(f"🟡 보통 GPU 활용도 ({gpu_util:.1f}%)")
            suggestions.append("• 비동기 실행 및 스트림 활용 고려")
        else:
            issues.append(f"🟢 높은 GPU 활용도 ({gpu_util:.1f}%)")
        
        # 커널 개수 분석
        kernel_count = stats.get("kernel_count", 0)
        if kernel_count > 1000:
            issues.append(f"🔴 과도한 커널 실행 횟수 ({kernel_count:,}개)")
            suggestions.append("• Kernel fusion으로 실행 횟수 감소")
            suggestions.append("• Loop 내부의 작은 커널들 통합 고려")
        elif kernel_count > 100:
            issues.append(f"🟡 많은 커널 실행 ({kernel_count:,}개)")
            suggestions.append("• Launch overhead 측정 및 최적화")
        
        # 메모리 전송 분석
        memory_transfers = stats.get("memory_transfers", 0)
        if memory_transfers > 50:
            issues.append(f"🔴 빈번한 메모리 전송 ({memory_transfers}회)")
            suggestions.append("• Pinned memory 사용 검토")
            suggestions.append("• 데이터 재사용성 증대")
            suggestions.append("• Unified Memory 활용 고려")
        
        # 전체 분석
        analysis = {
            "summary": {
                "gpu_utilization": gpu_util,
                "kernel_count": kernel_count,
                "memory_transfers": memory_transfers,
                "primary_bottleneck": self._identify_primary_bottleneck(stats)
            },
            "issues": issues,
            "suggestions": suggestions
        }
        
        return analysis
    
    def _identify_primary_bottleneck(self, stats):
        """주요 병목 요소 식별"""
        
        gpu_util = stats.get("gpu_utilization", 0)
        kernel_count = stats.get("kernel_count", 0)
        memory_transfers = stats.get("memory_transfers", 0)
        
        if gpu_util < 50 and memory_transfers > 20:
            return "Memory Transfer Bound"
        elif gpu_util < 60 and kernel_count > 500:
            return "Launch Overhead Bound"
        elif gpu_util > 80:
            return "Compute/Memory Bound (Good)"
        else:
            return "Mixed Bottleneck"
    
    def generate_nvtx_script(self, scenario_name):
        """NVTX 마커를 포함한 테스트 스크립트 생성"""
        
        nvtx_script = f'''
import torch
import sys
import os
import time

# NVTX 지원 확인
try:
    torch.cuda.nvtx.range_push("Application Start")
    nvtx_available = True
except:
    nvtx_available = False
    print("NVTX not available - continuing without markers")

# Add current directory to path for module import
sys.path.insert(0, os.getcwd())

# Import the compiled module
try:
    from nsys_timeline_kernels import *
except ImportError:
    print("Module not found, creating placeholders...")
    def sync_problem(x): return x
    def small_kernel(x): return x  
    def memory_intensive(x): return torch.zeros_like(x)
    def compute_intensive(x): return x
    def shared_memory(x): return x

def run_{scenario_name}_scenario():
    device = torch.device('cuda')
    
    if nvtx_available:
        torch.cuda.nvtx.range_push("Data Preparation")
    
    # 데이터 준비
    data_sizes = [100000, 500000, 1000000]
    test_data = []
    
    for size in data_sizes:
        data = torch.randn(size, device=device, dtype=torch.float32)
        test_data.append(data)
        
    if nvtx_available:
        torch.cuda.nvtx.range_pop()  # Data Preparation
    
    # 시나리오 실행
    if "{scenario_name}" == "cpu_gpu_sync":
        run_sync_scenario(test_data)
    elif "{scenario_name}" == "launch_overhead":
        run_launch_overhead_scenario(test_data)
    elif "{scenario_name}" == "memory_transfer":
        run_memory_transfer_scenario(test_data)
    elif "{scenario_name}" == "compute_intensive":
        run_compute_scenario(test_data)
    elif "{scenario_name}" == "mixed_workload":
        run_mixed_workload_scenario(test_data)
    
    if nvtx_available:
        torch.cuda.nvtx.range_push("Cleanup")
    
    # 정리
    del test_data
    torch.cuda.empty_cache()
    
    if nvtx_available:
        torch.cuda.nvtx.range_pop()  # Cleanup

def run_sync_scenario(test_data):
    """CPU-GPU 동기화 문제 시나리오"""
    if nvtx_available:
        torch.cuda.nvtx.range_push("Sync Problem Scenario")
    
    for i, data in enumerate(test_data):
        if nvtx_available:
            torch.cuda.nvtx.range_push(f"Sync Iteration {{i+1}}")
        
        # 동기화 없이 연속 실행 (문제 상황)
        sync_problem(data)
        
        # CPU 작업 (GPU와 겹침)
        time.sleep(0.001)  # CPU 작업 시뮬레이션
        
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
    
    # 마지막에 강제 동기화
    torch.cuda.synchronize()
    
    if nvtx_available:
        torch.cuda.nvtx.range_pop()

def run_launch_overhead_scenario(test_data):
    """Launch Overhead 문제 시나리오"""
    if nvtx_available:
        torch.cuda.nvtx.range_push("Launch Overhead Scenario")
    
    # 많은 작은 커널 실행
    for data in test_data:
        if nvtx_available:
            torch.cuda.nvtx.range_push("Multiple Small Kernels")
        
        # 100개의 작은 커널 실행
        for _ in range(100):
            small_kernel(data)
        
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
    
    if nvtx_available:
        torch.cuda.nvtx.range_pop()

def run_memory_transfer_scenario(test_data):
    """Memory Transfer 병목 시나리오"""
    if nvtx_available:
        torch.cuda.nvtx.range_push("Memory Transfer Scenario")
    
    for i, data in enumerate(test_data):
        if nvtx_available:
            torch.cuda.nvtx.range_push(f"H2D Transfer {{i+1}}")
        
        # CPU → GPU 전송
        cpu_data = torch.randn_like(data, device='cpu')
        gpu_data = cpu_data.cuda()
        
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"Kernel Execution {{i+1}}")
        
        # 커널 실행
        result = memory_intensive(gpu_data)
        
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"D2H Transfer {{i+1}}")
        
        # GPU → CPU 전송
        result_cpu = result.cpu()
        
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
    
    if nvtx_available:
        torch.cuda.nvtx.range_pop()

def run_compute_scenario(test_data):
    """Compute Intensive 시나리오"""
    if nvtx_available:
        torch.cuda.nvtx.range_push("Compute Intensive Scenario")
    
    for i, data in enumerate(test_data):
        if nvtx_available:
            torch.cuda.nvtx.range_push(f"Heavy Compute {{i+1}}")
        
        compute_intensive(data)
        
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
    
    if nvtx_available:
        torch.cuda.nvtx.range_pop()

def run_mixed_workload_scenario(test_data):
    """Mixed 워크로드 시나리오"""
    if nvtx_available:
        torch.cuda.nvtx.range_push("Mixed Workload Scenario")
    
    for i, data in enumerate(test_data):
        if nvtx_available:
            torch.cuda.nvtx.range_push(f"Mixed Iteration {{i+1}}")
        
        # 메모리 집약적 작업
        if nvtx_available:
            torch.cuda.nvtx.range_push("Memory Phase")
        result1 = memory_intensive(data)
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
        
        # 계산 집약적 작업
        if nvtx_available:
            torch.cuda.nvtx.range_push("Compute Phase")
        compute_intensive(result1)
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
        
        # Shared Memory 작업
        if nvtx_available:
            torch.cuda.nvtx.range_push("Shared Memory Phase")
        shared_memory(data)
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
        
        if nvtx_available:
            torch.cuda.nvtx.range_pop()
    
    if nvtx_available:
        torch.cuda.nvtx.range_pop()

if __name__ == "__main__":
    print(f"Running {{'{scenario_name}'}} scenario...")
    run_{scenario_name}_scenario()
    print("Scenario completed!")
'''
        
        return nvtx_script
    
    def cleanup(self):
        """임시 파일 정리"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"⚠️ 임시 파일 정리 실패: {e}")


# =============================================================================
# 타임라인 분석 실험들
# =============================================================================

def experiment_cpu_gpu_sync():
    """CPU-GPU 동기화 패턴 분석"""
    
    print("\n" + "="*70)
    print("🧪 NSYS 실험 1: CPU-GPU 동기화 패턴 분석")
    print("="*70)
    
    profiler = NSYSProfiler()
    
    # 동기화 문제가 있는 시나리오
    script = profiler.generate_nvtx_script("cpu_gpu_sync")
    results = profiler.profile_application(script, duration=8, output_name="cpu_gpu_sync")
    
    if results:
        analysis = profiler.analyze_timeline(results)
        print("\n📊 CPU-GPU 동기화 분석 결과:")
        display_timeline_analysis(analysis)
        
        print(f"\n💡 타임라인 해석 가이드:")
        print(f"   • CPU Idle 구간: GPU 작업 완료 대기")
        print(f"   • GPU Idle 구간: CPU 작업 또는 동기화 대기") 
        print(f"   • Memory Transfer: 데이터 이동 시간")
        print(f"   • Kernel Execution: 실제 GPU 연산 시간")
    
    profiler.cleanup()


def experiment_launch_overhead():
    """Kernel Launch Overhead 분석"""
    
    print("\n" + "="*70)
    print("🧪 NSYS 실험 2: Kernel Launch Overhead 분석")
    print("="*70)
    
    profiler = NSYSProfiler()
    
    # 많은 작은 커널들의 시나리오
    script = profiler.generate_nvtx_script("launch_overhead")
    results = profiler.profile_application(script, duration=10, output_name="launch_overhead")
    
    if results:
        analysis = profiler.analyze_timeline(results)
        print("\n📊 Launch Overhead 분석 결과:")
        display_timeline_analysis(analysis)
        
        print(f"\n🔍 Launch Overhead 최적화 방법:")
        print(f"   • Kernel Fusion: 여러 커널을 하나로 통합")
        print(f"   • Loop 내부 커널 → Loop를 커널 내부로")
        print(f"   • Larger Block Size: 더 많은 work per launch")
        print(f"   • Streams: 비동기 실행으로 overlap")
    
    profiler.cleanup()


def experiment_memory_transfer():
    """Memory Transfer 병목 분석"""
    
    print("\n" + "="*70)
    print("🧪 NSYS 실험 3: Memory Transfer 병목 분석")
    print("="*70)
    
    profiler = NSYSProfiler()
    
    # 메모리 전송 집약적 시나리오
    script = profiler.generate_nvtx_script("memory_transfer")
    results = profiler.profile_application(script, duration=8, output_name="memory_transfer")
    
    if results:
        analysis = profiler.analyze_timeline(results)
        print("\n📊 Memory Transfer 분석 결과:")
        display_timeline_analysis(analysis)
        
        print(f"\n⚡ Memory Transfer 최적화 방법:")
        print(f"   • Pinned Memory: 더 빠른 PCIe 전송")
        print(f"   • Async Transfers: Compute와 overlap")
        print(f"   • Data Locality: GPU 상주 시간 증가")
        print(f"   • Unified Memory: 자동 migration")
    
    profiler.cleanup()


def experiment_mixed_workload():
    """Mixed 워크로드 타임라인 분석"""
    
    print("\n" + "="*70)
    print("🧪 NSYS 실험 4: Mixed 워크로드 타임라인 분석")
    print("="*70)
    
    profiler = NSYSProfiler()
    
    # 복합 워크로드 시나리오
    script = profiler.generate_nvtx_script("mixed_workload")
    results = profiler.profile_application(script, duration=12, output_name="mixed_workload")
    
    if results:
        analysis = profiler.analyze_timeline(results)
        print("\n📊 Mixed 워크로드 분석 결과:")
        display_timeline_analysis(analysis)
        
        print(f"\n🎯 Mixed 워크로드 최적화 전략:")
        print(f"   • Phase 분석: 각 단계별 병목 식별")
        print(f"   • Pipeline: 단계간 overlap 가능성")
        print(f"   • Resource Balancing: CPU/GPU/Memory 균형")
        print(f"   • Multi-Stream: 병렬 실행 파이프라인")
    
    profiler.cleanup()


def display_timeline_analysis(analysis):
    """타임라인 분석 결과 시각적 출력"""
    
    if isinstance(analysis, str):
        print(analysis)
        return
        
    summary = analysis.get("summary", {})
    
    print(f"📈 타임라인 요약:")
    print(f"   GPU 활용도:     {summary.get('gpu_utilization', 0):6.1f}%")
    print(f"   커널 실행 횟수: {summary.get('kernel_count', 0):6,}개")
    print(f"   메모리 전송:    {summary.get('memory_transfers', 0):6}회")
    print(f"   주요 병목:      {summary.get('primary_bottleneck', 'Unknown')}")
    
    issues = analysis.get("issues", [])
    if issues:
        print(f"\n🚨 발견된 문제점:")
        for issue in issues[:3]:  # 최대 3개만 표시
            print(f"   {issue}")
    
    suggestions = analysis.get("suggestions", [])
    if suggestions:
        print(f"\n💡 최적화 제안:")
        for suggestion in suggestions[:3]:  # 최대 3개만 표시
            print(f"   {suggestion}")


def nsys_command_examples():
    """실무에서 사용하는 NSYS 명령어 예시들"""
    
    print("\n" + "="*70)
    print("📚 실무 NSYS 명령어 예시집")
    print("="*70)
    
    commands = [
        {
            "name": "기본 타임라인 프로파일링",
            "cmd": "nsys profile --trace=cuda,nvtx,osrt python my_script.py",
            "description": "CUDA, NVTX, OS runtime 추적"
        },
        {
            "name": "특정 시간 구간만 캡처",
            "cmd": "nsys profile --duration=30 --delay=5 ./my_program",
            "description": "5초 후 시작해서 30초간 프로파일링"
        },
        {
            "name": "분산 학습 프로파일링",
            "cmd": "nsys profile --trace=cuda,nvtx,mpi --mpi-impl=openmpi torchrun --nproc_per_node=4 train.py",
            "description": "MPI 기반 멀티GPU 학습 분석"
        },
        {
            "name": "메모리 사용량 추적",
            "cmd": "nsys profile --cuda-memory-usage=true --trace=cuda python train.py",
            "description": "GPU 메모리 사용 패턴 분석"
        },
        {
            "name": "NVTX 범위 기반 캡처",
            "cmd": "nsys profile --capture-range=nvtx --capture-range-end=stop python my_script.py",
            "description": "코드 내 NVTX 마커로 캡처 구간 제어"
        },
        {
            "name": "Python + CUDA 상세 추적",
            "cmd": "nsys profile --python-backtrace=cuda --trace=cuda,nvtx,python-tracer python train.py",
            "description": "Python call stack과 CUDA 연결 추적"
        }
    ]
    
    for i, cmd_info in enumerate(commands, 1):
        print(f"\n{i}. {cmd_info['name']}")
        print(f"   명령어: {cmd_info['cmd']}")
        print(f"   설명: {cmd_info['description']}")
    
    print(f"\n💡 NSYS 사용 팁:")
    print(f"   • GUI 분석: nsys-ui timeline.nsys-rep")
    print(f"   • CLI 통계: nsys stats --report gputrace timeline.nsys-rep")
    print(f"   • 긴 실행: --duration으로 캡처 시간 제한")
    print(f"   • NVTX 마커로 관심 구간만 선별적 분석")
    print(f"   • Multi-GPU: --trace=cuda,nvtx,mpi 추가")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """NSYS 타임라인 분석 실험 실행"""
    
    print("="*70)
    print("🕒 NVIDIA Nsight Systems (NSYS) 타임라인 분석")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA가 사용 불가능합니다.")
        return
        
    print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
    
    # NSYS 타임라인 실험들
    experiment_cpu_gpu_sync()
    experiment_launch_overhead()
    experiment_memory_transfer()
    experiment_mixed_workload()
    nsys_command_examples()
    
    print("\n" + "="*70)
    print("🎓 NSYS 타임라인 분석 실습 완료!")
    print("="*70)
    print("📚 핵심 학습:")
    print("  1. 전체 애플리케이션 성능을 타임라인으로 시각화")
    print("  2. CPU-GPU 상호작용 병목 지점 식별")
    print("  3. Memory Transfer vs Kernel Execution 비율 분석")
    print("  4. NVTX 마커로 커스텀 구간 성능 측정")
    print("  5. Launch Overhead와 동기화 문제 진단")
    print("\n🔬 NCU + NSYS = 완벽한 GPU 성능 분석 툴킷!")


if __name__ == "__main__":
    main()