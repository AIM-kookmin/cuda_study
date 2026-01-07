# Week 4: Profiling & Performance Analysis - GPU 병목 분석의 과학

> **핵심 문제**: "측정할 수 없으면 최적화할 수 없다"  
> **목표**: CPU와 GPU의 병목 차이를 이해하고, 전문 도구로 정확한 성능 분석 수행

---

## 🎯 학습 목표

1. **CPU vs GPU 병목의 근본적 차이** 이해
2. **NVIDIA Nsight Compute (ncu)** 활용법 마스터
3. **NVIDIA Nsight Systems (nsys)** 타임라인 분석
4. **Roofline Model**을 통한 성능 한계 분석
5. **실무 병목 진단** 프로세스 확립

---

## 🏗️ CPU vs GPU: 병목의 DNA가 다르다

### CPU 병목의 특징

```
🧠 CPU: Sequential Excellence
┌─────────────────────────────────────┐
│  Core 1    Core 2    Core 3    ... │  ← 소수 강력한 코어
│ [Cache]   [Cache]   [Cache]        │  ← 거대한 캐시
│   │         │         │            │
│ [Fetch] → [Decode] → [Execute]     │  ← 복잡한 파이프라인
│   │         │         │            │
│ [Branch Predictor] [Out-of-Order]  │  ← 고급 최적화
└─────────────────────────────────────┘
```

**CPU 병목 유형:**
1. **Cache Miss** (L1 → L2 → L3 → RAM: 1 → 3 → 12 → 300 cycles)
2. **Branch Misprediction** (파이프라인 flush로 10-20 cycle 손실)
3. **Memory Bandwidth** (DDR4: ~50GB/s, DDR5: ~100GB/s)
4. **Instruction Dependencies** (데이터 의존성으로 병렬 처리 제한)
5. **Context Switching** (프로세스/스레드 전환 오버헤드)

### GPU 병목의 특징

```
⚡ GPU: Parallel Throughput Machine
┌─────────────────────────────────────────────┐
│ SM0   SM1   SM2   SM3   ...   SM107        │  ← 수천 개 코어
│[32c] [32c] [32c] [32c]      [32c]         │  ← 단순한 코어들
│  │     │     │     │          │           │
│[Shared Memory] [Shared Memory] ...         │  ← 프로그래머 제어
│  │     │     │     │          │           │
│ [Warp Scheduler] [Warp Scheduler] ...      │  ← 대량 스레드 관리
└─────────────────────────────────────────────┘
        │
    Global Memory (HBM: ~1TB/s)
```

**GPU 병목 유형:**
1. **Memory Bandwidth** (HBM2: ~900GB/s, HBM3: ~3TB/s - 하지만 여전히 부족!)
2. **Occupancy** (SM당 활성 스레드 수 부족)
3. **Warp Divergence** (조건문으로 인한 SIMD 효율성 저하)  
4. **Memory Coalescing** (비효율적 메모리 접근 패턴)
5. **Shared Memory Bank Conflicts** (동시 접근 충돌)
6. **Register Pressure** (레지스터 부족으로 스필링 발생)

---

## 📊 병목 분석 방법론의 차이

### CPU 성능 분석 (Sequential Mindset)

```bash
# CPU 프로파일링 예시 (perf)
perf stat -e cache-misses,branch-misses,instructions ./cpu_program
#  Performance counter stats:
#    45,234,567 cache-misses    # 12.3% miss rate
#     2,345,678 branch-misses  # 5.2% miss rate  
#   987,654,321 instructions   # 0.78 IPC (Instructions Per Cycle)
```

**CPU 최적화 접근:**
- Cache-friendly 데이터 구조
- Branch prediction 최적화  
- Loop unrolling & vectorization
- 메모리 프리페칭

### GPU 성능 분석 (Parallel Mindset)

```bash
# GPU 프로파일링 예시 (ncu)
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./gpu_program
#   sm__throughput.avg.pct_of_peak_sustained_elapsed: 15.2%
#   → SM 활용도가 15%만! 병목 발견!
```

**GPU 최적화 접근:**
- Memory bandwidth 최적화
- Occupancy 극대화
- Coalescing 패턴 개선
- Shared Memory 활용

---

## 🔬 Roofline Model: GPU 성능의 이론적 한계

### Roofline Model 이란?

**정의**: 주어진 하드웨어에서 달성 가능한 **이론적 최대 성능**을 시각화하는 모델

```
Performance (GFLOPS)
     │
     │     Compute Bound Region
     │    /│
     │   / │ ← Peak FLOPS (예: 19 TFLOPS)
     │  /  │
     │ /   │ Memory Bound Region  
     │/____│________________________
     │     │                        
     │     │                     Arithmetic Intensity
     └─────┼─────────────────────────► (FLOP/Byte)
           │
      Ridge Point (균형점)
```

### 핵심 개념들

#### 1. **Arithmetic Intensity (연산 집약도)**

```cpp
// 예시: Matrix Multiplication C = A × B
// 연산량: N³ FLOPs (N×N×N 곱셈-덧셈)
// 메모리: 3N² Bytes (A, B, C 각각 N² 원소)
// Arithmetic Intensity = N³ / (3N²) = N/3 FLOP/Byte

float arithmetic_intensity = total_flops / total_bytes_accessed;
```

#### 2. **Ridge Point (균형점)**

```cpp
// Ridge Point = Peak Memory Bandwidth / Peak Compute Throughput
// RTX 4090 예시:
float peak_bandwidth = 1008e9;    // 1008 GB/s (HBM3)
float peak_compute = 83e12;       // 83 TFLOPS (FP32)
float ridge_point = peak_compute / peak_bandwidth; 
// = 82.3 FLOP/Byte
```

**의미**: 
- **AI < Ridge Point**: Memory Bound (메모리가 병목)
- **AI > Ridge Point**: Compute Bound (연산이 병목)

### 실제 GPU별 Roofline 특성

| GPU 모델 | Peak FLOPS | Peak BW | Ridge Point | 특징 |
|----------|------------|---------|-------------|------|
| **RTX 4090** | 83 TFLOPS | 1008 GB/s | 82 FLOP/B | 균형잡힌 설계 |
| **H100** | 67 TFLOPS | 3350 GB/s | 20 FLOP/B | 메모리 중심 (AI 워크로드) |
| **RTX 3060** | 13 TFLOPS | 360 GB/s | 36 FLOP/B | 게이밍 중심 |
| **V100** | 15.7 TFLOPS | 900 GB/s | 17 FLOP/B | 데이터센터 최적화 |

---

## 🛠️ NVIDIA Nsight Compute (ncu): 커널 분석의 현미경

### ncu의 핵심 기능

**정의**: 개별 CUDA 커널의 **마이크로벤치마킹** 도구

```bash
# 기본 사용법
ncu --set full ./my_cuda_program

# 특정 메트릭만 측정
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    ./my_program

# 특정 커널만 프로파일링
ncu --kernel-name "my_kernel" ./my_program

# CSV 출력으로 자동 분석
ncu --csv --log-file results.csv --set full ./my_program
```

### 핵심 메트릭 완전 가이드

#### 1. **SM Throughput (컴퓨트 활용도)**

```
sm__throughput.avg.pct_of_peak_sustained_elapsed
```

- **의미**: SM(Streaming Multiprocessor) 활용률
- **범위**: 0-100%
- **목표**: 80% 이상
- **낮으면**: Occupancy 문제 또는 메모리 병목

#### 2. **Memory Throughput (메모리 대역폭)**

```
# Global Memory Read 대역폭
l1tex__t_throughput_gld_request_throughput.avg.pct_of_peak_sustained_elapsed

# Shared Memory 대역폭  
l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed
```

#### 3. **Occupancy (점유율)**

```
sm__warps_active.avg.pct_of_peak_sustained_active
```

- **의미**: 이론적 최대 대비 활성 Warp 비율
- **목표**: 60% 이상 (항상 100%일 필요는 없음)

#### 4. **Memory Coalescing 효율성**

```
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
```

- **이상값**: 1.0 (완벽한 coalescing)
- **실제**: 1.0-2.0 (양호), >4.0 (문제)

### ncu 출력 해석 실무 가이드

```bash
# 실제 ncu 출력 예시
==PROF== Report file moved to "report.ncu-rep"
==PROF== Report:
  my_kernel<<<(512,1,1),(256,1,1)>>> (2048, 0x7f8b2c000000, 0x7f8b2c200000)
    Section: GPU Speed Of Light Throughput
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      DRAM Frequency           cycle/second                 877.50
      SM Frequency             cycle/second                1410.00
      Elapsed Cycles           cycle                        4234567
      Memory Throughput        %                           45.2
      DRAM Throughput          %                           67.8
      Duration                 usecond                     3004.52
      L1/TEX Cache Throughput  %                           89.1
      L2 Cache Throughput      %                           23.4
      SM Active Cycles         cycle                       4567890
      Compute (SM) Throughput  %                           12.3  ← 🚨 문제!
```

**해석**:
- **Memory Throughput 45.2%**: 메모리 활용은 중간 수준
- **Compute Throughput 12.3%**: SM 활용도가 매우 낮음 → **컴퓨트 병목 아님!**  
- **결론**: 메모리 바운드 워크로드, Occupancy 개선 필요

---

## 🕘 NVIDIA Nsight Systems (nsys): 시간 여행자의 도구

### nsys의 핵심 기능

**정의**: **전체 애플리케이션**의 CPU-GPU 상호작용을 타임라인으로 분석

```bash
# 기본 프로파일링
nsys profile --trace cuda,nvtx ./my_program

# 상세 GPU 메모리 전송 추적
nsys profile --trace cuda,nvtx,cublas,cudnn \
    --cuda-memory-usage true \
    ./my_program

# Python 프로그램 프로파일링  
nsys profile --trace cuda,nvtx,osrt,python-tracer \
    python my_script.py

# 특정 시간 구간만 캡처
nsys profile --duration 30 --delay 5 ./my_program
```

### nsys 타임라인 분석 포인트

#### 1. **CPU-GPU 동기화 병목**

```
타임라인 예시:
CPU Thread 1: ████████░░░░░░░░████████░░░░░░░░
                     ↑ Idle        ↑ Idle
GPU Stream 0:   ░░░░████████░░░░░░░░████████
                   ↑ Kernel      ↑ Kernel

문제: CPU가 GPU 완료를 기다리며 유휴 상태
해결: 비동기 실행, 멀티 스트림 활용
```

#### 2. **Memory Transfer 병목**

```
GPU Memory: 
H2D Transfer: ████████████████████████ (80ms)
Kernel Exec:          ████ (5ms) 
D2H Transfer:              ████████████ (40ms)

문제: 커널 실행 시간 < 메모리 전송 시간
해결: 데이터 재사용, Pinned Memory, Unified Memory
```

#### 3. **Kernel Launch Overhead**

```
Timeline:
Small Kernels: █░█░█░█░█░█░█░█░ (launch overhead 지배)
              ↑ ↑ ↑ Launch gaps

해결: Kernel Fusion, Cooperative Groups
```

### nsys + Python 통합 분석

```python
# NVTX Markers로 코드 구간 표시
import torch
torch.cuda.nvtx.range_push("Data Loading")
data = load_data()
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("Model Forward")  
output = model(data)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("Loss Computation")
loss = criterion(output, target)
torch.cuda.nvtx.range_pop()

# nsys에서 구간별 시간 분석 가능!
```

---

## 🧮 실무 병목 진단 프로세스

### Phase 1: 거시적 분석 (nsys)

```bash
# Step 1: 전체 타임라인 캡처
nsys profile --trace cuda,nvtx,python-tracer python train.py

# Step 2: 시간 소비 TOP 10 확인
# GUI에서 또는 CLI로:
nsys stats --report gputrace report.qdrep
```

**판단 기준**:
- **GPU Utilization < 80%**: CPU 병목 또는 Launch Overhead
- **Memory Transfer > Kernel Time**: Memory Bound  
- **Irregular Patterns**: 동기화 문제

### Phase 2: 미시적 분석 (ncu)

```bash
# Step 1: 가장 시간 소비가 큰 커널 식별
ncu --set full --kernel-name "expensive_kernel" ./program

# Step 2: 해당 커널의 병목 유형 판단
# Memory Bound vs Compute Bound vs Launch Bound
```

**진단 결정 트리**:
```
SM Throughput < 60%?
├─ Yes → Occupancy 문제
│  ├─ Register 사용량 확인
│  └─ Block Size 최적화
└─ No → Memory Throughput < 80%? 
   ├─ Yes → Memory Bound
   │  ├─ Coalescing 확인
   │  └─ Cache Hit Rate 확인
   └─ No → Compute Bound (이상적!)
```

### Phase 3: 최적화 및 검증

```bash
# Before
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./program_before

# 최적화 적용 (예: Shared Memory Tiling)

# After  
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./program_after

# 성능 향상 정량화
```

---

## 📈 성능 메트릭의 실무 해석

### GPU 메트릭 해석 가이드

| 메트릭 | 양호 | 경고 | 위험 | 의미 & 해결책 |
|--------|------|------|------|---------------|
| **SM Throughput** | >80% | 50-80% | <50% | 컴퓨트 활용률. Occupancy 증가 필요 |
| **Memory Throughput** | >80% | 50-80% | <50% | 메모리 대역폭 활용. Coalescing 개선 |  
| **Occupancy** | >60% | 30-60% | <30% | 활성 Warp 비율. Block Size/Register 최적화 |
| **Coalescing Efficiency** | ~1.0 | 1.0-2.0 | >4.0 | 메모리 접근 효율. 접근 패턴 개선 |
| **Cache Hit Rate** | >90% | 80-90% | <80% | 캐시 효율성. 데이터 지역성 향상 |

### Roofline 분석 실무

```python
# 성능 분석 스크립트 예시
def analyze_kernel_performance(flops, bytes_accessed, runtime_ms):
    # Arithmetic Intensity 계산
    ai = flops / bytes_accessed
    
    # 실제 성능 계산  
    actual_gflops = flops / (runtime_ms * 1e-3) / 1e9
    actual_bw = bytes_accessed / (runtime_ms * 1e-3) / 1e9
    
    # 이론적 한계와 비교
    peak_gflops = 83000  # RTX 4090
    peak_bw = 1008       # GB/s
    ridge_point = peak_gflops / peak_bw
    
    if ai < ridge_point:
        print(f"Memory Bound: AI={ai:.1f} < Ridge={ridge_point:.1f}")
        print(f"BW Efficiency: {actual_bw/peak_bw*100:.1f}%")
    else:
        print(f"Compute Bound: AI={ai:.1f} > Ridge={ridge_point:.1f}")  
        print(f"Compute Efficiency: {actual_gflops/peak_gflops*100:.1f}%")
```

---

## 🚀 최신 프로파일링 기법 (2024-2025)

### 1. **Nsight Compute CLI 자동화**

```bash
#!/bin/bash
# 자동 성능 회귀 테스트
BASELINE=$(ncu --csv --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
           ./baseline_kernel | tail -1 | cut -d',' -f2)
OPTIMIZED=$(ncu --csv --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
            ./optimized_kernel | tail -1 | cut -d',' -f2)

IMPROVEMENT=$(echo "scale=2; $OPTIMIZED / $BASELINE * 100" | bc)
echo "Performance improvement: ${IMPROVEMENT}%"
```

### 2. **PyTorch Profiler 통합**

```python
# PyTorch 2.0+ 통합 프로파일링
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        output = model(input)
    
    with record_function("loss_computation"):
        loss = criterion(output, target)

# Chrome Trace 형식으로 export (nsys와 호환)
prof.export_chrome_trace("trace.json")
```

### 3. **MLOps 통합 모니터링**

```python
# Weights & Biases GPU 메트릭 통합
import wandb

def log_gpu_metrics(kernel_name, ncu_output):
    metrics = parse_ncu_output(ncu_output)
    wandb.log({
        f"{kernel_name}/sm_throughput": metrics['sm_throughput'],
        f"{kernel_name}/memory_throughput": metrics['memory_throughput'],
        f"{kernel_name}/occupancy": metrics['occupancy']
    })
```

---

## 🎯 핵심 포인트 정리

1. **CPU vs GPU 병목은 근본적으로 다르다**
   - CPU: Cache, Branch, Dependencies 중심
   - GPU: Memory Bandwidth, Occupancy, Coalescing 중심

2. **Roofline Model로 이론적 한계를 파악하라**
   - Arithmetic Intensity가 성능 특성을 결정
   - Ridge Point를 기준으로 최적화 방향 설정

3. **nsys로 숲을 보고, ncu로 나무를 봐라**
   - nsys: 전체 타임라인, CPU-GPU 상호작용
   - ncu: 개별 커널 마이크로벤치마킹

4. **메트릭 해석이 최적화의 시작이다**
   - 80% 룰: SM/Memory Throughput > 80% 목표
   - Occupancy는 60% 이상이면 충분한 경우가 많음

5. **자동화된 성능 모니터링을 구축하라**
   - CI/CD 파이프라인에 성능 회귀 테스트 통합
   - MLOps 도구와 GPU 메트릭 연동

---

## 🧪 이번 주 실습 및 과제

### 실습 파일
1. **`lecture4_bottleneck_basics.py`**: CPU vs GPU 병목 시나리오 체험
2. **`lecture4_ncu_profiling.py`**: ncu 메트릭 수집 자동화  
3. **`lecture4_nsys_timeline.py`**: nsys 타임라인 분석
4. **`lecture4_roofline_analysis.py`**: Roofline Model 실습

### 과제: Performance Detective 🕵️
- 다양한 병목 시나리오가 포함된 "버그가 있는" 커널들을 제공
- ncu/nsys를 사용해 병목을 찾아내고 최적화
- Before/After 성능 비교 리포트 작성

다음 주(Week 5)에서는 **Advanced Memory Patterns**을 배웁니다! 💪

---

*"Profiling은 추측을 사실로 바꾸는 마법입니다!" 🔍✨*