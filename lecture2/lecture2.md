# Week 2: GPU Architecture & Programming Model

생성일: 2025년 12월 22일
태그: CUDA, Architecture, Warp, Grid-Stride

---

## 📌 이번 주 목표

Week 1에서는 **"일단 돌려보자(Software)"**였다면, 이번 주는 **"하드웨어에 맞춰 돌리자(Hardware)"**입니다.

1. **Architecture**: 내 코드가 GPU의 어느 부분에서 실행되는지 이해한다.
2. **Warp**: 스레드 32개가 한 몸처럼 움직이는 **Warp(워프)** 개념을 이해한다.
3. **Scalability**: 데이터가 아무리 많아도 처리할 수 있는 **Grid-Stride Loop** 패턴을 익힌다.

### 학습 자료
- 📕 **[PMPP]** Ch 1-3 (Data Parallelism, Scalability)
- 📺 **[GPU-MODE] Lecture 2:** PMPP 1-3장 요약
- 📺 **[GPU-MODE] Lecture 4:** Compute & Memory Architecture

---

## 🔄 Week 1 Recap (간략 복습)

지난 주에 배운 핵심 내용을 복습합니다.

### 1. JIT Compilation Pipeline
```
Python 문자열(C++ 코드) → load_inline() → nvcc 컴파일 → .so 파일 → Python에서 import
```

### 2. Host vs Device
- **Host**: CPU + RAM. 전체 흐름 제어, 데이터 준비, 커널 실행 명령.
- **Device**: GPU + VRAM. 실제 병렬 연산 수행.

### 3. Indexing 공식 (1D)
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
- `blockIdx.x`: 몇 번째 블록?
- `blockDim.x`: 블록당 스레드 수
- `threadIdx.x`: 블록 내에서 몇 번째 스레드?

### 4. Boundary Check의 중요성
스레드 수가 데이터 수보다 많을 수 있으므로 항상 범위를 체크해야 합니다.
```cpp
if (i < n) {
    // 작업 수행
}
```

---

## 🔍 왜 GPU 아키텍처를 알아야 하는가?

Week 1에서는 "스레드를 많이 만들면 알아서 빨라지겠지"라고 생각했을 수 있습니다.
하지만 이는 **절반만 맞는 말**입니다.

### 예시: 같은 작업, 다른 성능

```python
# 설정 A: Block 1개, Thread 1024개
kernel<<<1, 1024>>>(...)

# 설정 B: Block 4개, Thread 256개
kernel<<<4, 256>>>(...)
```

두 설정 모두 총 스레드 수는 1024개로 동일합니다.
하지만 **설정 B가 훨씬 빠를 수 있습니다.** 왜일까요?

→ **하드웨어 구조**를 이해해야만 답할 수 있습니다.

### GPU 최적화의 핵심 질문들

| 질문 | 관련 개념 |
|------|-----------|
| Block 크기는 몇으로 해야 하지? | SM, Occupancy |
| 왜 스레드 32개 단위로 생각해야 하지? | Warp |
| if-else를 쓰면 왜 느려지지? | Warp Divergence |
| 데이터가 10억 개인데 스레드를 10억 개 못 만드는데? | Grid-Stride Loop |

이 질문들에 답하려면 **GPU의 내부 구조**를 알아야 합니다.

---

## 🏗️ GPU 아키텍처 개요

### CPU vs GPU: 설계 철학의 차이

| 특성 | CPU | GPU |
|------|-----|-----|
| **설계 목표** | Latency (지연 시간 최소화) | Throughput (처리량 최대화) |
| **코어 수** | 수 개 ~ 수십 개 | 수천 개 |
| **코어 복잡도** | 매우 복잡 (OoO, Branch Prediction) | 단순 (In-Order) |
| **캐시 크기** | 크다 (L1/L2/L3) | 작다 |
| **비유** | 페라리 (빠르지만 1명만) | 버스 (느리지만 100명 태움) |

### GPU 하드웨어 구조 (NVIDIA 기준)

```
┌─────────────────────────────────────────────────────────────┐
│                         GPU                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │   SM 0  │ │   SM 1  │ │   SM 2  │ │  SM N   │  ...      │
│  │┌───────┐│ │┌───────┐│ │┌───────┐│ │┌───────┐│           │
│  ││ Warp  ││ ││ Warp  ││ ││ Warp  ││ ││ Warp  ││           │
│  ││Scheduler│ ││Scheduler│ ││Scheduler│ ││Scheduler│           │
│  │└───────┘│ │└───────┘│ │└───────┘│ │└───────┘│           │
│  │┌───────┐│ │┌───────┐│ │┌───────┐│ │┌───────┐│           │
│  ││ CUDA  ││ ││ CUDA  ││ ││ CUDA  ││ ││ CUDA  ││           │
│  ││ Cores ││ ││ Cores ││ ││ Cores ││ ││ Cores ││           │
│  │└───────┘│ │└───────┘│ │└───────┘│ │└───────┘│           │
│  │┌───────┐│ │┌───────┐│ │┌───────┐│ │┌───────┐│           │
│  ││Shared ││ ││Shared ││ ││Shared ││ ││Shared ││           │
│  ││Memory ││ ││Memory ││ ││Memory ││ ││Memory ││           │
│  │└───────┘│ │└───────┘│ │└───────┘│ │└───────┘│           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Global Memory (VRAM)                 │  │
│  │                     (수 GB ~ 수십 GB)                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 구성요소 설명

#### 1. SM (Streaming Multiprocessor)
- GPU의 **기본 연산 유닛**. 여러 개의 SM이 GPU 안에 있습니다.
- 예: RTX 3060 → 28개 SM, RTX 4090 → 128개 SM
- 각 SM은 독립적으로 Block을 실행합니다.

#### 2. CUDA Core
- SM 안에 있는 **산술 연산 유닛** (ALU).
- 한 SM에 수십~수백 개의 CUDA Core가 있습니다.
- 정수/부동소수점 연산을 수행합니다.

#### 3. Warp Scheduler
- SM 안에서 **Warp 단위로 명령어를 발급**하는 스케줄러.
- 한 SM에 보통 4개의 Warp Scheduler가 있습니다.

#### 4. Shared Memory
- SM 내의 **고속 메모리** (L1 캐시와 공유).
- 같은 Block 내의 스레드들이 공유할 수 있습니다.
- Week 3에서 자세히 다룹니다.

#### 5. Global Memory (VRAM)
- GPU의 **메인 메모리**. 모든 스레드가 접근 가능.
- 용량은 크지만 (수 GB) 속도는 느립니다.
- 메모리 접근 패턴이 성능에 큰 영향을 미칩니다.

---

## 🔗 Software → Hardware 매핑

CUDA 프로그래밍 모델(Grid-Block-Thread)이 실제 하드웨어에 어떻게 매핑되는지 이해해야 합니다.

### 매핑 테이블

| Software (CUDA) | Hardware (GPU) | 설명 |
|-----------------|----------------|------|
| **Grid** | 전체 GPU | 하나의 커널 실행 단위 |
| **Block** | SM | 하나의 SM에서 실행됨 |
| **Thread** | CUDA Core에서 실행 | 실제 연산 수행 |
| **Warp (32 threads)** | SIMT 실행 단위 | 동시에 같은 명령어 실행 |

### 핵심 규칙

1. **한 Block은 하나의 SM에서만 실행됩니다.**
   - Block은 절대로 여러 SM에 걸쳐 실행되지 않습니다.
   - 따라서 Block 내 스레드들은 Shared Memory를 공유할 수 있습니다.

2. **한 SM은 여러 Block을 동시에 실행할 수 있습니다.**
   - SM의 리소스(레지스터, Shared Memory)가 허용하는 한.

3. **Block의 스레드들은 32개씩 묶여서 Warp로 실행됩니다.**
   - Block 크기가 256이면 → 8개의 Warp로 나뉩니다.

```
Block (256 threads)
├── Warp 0: Thread 0-31
├── Warp 1: Thread 32-63
├── Warp 2: Thread 64-95
├── Warp 3: Thread 96-127
├── Warp 4: Thread 128-159
├── Warp 5: Thread 160-191
├── Warp 6: Thread 192-223
└── Warp 7: Thread 224-255
```

---

## 🌀 Warp (워프): GPU 실행의 핵심 단위

### Warp란 무엇인가?

**Warp**는 **32개의 스레드가 완전히 동기화되어 같은 명령어를 실행하는 단위**입니다.

- NVIDIA GPU에서 Warp 크기는 항상 **32**입니다. (AMD는 64)
- 같은 Warp 내의 스레드들은 **동시에 같은 명령어**를 실행합니다.
- 이를 **SIMT (Single Instruction, Multiple Threads)** 실행 모델이라고 합니다.

### SIMT vs SIMD

| 개념 | 설명 |
|------|------|
| **SIMD** (CPU) | 하나의 명령어가 여러 데이터를 처리 (예: AVX) |
| **SIMT** (GPU) | 여러 스레드가 같은 명령어를 각자의 데이터에 적용 |

SIMT의 핵심은 **프로그래머 입장에서는 스레드가 독립적으로 보이지만, 하드웨어는 32개씩 묶어서 처리**한다는 것입니다.

### Warp 실행 예시

```cpp
// 모든 스레드가 이 코드를 "각자" 실행하는 것처럼 보이지만...
int i = threadIdx.x;
float result = a[i] + b[i];
c[i] = result;
```

실제 하드웨어에서는:
```
시점 T0: Warp 0의 32개 스레드가 동시에 "int i = threadIdx.x" 실행
시점 T1: Warp 0의 32개 스레드가 동시에 "a[i] 로드" 실행
시점 T2: Warp 0의 32개 스레드가 동시에 "b[i] 로드" 실행
시점 T3: Warp 0의 32개 스레드가 동시에 "덧셈" 실행
시점 T4: Warp 0의 32개 스레드가 동시에 "c[i] 저장" 실행
```

### ⚠️ Warp Divergence (워프 발산)

**문제 상황**: Warp 내의 스레드들이 서로 다른 분기(if-else)를 타야 할 때.

```cpp
// Warp Divergence 발생!
if (threadIdx.x % 2 == 0) {
    // 짝수 스레드: 경로 A
    do_something_A();
} else {
    // 홀수 스레드: 경로 B
    do_something_B();
}
```

**왜 문제인가?**
- Warp의 모든 스레드는 **같은 명령어**를 실행해야 합니다.
- 하지만 위 코드에서 절반은 A로, 절반은 B로 가야 합니다.
- 해결책: **직렬화 (Serialization)**
  1. 먼저 짝수 스레드가 A를 실행 (홀수는 대기)
  2. 그다음 홀수 스레드가 B를 실행 (짝수는 대기)

**결과**: 이론적으로 2배 느려집니다!

```
┌─────────────────────────────────────────┐
│ Warp 실행 타임라인 (Divergence 발생)     │
├─────────────────────────────────────────┤
│ T0: [Thread 0,2,4...30 실행 A]          │
│     [Thread 1,3,5...31 대기 ⏸️]          │
│                                          │
│ T1: [Thread 0,2,4...30 대기 ⏸️]          │
│     [Thread 1,3,5...31 실행 B]          │
└─────────────────────────────────────────┘
```

### Warp Divergence 최소화 전략

1. **같은 Warp 내 스레드는 같은 경로를 타도록 설계**
   ```cpp
   // Bad: Warp 내에서 divergence
   if (threadIdx.x % 2 == 0) { ... }
   
   // Better: Warp 단위로 분기
   if (threadIdx.x / 32 % 2 == 0) { ... }
   ```

2. **분기 대신 조건부 연산 사용**
   ```cpp
   // Bad: 분기 발생
   if (condition) x = a; else x = b;
   
   // Better: 분기 없음
   x = condition ? a : b;  // 또는 비트 연산 활용
   ```

### Warp 관련 핵심 숫자

| 항목 | 값 | 의미 |
|------|-----|------|
| Warp Size | 32 | 동시 실행 스레드 수 |
| Block당 최대 Warp | 32 | 1024 threads / 32 |
| SM당 최대 Warp | 48~64 | 아키텍처마다 다름 |

---

## 📊 Occupancy (점유율)

### Occupancy란?

**Occupancy** = (SM에서 실제 활성화된 Warp 수) / (SM이 지원하는 최대 Warp 수)

예를 들어, SM이 최대 64개 Warp를 지원하는데 32개만 활성화되었다면:
- Occupancy = 32 / 64 = **50%**

### 왜 Occupancy가 중요한가?

GPU는 **메모리 지연 시간을 숨기기 위해** 여러 Warp를 번갈아 실행합니다.

```
Warp 0: [실행] [메모리 대기...] [실행]
Warp 1:        [실행] [메모리 대기...] [실행]
Warp 2:               [실행] [메모리 대기...] [실행]
...
```

Occupancy가 낮으면 → 대기 중에 실행할 Warp가 없음 → **GPU가 놀게 됨**.

### Occupancy에 영향을 미치는 요소

| 요소 | 설명 |
|------|------|
| **Block 크기** | Block이 너무 작으면 SM 활용도 저하 |
| **레지스터 사용량** | 커널이 레지스터를 많이 쓰면 동시 실행 Warp 감소 |
| **Shared Memory 사용량** | Shared Memory를 많이 쓰면 동시 실행 Block 감소 |

### 권장 Block 크기

- **최소**: 128 (4 Warps)
- **권장**: 256 (8 Warps)
- **최대**: 1024 (32 Warps)

> 💡 Block 크기는 32의 배수로 설정하세요. 그래야 Warp가 낭비 없이 채워집니다.

---

## 🔁 Grid-Stride Loop: 확장 가능한 커널 패턴

### 문제 상황

Week 1에서 배운 패턴:
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    c[i] = a[i] + b[i];
}
```

이 패턴은 **데이터 1개당 스레드 1개**를 가정합니다.

**문제**: 데이터가 10억 개인데, GPU 최대 스레드 수는 수백만 개입니다!
- 최대 Grid 크기 제한 (예: 2^31 - 1 blocks)
- 최대 Block 크기 제한 (1024 threads)
- 실제로는 SM 수 × 동시 실행 Block 수에 의해 제한됨

### 해결책: Grid-Stride Loop

**아이디어**: 스레드가 자기 몫을 처리한 후, **stride만큼 건너뛰어서 다음 데이터도 처리**합니다.

```cpp
__global__ void vector_add(float* a, float* b, float* c, int n) {
    // 전체 스레드 수 = gridDim.x * blockDim.x
    int stride = blockDim.x * gridDim.x;
    
    // 시작점: 이 스레드의 글로벌 ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // stride씩 건너뛰면서 끝까지 처리
    for (int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
```

### 동작 원리 시각화

데이터 12개, 스레드 4개 (Block 1개 × Thread 4개)인 경우:

```
데이터:    [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11]
            ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓    ↓    ↓
Thread 0:  [0]             [4]             [8]
Thread 1:      [1]             [5]             [9]
Thread 2:          [2]             [6]             [10]
Thread 3:              [3]             [7]              [11]

stride = 4 (전체 스레드 수)
```

각 스레드가 3개씩 처리합니다:
- Thread 0: index 0, 4, 8
- Thread 1: index 1, 5, 9
- Thread 2: index 2, 6, 10
- Thread 3: index 3, 7, 11

### Monolithic vs Grid-Stride 비교

| 방식 | Monolithic (Week 1) | Grid-Stride (Week 2) |
|------|---------------------|----------------------|
| **코드** | `if (i < n) {...}` | `for (i = idx; i < n; i += stride)` |
| **스레드당 작업** | 최대 1개 | 여러 개 |
| **Grid 크기** | `cdiv(n, block_size)` | 고정값 (예: SM 수 × 4) |
| **데이터 크기 제한** | Grid 최대 크기에 제한 | **무제한** |
| **유연성** | 낮음 | 높음 |

### Grid-Stride Loop의 장점

1. **확장성 (Scalability)**
   - 데이터가 아무리 많아도 처리 가능.
   - Grid 크기를 고정해도 됨.

2. **디버깅 용이성**
   - Grid 크기를 1로 줄여서 순차 실행 테스트 가능.
   
   ```cpp
   // 디버깅 모드: 순차 실행
   kernel<<<1, 1>>>(...)
   ```

3. **성능 일관성**
   - 최적의 Block/Grid 크기를 한 번 찾으면 재사용 가능.
   - 데이터 크기가 바뀌어도 설정을 바꿀 필요 없음.

### 권장 Grid 크기 계산

```python
# 권장: SM 수의 배수로 설정
device = torch.cuda.current_device()
sm_count = torch.cuda.get_device_properties(device).multi_processor_count

threads_per_block = 256
blocks_per_grid = sm_count * 4  # SM당 4개 Block (경험적 최적값)
```

---

## 🖥️ Device Properties: 내 GPU 해부하기

코드를 최적화하려면 내 GPU의 스펙을 알아야 합니다.

### CUDA API로 조회할 수 있는 정보

| 속성 | 의미 | 예시 (RTX 3060) |
|------|------|-----------------|
| `name` | GPU 이름 | "NVIDIA GeForce RTX 3060" |
| `multi_processor_count` | SM 개수 | 28 |
| `max_threads_per_block` | 블록당 최대 스레드 | 1024 |
| `max_threads_per_multi_processor` | SM당 최대 스레드 | 1536 |
| `warp_size` | Warp 크기 | 32 |
| `total_memory` | 총 VRAM | ~12 GB |
| `shared_memory_per_block` | 블록당 Shared Memory | 48 KB |

### Python (PyTorch)에서 조회

```python
import torch

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)

print(f"GPU: {props.name}")
print(f"SM 개수: {props.multi_processor_count}")
print(f"블록당 최대 스레드: {props.max_threads_per_block}")
print(f"Warp 크기: {props.warp_size}")
```

---

## 📚 핵심 용어 정리

| 용어 | 정의 |
|------|------|
| **SM (Streaming Multiprocessor)** | GPU의 기본 연산 유닛. Block이 실행되는 곳. |
| **Warp** | 32개 스레드가 동기화되어 실행되는 단위. |
| **SIMT** | Single Instruction, Multiple Threads. GPU 실행 모델. |
| **Warp Divergence** | Warp 내 스레드들이 다른 분기를 타서 성능 저하되는 현상. |
| **Occupancy** | SM의 Warp 활용률 (%). |
| **Grid-Stride Loop** | 스레드가 stride만큼 건너뛰며 여러 데이터를 처리하는 패턴. |
| **Latency Hiding** | 메모리 대기 시간을 다른 Warp 실행으로 숨기는 기법. |

---

## 🔜 다음 주 예고: Memory Hierarchy

Week 3에서는 **메모리 최적화**를 다룹니다.

- Global Memory의 Coalesced Access
- Shared Memory를 활용한 Tiling
- Matrix Multiplication 최적화

> "Compute는 공짜, Memory는 비싸다" — GPU 최적화의 핵심 격언

---

## ✅ Week 2 과제 (Homework)

### 1. 과제 1: 내 GPU 스펙 시트 작성
- `lecture2_device_query.py` 실행 결과 캡처.
- 다음 표 채우기:
  | 항목 | 내 GPU 값 |
  |------|-----------|
  | GPU Name | (예: RTX 3060) |
  | SM Count | (예: 28) |
  | Max Threads per SM | (예: 1536) |
  | Max Warps per SM | (계산: Max Threads / 32) |

### 2. 과제 2: 1D Convolution with Grid-Stride Loop
- **목표**: Vector Add보다 복잡한 커널을 Grid-Stride Loop로 구현.
- **설명**: 입력 벡터 `x`에 대해 3-tap 필터 `[0.2, 0.6, 0.2]`를 적용하는 1D Convolution 구현.
- **수식**: `y[i] = 0.2*x[i-1] + 0.6*x[i] + 0.2*x[i+1]`
- **주의사항**:
  - `i=0`일 때 `x[-1]`, `i=N-1`일 때 `x[N]` 접근 시 범위 체크 필요 (0으로 처리).
  - Grid-Stride Loop 패턴 반드시 적용.

### 3. 과제 3: Warp Divergence 실험
- **목표**: 분기(Branch)가 성능에 미치는 악영향 직접 확인.
- **실험**: 다음 두 커널을 구현하고 `ncu` 또는 `Event`로 수행 시간 비교.
  - **Kernel A (Bad)**: `if (threadIdx.x % 2 == 0) { ... }` (짝수/홀수 분기)
  - **Kernel B (Good)**: `if (threadIdx.x < 16) { ... }` (Warp 단위 분기)
- **예상 결과**: Kernel A가 더 느릴 것임 (왜? 직렬화 때문).

---

이번 과제를 통해 GPU 아키텍처와 Warp의 동작 원리를 몸소 체험해 보시길 바랍니다. 궁금한 점은 언제든 질문해 주세요. 화이팅! 🚀

