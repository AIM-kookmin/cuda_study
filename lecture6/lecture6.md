# Lecture 6: Reductions & Warp-level Primitives

## 학습 목표

1. **Parallel Reduction** 알고리즘의 원리를 이해하고, CPU 순차 처리와의 차이를 설명할 수 있다
2. Reduction 커널을 **5단계에 걸쳐 최적화**하는 과정을 따라가며 각 단계의 병목을 파악한다
3. **Warp Shuffle** 프리미티브의 동작 원리를 이해하고 직접 사용할 수 있다
4. **Atomic Operations**의 종류, 용도, 성능 트레이드오프를 이해한다
5. **Histogram** 같은 실전 패턴에서 atomic과 privatization을 적용할 수 있다

---

## Part 1: Parallel Reduction이란?

### 1.1 문제 정의

배열 `[a0, a1, a2, ..., a(N-1)]`에서 하나의 스칼라 값을 뽑아내는 연산:
- **Sum**: `a0 + a1 + a2 + ... + a(N-1)`
- **Max**: `max(a0, a1, a2, ..., a(N-1))`
- **Min**: `min(a0, a1, a2, ..., a(N-1))`
- **Product**: `a0 * a1 * a2 * ... * a(N-1)`

이런 연산을 **reduction**이라 부른다. 결합법칙(associative)과 교환법칙(commutative)이 성립하는 이항 연산이면 모두 적용 가능하다.

### 1.2 CPU에서의 Reduction

```cpp
// CPU: 순차 처리 O(N)
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum += arr[i];
}
```

N개 원소를 하나씩 더하므로 **O(N)** 연산이 필요하다. 이것을 병렬화할 수 있을까?

### 1.3 병렬 Reduction의 핵심 아이디어: 트리 구조

핵심 아이디어: **짝을 지어서 동시에 합산하면 단계(step)마다 절반으로 줄어든다.**

```
Step 0:  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]     ← N=8개 원소
          \  /      \  /      \  /      \  /
Step 1:   [3]      [7]      [11]     [15]             ← N/2 = 4개
            \      /            \      /
Step 2:     [10]                [26]                   ← N/4 = 2개
               \                /
Step 3:           [36]                                 ← 최종 합 (log2(8) = 3 단계)
```

- **시간 복잡도**: O(log N) — 8개 원소를 3단계만에 합산
- **일(work) 복잡도**: O(N) — 총 덧셈 횟수는 여전히 N-1회
- **병렬도**: Step 0에서 N/2개 스레드가 동시에 작업

| N | 순차 O(N) | 병렬 O(log N) | 속도 향상 |
|---|-----------|---------------|-----------|
| 256 | 256 steps | 8 steps | 32x |
| 1,024 | 1,024 steps | 10 steps | ~100x |
| 1,000,000 | 1,000,000 steps | 20 steps | ~50,000x |

### 1.4 GPU에서의 Reduction 구조

GPU에서는 reduction을 **2단계**로 수행한다:

```
[전체 배열: N개 원소]
        |
   ┌────┼────┬────┬────┐
   ▼    ▼    ▼    ▼    ▼
 Block0 Block1 Block2 Block3 ...   ← 각 블록이 부분 합 계산 (shared memory)
   |      |      |      |
   ▼      ▼      ▼      ▼
 [p0]   [p1]   [p2]   [p3]  ...   ← partial sums (블록 수만큼)
        |
   CPU에서 최종 합산 (또는 2차 커널)
```

**왜 2단계인가?**
- 블록 간 동기화(`__syncthreads()`)는 불가능하다 — CUDA에서 블록은 독립 실행 단위
- 따라서 각 블록이 자기 담당 영역을 reduction하고, partial sum을 출력
- 마지막에 partial sum들을 다시 합산 (CPU 또는 2차 커널)

---

## Part 2: Reduction 최적화 5단계

NVIDIA의 고전 논문 "Optimizing Parallel Reduction in CUDA" (Mark Harris)의 단계를 따른다.

### Stage 1: Interleaved Addressing (Divergent Branch)

```
블록 내 스레드: [t0] [t1] [t2] [t3] [t4] [t5] [t6] [t7]
shared memory:  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]

Step 1 (stride=1):
  t0: s[0] += s[1]  →  [3]  [2]  [3]  [4]  [5]  [6]  [7]  [8]
  t2: s[2] += s[3]  →  [3]  [2]  [7]  [4]  [5]  [6]  [7]  [8]
  t4: s[4] += s[5]  →  [3]  [2]  [7]  [4] [11]  [6]  [7]  [8]
  t6: s[6] += s[7]  →  [3]  [2]  [7]  [4] [11]  [6] [15]  [8]
  (t1, t3, t5, t7은 놀고 있음)

Step 2 (stride=2):
  t0: s[0] += s[2]  → [10]  [2]  [7]  [4] [11]  [6] [15]  [8]
  t4: s[4] += s[6]  → [10]  [2]  [7]  [4] [26]  [6] [15]  [8]
  (t1,t2,t3,t5,t6,t7 놀고 있음)

Step 3 (stride=4):
  t0: s[0] += s[4]  → [36]  ...
```

**CUDA 코드:**
```cuda
for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {       // ← divergent branch!
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
}
```

**문제점: Warp Divergence**

`tid % (2 * stride) == 0` 조건을 보자:
- Step 1: tid=0,2,4,6,... 만 실행 → warp 32개 중 16개만 활성 (50%)
- Step 2: tid=0,4,8,12,... 만 실행 → 8개만 활성 (25%)
- Step 3: tid=0,8,16,24 만 실행 → 4개만 활성 (12.5%)

같은 warp 안에서 일부는 실행하고 일부는 대기 → **warp divergence**가 발생한다.
Warp는 32개 스레드가 같은 명령어를 실행해야 하므로, 분기가 갈리면 양쪽을 순차 실행한다.

### Stage 2: Sequential Addressing (Divergence 감소)

**아이디어**: stride를 반대로 — 큰 값에서 시작해서 줄여나간다.

```
블록 내 스레드: [t0] [t1] [t2] [t3] [t4] [t5] [t6] [t7]
shared memory:  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]

Step 1 (stride=4):    ← blockDim/2 = 4부터 시작
  t0: s[0] += s[4]  →  [6]  [2]  [3]  [4]  [5]  [6]  [7]  [8]
  t1: s[1] += s[5]  →  [6]  [8]  [3]  [4]  [5]  [6]  [7]  [8]
  t2: s[2] += s[6]  →  [6]  [8] [10]  [4]  [5]  [6]  [7]  [8]
  t3: s[3] += s[7]  →  [6]  [8] [10] [12]  [5]  [6]  [7]  [8]
  (t0~t3 모두 활성! → 연속된 스레드)

Step 2 (stride=2):
  t0: s[0] += s[2]  → [16]  [8] [10] [12]  ...
  t1: s[1] += s[3]  → [16] [20] [10] [12]  ...

Step 3 (stride=1):
  t0: s[0] += s[1]  → [36] ...
```

**CUDA 코드:**
```cuda
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {                    // ← 연속된 스레드가 활성
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
}
```

**왜 더 나은가?**
- Step 1: tid < 4인 스레드 → t0,t1,t2,t3이 **연속적으로** 활성
- 같은 warp 안에서 앞쪽 스레드가 활성, 뒷쪽이 비활성 → divergence가 줄어듦
- 메모리 접근도 연속 → **bank conflict 회피** (shared memory는 연속 접근에 최적화)

### Stage 3: First-add-during-load (Idle Thread 제거)

Stage 2의 문제: Step 1에서 이미 절반의 스레드가 놀고 있다.
256개 스레드 중 128개만 Step 1에서 활성 → 나머지 128개는 데이터 로드만 하고 끝.

**아이디어**: 데이터를 shared memory에 로드할 때 이미 2개씩 합산해버리자.

```
기존: 블록이 BLOCK_SIZE개 원소 담당
      → 로드 후 첫 step에서 절반이 놀음

개선: 블록이 2 * BLOCK_SIZE개 원소 담당
      → 로드할 때 2개를 합산하여 shared memory에 저장
      → 모든 스레드가 첫 step부터 유용한 일을 함
```

**CUDA 코드:**
```cuda
int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

// 로드 시 2개를 합산!
float val = 0.0f;
if (idx < n) val += input[idx];
if (idx + blockDim.x < n) val += input[idx + blockDim.x];
sdata[tid] = val;
__syncthreads();

// 이후는 Stage 2와 동일
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { ... }
```

**효과:**
- 블록 수가 절반으로 줄어듦 → 오버헤드 감소
- 첫 덧셈이 "공짜" (로드와 동시에 수행)
- 모든 스레드가 의미 있는 일을 함

### Stage 4: Warp 내부에서 `__syncthreads()` 제거

stride가 32 이하가 되면, 활성 스레드가 모두 **같은 warp** 안에 있다.
Warp 내부는 하드웨어적으로 동기(lockstep) 실행이므로 `__syncthreads()` 불필요!

```cuda
// 마지막 warp 최적화 (volatile 필요)
volatile float* smem = sdata;
if (tid < 32) {
    if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}
```

**주의: `volatile` 키워드**
- 컴파일러가 shared memory 접근을 레지스터에 캐싱하는 것을 방지
- 없으면 다른 스레드의 업데이트를 못 볼 수 있음

하지만 이 방법은 **Warp Shuffle이 등장하면서 구식이 되었다** — Stage 5로!

### Stage 5: Warp Shuffle (최종 최적화)

Warp Shuffle은 shared memory 없이 **레지스터 간 직접 데이터 교환**을 가능하게 한다.

```
Warp (32 threads, lane 0~31):
  각 lane이 레지스터에 값을 보유

__shfl_down_sync(mask, val, 16):
  lane 0은 lane 16의 val을 받음
  lane 1은 lane 17의 val을 받음
  ...
  lane 15는 lane 31의 val을 받음
  (lane 16~31은 자기 자신의 val 유지)
```

**Warp Shuffle로 reduction:**
```
초기:  lane0=v0, lane1=v1, ..., lane31=v31

Step 1: __shfl_down_sync(FULL_MASK, val, 16)
  lane0: val += (lane16의 val)  → v0+v16
  lane1: val += (lane17의 val)  → v1+v17
  ...
  lane15: val += (lane31의 val) → v15+v31

Step 2: __shfl_down_sync(FULL_MASK, val, 8)
  lane0: val += (lane8의 val)  → v0+v16+v8+v24
  ...

Step 3: __shfl_down_sync(FULL_MASK, val, 4)
Step 4: __shfl_down_sync(FULL_MASK, val, 2)
Step 5: __shfl_down_sync(FULL_MASK, val, 1)
  → lane 0에 전체 warp의 합이 모임!
```

**CUDA 코드 (warp-level reduction helper):**
```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // lane 0에 최종 합
}
```

**Warp Shuffle의 장점:**
| 항목 | Shared Memory | Warp Shuffle |
|------|--------------|--------------|
| 메모리 | Shared Memory (48KB 제한) | 레지스터 (제한 없음) |
| 동기화 | `__syncthreads()` 필요 | 불필요 (하드웨어 동기) |
| 지연시간 | ~5 cycles | ~1 cycle |
| Bank conflict | 가능 | 불가능 |

### 전체 최적화 효과 비교

| 단계 | 기법 | 상대 성능 (개선 이유) |
|------|------|----------------------|
| 1 | Interleaved Addressing | 1x (기준) — divergence + bank conflict |
| 2 | Sequential Addressing | ~2x — divergence 감소 |
| 3 | First-add-during-load | ~2x 추가 — idle thread 제거 |
| 4 | Warp Unroll | ~1.5x 추가 — syncthreads 오버헤드 제거 |
| 5 | Warp Shuffle | ~1.3x 추가 — shared memory 불필요, 레지스터 직접 교환 |

---

## Part 3: Warp Shuffle 프리미티브 상세

### 3.1 Warp와 Lane

GPU에서 스레드는 **warp** 단위(32개)로 실행된다. Warp 내의 각 스레드를 **lane**이라 부른다.

```
Thread ID:  0  1  2  ... 31 | 32 33 34 ... 63 | 64 ...
            └── Warp 0 ──┘  └── Warp 1 ──┘   └── ...
Lane:       0  1  2  ... 31   0  1  2  ... 31    0 ...
```

`lane = threadIdx.x % 32`

### 3.2 `__shfl_sync` (Direct Read / Broadcast)

```cuda
float result = __shfl_sync(mask, val, srcLane);
// 모든 참여 스레드가 srcLane의 val 값을 받음
```

**용도: Broadcast**
```
Before: lane0=10, lane1=20, lane2=30, lane3=40, ...
__shfl_sync(0xFFFFFFFF, val, 0):
After:  lane0=10, lane1=10, lane2=10, lane3=10, ...
        (모든 lane이 lane0의 값을 받음)
```

**실전 예: Warp 내 최솟값의 인덱스를 모든 lane에 전파**
```cuda
// 어떤 lane이 최솟값을 가지고 있는지 찾은 후
int min_lane = ...;
float min_val = __shfl_sync(FULL_MASK, my_val, min_lane);
// → 모든 lane이 최솟값을 알게 됨
```

### 3.3 `__shfl_down_sync` (Shift Down)

```cuda
float result = __shfl_down_sync(mask, val, delta);
// 현재 lane + delta 위치에 있는 스레드의 val을 읽어옴
// lane + delta >= 32이면 자기 자신의 val 유지
```

**시각화 (delta=2):**
```
lane:    0    1    2    3    4    5    6    7  ...
val:    [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]

__shfl_down_sync(mask, val, 2):
result: [a2] [a3] [a4] [a5] [a6] [a7] [a6] [a7]
         ↑각 lane이 자기+2의 값을 받음   ↑범위 초과→자기 값
```

**핵심 용도: Reduction** (위 Stage 5에서 상세 설명)

### 3.4 `__shfl_up_sync` (Shift Up)

```cuda
float result = __shfl_up_sync(mask, val, delta);
// 현재 lane - delta 위치에 있는 스레드의 val을 읽어옴
// lane - delta < 0이면 자기 자신의 val 유지
```

**시각화 (delta=1):**
```
lane:    0    1    2    3    4    5    6    7
val:    [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]

__shfl_up_sync(mask, val, 1):
result: [a0] [a0] [a1] [a2] [a3] [a4] [a5] [a6]
         ↑범위 초과  ↑각 lane이 자기-1의 값을 받음
```

**용도: Inclusive/Exclusive Scan (Prefix Sum)**
```cuda
// Exclusive prefix sum (Hillis-Steele 방식)
float val = input[lane];
for (int d = 1; d < 32; d <<= 1) {
    float received = __shfl_up_sync(FULL_MASK, val, d);
    if (lane >= d) val += received;
}
// val에 prefix sum 결과
```

### 3.5 `__shfl_xor_sync` (Butterfly / XOR Swap)

```cuda
float result = __shfl_xor_sync(mask, val, laneMask);
// 현재 lane XOR laneMask 위치에 있는 스레드의 val을 읽어옴
// lane=5, laneMask=3이면 → lane 5^3 = 6의 값을 읽음
```

**시각화 (laneMask=1):**
```
lane:    0    1    2    3    4    5    6    7
val:    [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]

__shfl_xor_sync(mask, val, 1):  (0^1=1, 1^1=0, 2^1=3, 3^1=2, ...)
result: [a1] [a0] [a3] [a2] [a5] [a4] [a7] [a6]
         ↑인접 쌍이 서로 교환
```

**용도: Butterfly Reduction (All-reduce)**

`__shfl_down_sync`는 결과가 lane 0에만 모이지만, `__shfl_xor_sync`를 사용하면 **모든 lane에 최종 합이 모인다**:

```
초기:     lane0=1, lane1=2, lane2=3, lane3=4

XOR mask=1: 인접 쌍 교환 후 합산
  lane0: 1 + val_from_lane(0^1=1) = 1+2 = 3
  lane1: 2 + val_from_lane(1^1=0) = 2+1 = 3
  lane2: 3 + val_from_lane(2^1=3) = 3+4 = 7
  lane3: 4 + val_from_lane(3^1=2) = 4+3 = 7

XOR mask=2: 2칸 떨어진 쌍 교환 후 합산
  lane0: 3 + val_from_lane(0^2=2) = 3+7 = 10
  lane1: 3 + val_from_lane(1^2=3) = 3+7 = 10
  lane2: 7 + val_from_lane(2^2=0) = 7+3 = 10
  lane3: 7 + val_from_lane(3^2=1) = 7+3 = 10

→ 모든 lane이 전체 합 10을 보유!
```

```cuda
// Butterfly all-reduce
float val = my_value;
for (int mask = 1; mask < 32; mask <<= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, mask);
}
// 모든 lane이 동일한 전체 합을 보유
```

**`__shfl_down_sync` vs `__shfl_xor_sync` 비교:**

| 항목 | `__shfl_down_sync` | `__shfl_xor_sync` |
|------|-------------------|-------------------|
| 결과 위치 | lane 0에만 | 모든 lane에 |
| 용도 | 일반 reduction | All-reduce (broadcast 불필요) |
| 코드 | 더 직관적 | butterfly 패턴 |

### 3.6 `mask` 파라미터

모든 shuffle 함수의 첫 번째 인자는 **mask** (32비트 정수):
- `0xFFFFFFFF` (= `FULL_MASK`): warp의 모든 32개 스레드가 참여
- 특정 비트만 1: 해당 lane만 참여

**규칙**: mask에 명시된 모든 lane이 실제로 shuffle 호출에 도달해야 한다.
일반적으로 `0xFFFFFFFF`를 사용하면 된다.

---

## Part 4: 완전 최적화 Reduction 설계

실습 파일 `lecture6_warp_shuffle.py`의 최종 커널 구조:

```
[전체 배열: N개 원소]
        │
   Grid-Stride Loop
   (각 스레드가 여러 원소를 합산 → 레지스터에 local_sum)
        │
        ▼
   Warp Shuffle Reduction
   (__shfl_down_sync로 warp 내 32개 → 1개로 축소)
        │
        ▼
   Shared Memory에 warp 대표값 저장
   (블록 내 warp 수 = 256/32 = 8개 → shared[8])
        │
        ▼
   첫 번째 warp가 shared[8]을 Warp Shuffle로 최종 합산
        │
        ▼
   output[blockIdx.x] = 블록 partial sum
```

**왜 Grid-Stride Loop?**
- 블록 수를 적게 유지 (예: SM 수 * 4 = 256개)
- 각 스레드가 N / (blocks * threads) 개 원소를 처리
- 블록 수가 적으면 partial sum도 적음 → 2차 reduction 비용 감소
- GPU occupancy도 충분히 유지

---

## Part 5: Atomic Operations

### 5.1 Atomic이 필요한 상황

여러 스레드가 **같은 메모리 주소**에 동시에 쓰기를 할 때:

```cuda
// 위험! Race condition
output[0] += local_sum;  // Read-Modify-Write가 원자적이지 않음

// Thread A: reads output[0] = 100
// Thread B: reads output[0] = 100  (A가 쓰기 전에 읽음!)
// Thread A: writes output[0] = 100 + 5 = 105
// Thread B: writes output[0] = 100 + 3 = 103  ← A의 결과를 덮어씀!
// 기대값: 108, 실제값: 103 또는 105 (비결정적)
```

**Atomic operation**은 Read-Modify-Write를 **하나의 불가분(indivisible) 연산**으로 수행한다:

```cuda
atomicAdd(&output[0], local_sum);  // 안전! 순서는 비결정적이지만 결과는 정확
```

### 5.2 주요 Atomic 함수들

| 함수 | 동작 | 지원 타입 |
|------|------|-----------|
| `atomicAdd(addr, val)` | `*addr += val` | int, float, double |
| `atomicSub(addr, val)` | `*addr -= val` | int |
| `atomicMax(addr, val)` | `*addr = max(*addr, val)` | int (float는 직접 구현) |
| `atomicMin(addr, val)` | `*addr = min(*addr, val)` | int (float는 직접 구현) |
| `atomicExch(addr, val)` | `old = *addr; *addr = val; return old` | int, float |
| `atomicCAS(addr, compare, val)` | `if (*addr == compare) *addr = val` | int |
| `atomicAnd(addr, val)` | `*addr &= val` | int |
| `atomicOr(addr, val)` | `*addr \|= val` | int |
| `atomicXor(addr, val)` | `*addr ^= val` | int |

### 5.3 `atomicCAS`로 커스텀 Atomic 구현

`atomicCAS` (Compare-And-Swap)는 가장 범용적인 atomic:

```cuda
int atomicCAS(int* address, int compare, int val);
// 의사코드:
//   old = *address;
//   if (old == compare) *address = val;
//   return old;
```

이를 이용해 float용 atomicMax를 구현:

```cuda
__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;        // float를 int 비트로 해석
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_val = __int_as_float(expected);
        if (old_val >= val) break;        // 이미 더 큰 값이면 중단
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (old != expected);            // 다른 스레드가 끼어들었으면 재시도
    return __int_as_float(old);
}
```

**동작 원리:**
1. 현재 값을 읽음 (`old`)
2. 내 값(`val`)이 더 크면 교체 시도
3. `atomicCAS`가 성공하면 (다른 스레드가 안 끼어들었으면) 완료
4. 실패하면 (다른 스레드가 값을 바꿨으면) 새 값으로 재시도

### 5.4 Atomic Contention 문제

Atomic은 편리하지만, 많은 스레드가 **같은 주소**에 동시 접근하면 **직렬화**된다:

```
1000개 스레드가 atomicAdd(&output[0], val):
→ 하드웨어가 한 번에 하나씩 처리
→ 사실상 순차 실행과 동일한 성능
```

**해결: Privatization (사유화)**

각 블록이 **로컬 복사본**(shared memory)에서 작업하고, 마지막에만 global에 합산:

```
[1000 스레드 → 1개 주소에 atomicAdd]     ← Contention 심각

vs.

[Block 0의 256 스레드 → shared_local[0]]  ← Block 내부에서만 contention
[Block 1의 256 스레드 → shared_local[0]]
[Block 2의 256 스레드 → shared_local[0]]
[Block 3의 256 스레드 → shared_local[0]]
                    ↓
[4개 블록만 global에 atomicAdd]           ← Contention 극소
```

### 5.5 실전 예제: Histogram

Histogram은 atomic의 대표적 사용례:
- 입력: 정수 배열 `[3, 1, 4, 1, 5, 9, 2, 6, ...]`
- 출력: 각 값의 등장 횟수 `hist[0]=0, hist[1]=2, hist[2]=1, hist[3]=1, ...`

**Naive 구현:**
```cuda
// 모든 스레드가 global memory에 직접 atomic
atomicAdd(&histogram[input[i]], 1);
```

**Privatized 구현:**
```cuda
// 1) Shared memory에 로컬 히스토그램 초기화
__shared__ int local_hist[NUM_BINS];
for (int i = tid; i < NUM_BINS; i += blockDim.x)
    local_hist[i] = 0;
__syncthreads();

// 2) Shared memory에 카운트 (블록 내부만 contention)
atomicAdd(&local_hist[input[idx]], 1);
__syncthreads();

// 3) 로컬 → 글로벌 합산 (블록 수만큼만 contention)
for (int i = tid; i < NUM_BINS; i += blockDim.x)
    if (local_hist[i] > 0)
        atomicAdd(&histogram[i], local_hist[i]);
```

**성능 차이**: bin 수가 적고 데이터가 클수록 privatization 효과가 큼.

---

## Part 6: Reduction vs Atomic 트레이드오프

| 항목 | Reduction (Shared Mem / Shuffle) | Atomic |
|------|--------------------------------|--------|
| 구현 복잡도 | 높음 (트리 구조, 동기화) | 낮음 (한 줄) |
| 정확도 | 높음 (결정적 순서) | 높음 (float은 순서 비결정적 → 반올림 차이) |
| 성능 (소규모) | 오버헤드 있음 | 괜찮음 |
| 성능 (대규모) | 좋음 (O(log N)) | 나쁨 (contention) |
| Partial Sum 관리 | 필요 (2단계) | 불필요 (1단계) |
| 유연성 | Sum/Max/Min 등 구현 필요 | atomicAdd만 한 줄 |

**가이드라인:**
- 원소 수가 적거나, 결과 주소가 분산되어 있으면 → **Atomic**
- 원소 수가 많고, 같은 주소에 집중되면 → **Reduction + 마지막에 Atomic** (하이브리드)
- 최고 성능이 필요하면 → **Grid-Stride + Warp Shuffle + Shared Memory** (완전 최적화)

---

## 실습 파일 구조

| 파일 | 내용 | 핵심 포인트 |
|------|------|-------------|
| `lecture6_reduction_naive.py` | Interleaved → Sequential → First-add | divergence, bank conflict, idle thread |
| `lecture6_warp_shuffle.py` | Warp Shuffle + 완전 최적화 | `__shfl_down_sync`, `__shfl_xor_sync`, 하이브리드 |
| `lecture6_atomic_ops.py` | Atomic + Histogram | contention, privatization, CAS |

## 실행 방법

```bash
wsl -d Ubuntu
cd /mnt/d/develop/cuda_study
source ~/cuda_env/bin/activate
export PATH=/usr/local/cuda-13.1/bin:$PATH

python lecture6/lecture6_reduction_naive.py
python lecture6/lecture6_warp_shuffle.py
python lecture6/lecture6_atomic_ops.py
```

## 핵심 요약 (치트시트)

```
Reduction 최적화 흐름:
  Interleaved → Sequential → First-add → Warp Unroll → Warp Shuffle

Warp Shuffle 함수:
  __shfl_sync(mask, val, srcLane)       ← broadcast
  __shfl_down_sync(mask, val, delta)    ← reduction (결과: lane 0)
  __shfl_up_sync(mask, val, delta)      ← prefix sum
  __shfl_xor_sync(mask, val, laneMask)  ← all-reduce (결과: 모든 lane)

Atomic 패턴:
  단순: atomicAdd(&out, val)
  커스텀: atomicCAS loop
  최적화: privatization (shared mem → global)

성능 규칙:
  Warp Shuffle > Shared Memory > Global Atomic
  레지스터 > Shared Memory > Global Memory
```
