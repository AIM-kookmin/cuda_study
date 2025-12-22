# Week 1

생성일: 2025년 12월 13일 오전 4:09
태그:  sim, gemini

## 📅 Phase 1: Foundations & Architecture (Weeks 1-4)

**목표:** GPU 하드웨어 구조(SM, Warp)를 이해하고, PyTorch와 CUDA를 연결하는 파이프라인을 구축한다.

### Week 1: CUDA Quickstart (Setup + Basic Kernel)

- **주제:** 환경 설정 및 Python 안에서 C++ 커널 돌리기
- **학습 자료:**
    - 📺 **[GPU-MODE] Lecture 1:** Profiling and Integrating CUDA kernels in PyTorch
    - 📺 **[GPU-MODE] Lecture 3:** Getting Started With CUDA (Colab setup)
- **활동:**
    - [ ]  `ninja`, `nvcc` 환경 구축 (WSL2 권장)
    - [ ]  Lecture 1의 `load_inline` 예제 실습
    - [ ]  `torch.utils.cpp_extension`으로 Hello World 커널 실행

---

# Week 1: GPU Architecture & Hello CUDA (Deep Dive)

**1주차 목표:**

1. **Architecture:** CPU(Latency)와 GPU(Throughput)의 설계 철학 차이를 이해한다.
2. **Environment:** PyTorch JIT 컴파일러(`load_inline`)를 통해 Python 안에서 C++ CUDA를 실행한다.
3. **Kernel Basics:** Grid–Block–Thread 계층 구조를 이해하고, 1D/2D 인덱싱을 직접 계산한다.

---

## 🟢 Step 1: 이론 배경 (Hardware & Compilation)

> 학습 자료: Lecture 2 (Recap PMPP), Lecture 1 (Slides 2–9)
> 

### 1. Latency vs. Throughput (지연 시간 vs 처리량)

컴퓨터 구조에서 CPU와 GPU는 서로 다른 목적을 위해 태어났다.

- **CPU (Latency Oriented):** "스포츠카"
    - **목표:** 명령 하나를 최대한 빨리 끝내는 것
    - **구조:** 복잡한 제어 유닛(Branch Prediction), 거대한 캐시(L1/L2/L3)를 가짐
    - **용도:** 운영체제 실행, 복잡한 로직, 순차적인 작업
- **GPU (Throughput Oriented):** "공항 셔틀버스"
    - **목표:** 느리더라도 한 번에 엄청나게 많은 데이터를 처리하는 것
    - **구조:** 제어 유닛은 단순하지만, 수천 개의 산술 논리 장치(ALU)를 가짐
    - **용도:** 픽셀 처리, 행렬 연산, 딥러닝

### 2. Heterogeneous Computing (이기종 컴퓨팅)

CUDA 프로그래밍은 혼자 돌지 않는다. 항상 **Host**와 **Device**가 2인 3각으로 움직인다.

- **Host (CPU + RAM):**
    - 전체 프로그램의 흐름을 제어한다 (Python 코드)
    - GPU에 데이터를 보내고(Input), GPU에게 "일해라(Launch)"라고 명령하고, 결과를 받아온다(Output)
- **Device (GPU + VRAM):**
    - Host가 시킨 단순 반복 작업(kernel)을 병렬로 수행한다.
    - **Kernel(커널):** GPU에서 실행되는 C++ 함수를 말한다. (`__global__` 키워드 사용)

### 3. Compilation Flow (어떻게 Python에서 C++이 돌까?)

Lecture 1에서 소개하는 **`load_inline`** 방식의 원리이다.

1. **String:** Python 파일 안에 C++ 코드를 문자열로 적는다.
2. **JIT(Just-In-Time):** 프로그램 실행 중에 `load_inline` 함수가 호출되면,
3. **Ninja & NVCC:** 백그라운드에서 `nvcc` 컴파일러가 이 문자열을 `.so`(리눅스) 또는 `.pyd`(윈도우) 라이브러리로 빌드한다.
4. **Binding:** PyBind11이 이 라이브러리를 Python 모듈로 연결해준다.
5. **Execution:** 우리는 마치 `import torch` 하듯이 C++ 함수를 호출할 수 있게 된다.

---

## 🟡 Step 2: 실전 코드 (Hello World & Profiling)

> 학습 자료: Lecture 1 (Profiling and Integration)
> 

이제 실제로 코드를 돌려볼 차례이다. 스터디원들과 이 코드를 Colab에서 실행하며 분석해 보자.

### 1. Minimal Hello World

가장 먼저 "내 GPU의 스레드 번호"를 출력해 보는 예제이다.

```python
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

# JIT 컴파일
my_module = load_inline(
    name='hello_extension',
    cpp_sources=[cpp_header],    # 선언부
    cuda_sources=[cuda_source],  # 구현부
    functions=['hello'],         # 'hello_kernel'은 지워야 합니다!
    verbose=True
)

# 실행: 블록 2개, 스레드 2개
print("\n=== Launching CUDA Kernel ===")
my_module.hello(2, 2)
print("=== Execution Finished ===")
```

```bash
rm -rf ~/.cache/torch_extensions
CC=gcc-10 CXX=g++-10 python ./lecture1.py
```

### 2. 정확한 시간 측정 (Profiling)

Lecture 1의 핵심 내용 중 하나는 **"비동기 실행의 함정"**이다.

- **문제점:** `kernel_launch()`를 호출하면 CPU는 GPU에게 명령만 던지고 바로 다음 줄로 넘어간다. 그래서 Python의 `time.time()`으로 재면 실제 연산 시간보다 훨씬 짧게 나온다.
- **해결책:** `cudaDeviceSynchronize()`를 호출하거나, `torch.cuda.Event`를 사용해야 한다.

```python
# 잘못된 측정 (Wrong)
start = time.time()
my_module.hello(100, 100)  # CPU는 명령만 내리고 바로 통과
print(f"Time: {time.time() - start}")  # 거의 0초가 나옴

# 올바른 측정 (Correct)
torch.cuda.synchronize()  # 기존 작업 대기
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
my_module.hello(100, 100)
end.record()

torch.cuda.synchronize()  # GPU 작업 완료 대기
print(f"Elapsed time: {start.elapsed_time(end)} ms")
```

---

## 🔴 Step 3: 핵심 심화 (Thread Indexing & Vector Add)

> 학습 자료: Lecture 3 (Getting Started), Lecture 2
> 

가장 헷갈리지만, 이것만 알면 CUDA 기초의 80%는 끝난다.

### 1. Grid–Block–Thread 계층 구조

GPU는 작업을 거대한 **Grid(그리드)** 단위로 받는다.

- **Grid:** 전체 작업 (예: 이미지 전체)
- **Block:** 작업의 구획 (예: 이미지의 16×16 타일). SM(Streaming Multiprocessor) 하나에 할당된다.
- **Thread:** 작업의 최소 단위 (예: 픽셀 하나)

### 2. Indexing Math (공식 암기 필수)

각 스레드는 자신이 전체 데이터 중 "몇 번째 데이터"를 처리해야 하는지 스스로 계산해야 한다.

- `blockIdx.x`: 나는 몇 번째 블록인가?
- `blockDim.x`: 한 블록에 스레드가 몇 명인가? (보통 128, 256, 1024)
- `threadIdx.x`: 나는 블록 내에서 몇 번인가?

$$
\text{Global ID} = (\text{blockIdx.x} \times \text{blockDim.x}) + \text{threadIdx.x}
$$

> 비유: 군대 연병장에 군인들이 줄 서 있다.
> 

> 
> 

> - "나는 3중대(blockIdx = 2) 소속이고, 중대 내에서 5번째(threadIdx = 4)다."
> 

> - "한 중대는 100명(blockDim = 100)이다."
> 

> - "나의 전체 번호는? $2 times 100 + 4 = 204$번"
> 

### 3. 실습 과제: Vector Addition

길이가 $N$인 배열 $A$와 $B$를 더해서 $C$를 만드는 커널이다.

**CUDA Kernel (C++)**

```cpp
__global__ void vector_add_kernel(float* a, float* b, float* c, int n) {
    // 1. 전역 인덱스 계산
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 중요: 데이터 크기보다 스레드가 많을 수 있으므로 범위 체크 (Boundary Check)
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

**Python Wrapper (데이터 준비)**

```python
def vector_add(a, b):
    n = a.numel()
    c = torch.zeros_like(a)
    
    # Grid/Block 설정 계산
    threads_per_block = 1024
    # cdiv: (n + threads - 1) // threads -> 올림 나눗셈
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # 커널 실행
    module.vector_add_kernel(
        blocks_per_grid, threads_per_block,  # <<<grid, block>>>
        a, b, c, n,  # Arguments
    )
    return c
```

---

### ✅ 1주차 스터디 과제 (Assignment)

스터디원들에게 다음 과제를 내주고, 다음 모임 때 코드를 리뷰한다.

1. **환경 구축 인증:** `load_inline`으로 "Hello CUDA"가 출력되는 스크린샷 공유.
2. **Indexing 이해하기:**
    - 총 데이터 개수 $N = 1000$일 때, block size가 256이라면 grid size는 얼마여야 하는가? (정답: 4, 왜냐하면 $4 times 256 = 1024 > 1000$)
    - 이 계산을 해주는 `cdiv` 함수를 Python으로 구현해 오기.
3. **RGB to Grayscale 구현 (Lecture 3 도전 과제):**
    - 입력: `[H, W, 3]` 크기의 랜덤 텐서(이미지).
    - 커널: 각 픽셀 $(r, g, b)$를 읽어서 $0.21r + 0.72g + 0.07b$ 공식을 적용해 `[H, W]` 출력 텐서 만들기.
    - 힌트: 2D 인덱싱이 필요하다.
        - `row = blockIdx.y * blockDim.y + threadIdx.y`
        - `col = blockIdx.x * blockDim.x + threadIdx.x`