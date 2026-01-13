# Week 5: 1-4주차 실습 복습 (Hands-On Review)

> **핵심 목표**: 1-4주차에서 배운 이론을 실습 중심으로 통합 복습하며 실무 감각 체득  
> **학습 방식**: 이론 최소화, 코드 작성 및 디버깅 최대화

---

## 📌 5주차 개요

### 왜 복습 주차가 필요한가?

1-4주차 동안 많은 개념을 배웠지만, **이론 ≠ 실전**입니다.
- Week 1: JIT 컴파일, Grid-Block-Thread
- Week 2: Warp, Grid-Stride Loop, Occupancy
- Week 3: Memory Hierarchy, Shared Memory, Tiling
- Week 4: Profiling, Roofline Model, 병목 진단

이번 주는 **배운 모든 것을 종합**하여 실전 문제를 해결합니다.

---

## 🎯 5주차 학습 목표

1. **복합 커널 작성**: 여러 최적화 기법을 동시에 적용
2. **성능 디버깅**: 주어진 느린 커널의 병목을 찾아 최적화
3. **Profiling 마스터**: ncu/nsys를 자유자재로 활용
4. **실무 패턴 습득**: 실제 업무에서 자주 쓰이는 커널 패턴 연습

---

## 📚 복습 구조

### Day 1-2: Foundations Review (기초 다지기)
- JIT 컴파일 파이프라인 완전 이해
- 1D/2D/3D Thread Indexing 마스터
- Memory 접근 패턴 최적화

### Day 3-4: Advanced Techniques (고급 기법)
- Shared Memory + Tiling 실전 적용
- Warp-level Programming
- Bank Conflict 회피 기법

### Day 5-6: Performance Tuning (성능 튜닝)
- Profiling 기반 최적화 사이클
- Before/After 성능 비교
- Roofline Analysis 실습

### Day 7: Capstone Challenge (종합 문제)
- 실전 시나리오 기반 종합 문제
- 제한 시간 내 최적화 챌린지

---

## 🔄 Week 1-2 복습: 기초 커널 작성

### 실습 1: Vector Operations (벡터 연산 종합)

**목표**: 다양한 1D 커널 패턴 숙달

```python
# lecture5_review_week1.py
"""
Week 1-2 복습: 기초 벡터 연산 구현

구현할 커널들:
1. Vector Add: c[i] = a[i] + b[i]
2. Vector Scale: b[i] = alpha * a[i]
3. AXPY: y[i] = alpha * x[i] + y[i]
4. Dot Product: sum(a[i] * b[i])
5. L2 Norm: sqrt(sum(x[i]^2))

학습 포인트:
- Grid-Stride Loop 적용
- Reduction 패턴 이해
- Boundary Check 완벽 처리
"""
```

**과제**:
- 각 커널을 Grid-Stride Loop로 구현
- PyTorch 결과와 비교 검증
- 다양한 데이터 크기(N=1000, 1M, 100M)에서 테스트

---

### 실습 2: 2D Kernels (이미지 처리)

**목표**: 2D Thread Indexing 완전 이해

```python
# lecture5_review_week1_2d.py
"""
2D 커널 실습: 이미지 처리 파이프라인

구현할 커널들:
1. RGB to Grayscale: gray = 0.21*R + 0.72*G + 0.07*B
2. Image Transpose: output[j][i] = input[i][j]
3. Box Blur (3x3): 주변 픽셀 평균
4. Sobel Edge Detection

학습 포인트:
- 2D Grid/Block 설정
- 경계 조건 처리 (이미지 가장자리)
- Memory Coalescing 고려
"""
```

**과제**:
- 실제 이미지(512x512, 1920x1080)에 적용
- 경계 처리 방법 비교 (zero-padding vs clamp)
- PyTorch Conv2d와 성능 비교

---

## 🚀 Week 3 복습: Shared Memory & Tiling

### 실습 3: Matrix Multiply 완전 정복

**목표**: Tiling 기법의 완전한 이해와 적용

```python
# lecture5_review_week3_matmul.py
"""
Matrix Multiplication 단계별 구현

Level 1: Naive (Baseline)
- Global Memory만 사용
- 성능 측정 (Baseline 설정)

Level 2: Shared Memory Tiling
- 16x16 타일 크기
- __syncthreads() 적용
- 성능 향상 측정

Level 3: Bank Conflict 최적화
- Padding (+1) 추가
- Coalesced Access 보장

Level 4: Register Tiling
- 각 Thread가 4x4 타일 처리
- 더 높은 레지스터 재사용

학습 포인트:
- Tiling이 성능에 미치는 영향 정량화
- Bank Conflict 실제 영향 측정
- GFLOPS 계산 및 비교
"""
```

**과제**:
- 각 Level별로 구현하고 성능 측정
- 타일 크기 변화(8, 16, 32)에 따른 성능 변화
- cuBLAS와 비교 (도달 가능한 최대 성능 파악)

---

### 실습 4: Transpose 최적화 심화

**목표**: Memory Access Pattern의 중요성 체감

```python
# lecture5_review_week3_transpose.py
"""
Matrix Transpose 최적화 여정

Version 1: Naive
- Read Coalesced, Write Strided
- 성능 측정

Version 2: Shared Memory (기본)
- Tiling으로 Coalescing 확보
- Bank Conflict 발생

Version 3: Shared Memory + Padding
- Bank Conflict 해결
- 최종 성능 측정

추가 도전:
- In-place Transpose (정사각 행렬)
- Non-square Matrix Transpose
- Batched Transpose

학습 포인트:
- Coalescing의 실제 성능 영향 (2-10배 차이)
- Bank Conflict 해결의 미묘한 차이 (+1 padding의 마법)
- ncu로 실제 Memory Transaction 수 확인
"""
```

**과제**:
- ncu로 각 버전의 Memory Efficiency 측정
- 다양한 행렬 크기(512, 1024, 2048, 4096)에서 테스트
- Bandwidth 계산 및 Peak BW 대비 효율성

---

## 📊 Week 4 복습: Profiling & Optimization Cycle

### 실습 5: 병목 찾기 게임 (Performance Detective)

**목표**: Profiling 도구로 병목을 정확히 진단하고 해결

```python
# lecture5_review_week4_detective.py
"""
병목 시나리오 게임

제공되는 것:
- 3개의 "느린" 커널 (각각 다른 병목 유형)
  * Kernel A: Low Occupancy
  * Kernel B: Poor Coalescing
  * Kernel C: Warp Divergence

여러분의 임무:
1. ncu로 병목 진단
2. 코드 수정으로 최적화
3. 성능 향상 정량화

학습 포인트:
- ncu 메트릭 해석 능력
- 병목 유형별 해결 전략
- Before/After 성능 비교 방법론
"""
```

**병목 시나리오 상세**:

#### Scenario A: Low Occupancy Kernel
```cpp
// 문제: 레지스터를 너무 많이 사용
__global__ void low_occupancy_kernel(float* data, int n) {
    // 많은 지역 변수 (레지스터 압박)
    float temp[32];  // 레지스터 스필링 유발
    // ...
}
```
**진단**: `sm__warps_active.avg.pct < 30%`  
**해결**: 레지스터 사용 줄이기, Block Size 조정

#### Scenario B: Poor Coalescing
```cpp
// 문제: Strided Memory Access
__global__ void poor_coalescing_kernel(float* data, int stride, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i * stride];  // ❌ Non-coalesced
        // ...
    }
}
```
**진단**: `l1tex__average_t_sectors_per_request > 4.0`  
**해결**: 메모리 접근 패턴 재구성

#### Scenario C: Warp Divergence
```cpp
// 문제: Warp 내에서 분기
__global__ void divergence_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (threadIdx.x % 2 == 0) {  // ❌ Divergence!
            // 짝수 스레드 경로
        } else {
            // 홀수 스레드 경로
        }
    }
}
```
**진단**: `smsp__average_warps_issue_stalled_branch_resolving.pct > 10%`  
**해결**: Warp 단위 분기 또는 조건 없는 연산

---

### 실습 6: Roofline Analysis 실전

**목표**: 이론적 성능 한계와 실제 성능 비교

```python
# lecture5_review_week4_roofline.py
"""
Roofline Model 실습

구현할 것:
1. 다양한 Arithmetic Intensity를 가진 커널들
   - AI = 0.25 (Memory Bound): Copy
   - AI = 1.0 (Balanced): AXPY
   - AI = 4.0 (Balanced): Convolution
   - AI = 64.0 (Compute Bound): Matrix Multiply

2. 각 커널의 실제 성능 측정
   - GFLOPS
   - Memory Bandwidth
   - 이론적 최대 대비 효율성

3. Roofline Plot 생성
   - X축: Arithmetic Intensity
   - Y축: Performance (GFLOPS)
   - Roofline: Peak Performance vs BW Limit

학습 포인트:
- AI가 성능 특성을 어떻게 결정하는지
- 최적화 방향 설정 (Compute vs Memory)
- 내 GPU의 특성 완전 파악
"""
```

**과제**:
- 자신의 GPU에 대한 Roofline Plot 작성
- 각 커널이 어느 영역에 위치하는지 분석
- 최적화 전후 위치 변화 시각화

---

## 🎮 Capstone Challenge: 통합 실습 문제

### 프로젝트: 2D Convolution 완전 구현

**목표**: 1-4주차 모든 기법을 통합 적용

```python
# lecture5_capstone_conv2d.py
"""
2D Convolution 완전 구현 프로젝트

요구사항:
1. Im2Col + Matrix Multiply 방식으로 구현
2. 다양한 최적화 기법 적용
   - Shared Memory Tiling
   - Coalesced Access
   - Bank Conflict 회피
3. Profiling으로 병목 진단 및 최적화
4. PyTorch Conv2d와 성능 비교

입력:
- Input: [N, C, H, W] (Batch, Channel, Height, Width)
- Kernel: [K, C, R, S] (OutChannel, InChannel, KernelH, KernelW)
- Output: [N, K, H', W']

제약 조건:
- Stride = 1, Padding = 1 고정
- Kernel Size = 3x3 고정
- 최종 성능 목표: PyTorch 대비 60% 이상

평가 기준:
1. 정확성 (PyTorch 결과와 일치)
2. 성능 (GFLOPS)
3. 코드 품질 (가독성, 주석)
4. Profiling 분석 리포트
"""
```

**단계별 가이드**:

#### Phase 1: Naive 구현 (Baseline)
- 직접 Convolution (7중 루프)
- 성능 측정 및 병목 파악

#### Phase 2: Im2Col 변환
- Input을 Matrix로 변환
- GEMM (General Matrix Multiply) 호출

#### Phase 3: GEMM 최적화
- Week 3에서 배운 Tiling 적용
- Shared Memory 활용

#### Phase 4: 통합 최적화
- Profiling 기반 최적화
- 최종 성능 튜닝

---

## 📊 실습 체크리스트

### 필수 구현 항목

| 항목 | 난이도 | 완료 |
|------|--------|------|
| Vector Add (Grid-Stride) | ⭐ | ☐ |
| RGB to Grayscale | ⭐ | ☐ |
| Dot Product (Reduction) | ⭐⭐ | ☐ |
| Matrix Multiply (Naive) | ⭐⭐ | ☐ |
| Matrix Multiply (Tiled) | ⭐⭐⭐ | ☐ |
| Matrix Transpose (Optimized) | ⭐⭐⭐ | ☐ |
| Profiling 기반 병목 진단 | ⭐⭐⭐ | ☐ |
| Roofline Analysis | ⭐⭐⭐⭐ | ☐ |
| 2D Convolution (Capstone) | ⭐⭐⭐⭐⭐ | ☐ |

### 권장 학습 순서

```
Day 1: Vector Operations (기초 다지기)
  ├─ Vector Add
  ├─ Vector Scale
  └─ AXPY

Day 2: 2D Kernels (이미지 처리)
  ├─ RGB to Grayscale
  ├─ Image Transpose
  └─ Box Blur

Day 3: Matrix Multiply (Tiling 마스터)
  ├─ Naive 구현
  ├─ Shared Memory Tiling
  └─ Bank Conflict 최적화

Day 4: Transpose 심화 (Coalescing 마스터)
  ├─ Naive vs Tiled 비교
  ├─ Bank Conflict 해결
  └─ ncu Profiling

Day 5: Performance Detective (병목 진단)
  ├─ Low Occupancy 해결
  ├─ Poor Coalescing 해결
  └─ Warp Divergence 해결

Day 6: Roofline Analysis (이론적 한계)
  ├─ AI 계산
  ├─ 성능 측정
  └─ Plot 생성

Day 7: Capstone Challenge (2D Conv)
  ├─ Naive 구현
  ├─ Im2Col + GEMM
  ├─ 최적화 적용
  └─ 최종 성능 측정
```

---

## 🔧 디버깅 팁 모음

### 자주 발생하는 문제들

#### 1. JIT 컴파일 에러
```bash
# 캐시 삭제로 해결
rm -rf ~/.cache/torch_extensions

# 컴파일러 버전 확인
gcc --version  # 10 이상 권장
nvcc --version  # CUDA Toolkit 설치 확인
```

#### 2. 결과 불일치 (Numerical Error)
```python
# 부동소수점 오차 허용 범위 조정
torch.allclose(cuda_result, pytorch_result, rtol=1e-4, atol=1e-4)

# 최대 오차 확인
max_error = (cuda_result - pytorch_result).abs().max().item()
print(f"Max error: {max_error:.2e}")
```

#### 3. 성능이 예상보다 느림
```bash
# GPU가 정상 동작하는지 확인
nvidia-smi

# GPU Clock이 낮게 설정되어 있는지 확인
nvidia-smi -q -d CLOCK

# Profiling으로 병목 진단
ncu --set full python your_script.py
```

#### 4. Out of Memory
```python
# CUDA 메모리 확인
torch.cuda.empty_cache()
print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

---

## 📚 참고 자료

### 필수 복습 자료
- **Week 1 노트**: JIT 컴파일, Thread Indexing
- **Week 2 노트**: Warp, Grid-Stride Loop
- **Week 3 노트**: Shared Memory, Tiling
- **Week 4 노트**: Profiling, Roofline Model

### 추가 학습 자료
- NVIDIA CUDA C Programming Guide: [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
- PMPP Book Chapter 4-6: Memory Optimization
- GPU-MODE Lectures 1-4 재시청

---

## 🎯 학습 성과 체크

### 이번 주 끝날 때 여러분은...

✅ 복잡한 CUDA 커널을 **막힘없이** 작성할 수 있습니다  
✅ ncu/nsys를 **자유자재로** 활용할 수 있습니다  
✅ 병목을 **정확히 진단**하고 최적화할 수 있습니다  
✅ Roofline Model로 **이론적 한계**를 파악할 수 있습니다  
✅ 실무에서 바로 쓸 수 있는 **실전 감각**을 갖춥니다

---

## 💪 다음 주 예고

**Week 6: Reductions & Warp-level Primitives**
- Parallel Reduction 패턴
- Warp Shuffle 활용
- Atomic Operations 최적화

---

*"이론은 책에서, 실력은 손으로 배웁니다!" 🚀*
