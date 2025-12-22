# CUDA Parallel Programming Study 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

컴퓨터 구조 지식을 바탕으로 PyTorch와 연동되는 고성능 Custom CUDA Kernel을 직접 구현하고 최적화하는 스터디 저장소입니다.

## 📌 개요
이 프로젝트는 단순한 API 사용법을 넘어, GPU 아키텍처의 이해를 바탕으로 효율적인 병렬 알고리즘을 설계하고 구현하는 것을 목표로 합니다. 특히 `torch.utils.cpp_extension.load_inline`을 활용하여 Python 환경에서 빠르게 CUDA 커널을 실험하고 벤치마킹합니다.

## 🗓️ 스터디 커리큘럼 (15주 과정)

### Phase 1: Foundations & Architecture (1-4주차)
- **1주차**: 환경 설정 및 PyTorch-CUDA Bridge (`load_inline`)
- **2주차**: GPU 아키텍처 및 프로그래밍 모델 (Grid-Block-Thread)
- **3주차**: 메모리 계층 구조 (Global & Shared Memory, Tiling)
- **4주차**: 프로파일링 및 성능 측정 (ncu, Nsight Compute)

### Phase 2: Algorithms & Optimization Patterns (5-9주차)
- **5주차**: Reductions (Warp Shuffle 활용)
- **6주차**: Kernel Fusion 및 Optimizer 최적화
- **7주차**: Scan (Prefix Sum) 알고리즘
- **8주차**: Tensor Cores 및 WMMA 이해
- **9주차**: 양자화 기법 적용 (INT8/FP8)

### Phase 3: Advanced AI Systems (10-13주차)
- **10주차**: OpenAI Triton 활용
- **11주차**: Flash Attention 및 LLM 최적화
- **12주차**: 실전 Liger Kernel 분석
- **13주차**: Multi-GPU 프로그래밍 및 NCCL

### Phase 4: Capstone Project (14-15주차)
- **14-15주차**: 고성능 Custom Kernel 구현 프로젝트

## 🛠️ 요구 사항
본 실습을 진행하기 위해 다음 환경이 필요합니다:
- **OS**: Linux (추천)
- **GPU**: NVIDIA GPU (CUDA 지원)
- **Compiler**: `gcc-10` / `g++-10`
- **Python Libraries**: `torch`, `triton`

## 🚀 퀵 스타트
별도의 빌드 시스템 없이 Python 스크립트를 직접 실행하여 CUDA 커널을 JIT 컴파일하고 실행할 수 있습니다.

```bash
# 첫 번째 CUDA 커널(Hello World) 실행
python lecture1/lecture1_basic.py
```

## 📂 프로젝트 구조
```text
cuda_study/
├── lecture1/           # 기초 및 벤치마킹 예제
├── plan/               # 상세 학습 계획서
├── reference/          # 학습 참고 자료 (PMPP, CUDA Guide 등)
│   └── gpu-mode/       # GPU-MODE 강의 자료
├── AGENTS.md           # 에이전트 개발 가이드 및 코딩 규칙
└── README.md           # 프로젝트 소개
```

## 📚 참고 자료
- 📕 **[PMPP]** Programming Massively Parallel Processors
- 📘 **[Docs]** [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- 📺 **[GPU-MODE]** [GitHub Lectures](https://github.com/gpu-mode/lectures)

## ✍️ 진행 현황
- [x] 1주차: Hello World CUDA 커널 구현 완료
- [ ] 2주차: GPU 아키텍처 학습 중
