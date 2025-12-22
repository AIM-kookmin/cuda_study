"""
Week 2 실습 1: Device Properties Query (GPU 해부하기)

이 스크립트는 현재 시스템의 GPU 하드웨어 스펙을 조회합니다.
GPU 최적화를 위해서는 하드웨어의 한계를 알아야 합니다.

주요 확인 항목:
- SM (Streaming Multiprocessor) 개수
- 블록당 최대 스레드 수
- Warp 크기
- 메모리 정보
"""
import os
import torch

# 컴파일러 설정
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"


def bytes_to_gb(bytes_val: int) -> float:
    """바이트를 GB로 변환"""
    return bytes_val / (1024 ** 3)


def print_separator(title: str = "") -> None:
    """구분선 출력"""
    if title:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print('=' * 60)
    else:
        print('-' * 60)


def query_device_properties() -> None:
    """GPU 속성을 조회하고 출력"""
    
    # CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다!")
        print("   - NVIDIA GPU가 설치되어 있는지 확인하세요.")
        print("   - CUDA 드라이버가 설치되어 있는지 확인하세요.")
        return
    
    # GPU 개수 확인
    device_count = torch.cuda.device_count()
    print_separator("CUDA Device Information")
    print(f"🖥️  사용 가능한 GPU 개수: {device_count}")
    
    # 각 GPU에 대해 정보 출력
    for device_id in range(device_count):
        props = torch.cuda.get_device_properties(device_id)
        
        print_separator(f"GPU {device_id}: {props.name}")
        
        # 기본 정보
        print("\n📋 기본 정보")
        print(f"   • GPU 이름: {props.name}")
        print(f"   • Compute Capability: {props.major}.{props.minor}")
        print(f"   • 총 VRAM: {bytes_to_gb(props.total_memory):.2f} GB")
        
        # SM 및 코어 정보
        print("\n🔧 연산 유닛 정보")
        print(f"   • SM (Streaming Multiprocessor) 개수: {props.multi_processor_count}")
        print(f"   • Warp 크기: {props.warp_size}")
        
        # 스레드 제한
        print("\n🧵 스레드 제한")
        # PyTorch 버전에 따라 속성 이름이 다를 수 있음
        try:
            max_threads = props.max_threads_per_block
        except AttributeError:
            max_threads = 1024  # 일반적인 기본값
        print(f"   • 블록당 최대 스레드 수: {max_threads}")
        max_warps_per_block = max_threads // props.warp_size
        print(f"   • 블록당 최대 Warp 수: {max_warps_per_block}")
        
        # 블록 차원 제한
        print("\n📐 블록 차원 제한 (max_threads_per_block_dim)")
        # PyTorch에서는 직접 제공하지 않으므로 일반적인 값 사용
        print(f"   • X 차원: 1024")
        print(f"   • Y 차원: 1024")
        print(f"   • Z 차원: 64")
        
        # Grid 차원 제한
        print("\n🌐 Grid 차원 제한 (max_grid_dim)")
        print(f"   • X 차원: 2^31 - 1 (약 21억)")
        print(f"   • Y 차원: 65535")
        print(f"   • Z 차원: 65535")
        
        # 메모리 정보
        print("\n💾 메모리 정보")
        print(f"   • 총 Global Memory: {bytes_to_gb(props.total_memory):.2f} GB")
        # Shared Memory per Block (bytes)
        # PyTorch 2.0+ 에서는 shared_memory_per_block 속성 지원
        try:
            shared_mem = props.shared_memory_per_block
            print(f"   • 블록당 Shared Memory: {shared_mem / 1024:.1f} KB")
        except AttributeError:
            print(f"   • 블록당 Shared Memory: 48 KB (일반적인 값)")
        
        # 레지스터 정보
        try:
            regs = props.regs_per_block
            print(f"   • 블록당 레지스터: {regs:,}")
        except AttributeError:
            print(f"   • 블록당 레지스터: 65536 (일반적인 값)")
        
        # 현재 메모리 사용량
        print("\n📊 현재 메모리 상태")
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        print(f"   • 할당된 메모리: {bytes_to_gb(allocated):.4f} GB")
        print(f"   • 예약된 메모리: {bytes_to_gb(reserved):.4f} GB")
        print(f"   • 사용 가능: {bytes_to_gb(props.total_memory - reserved):.2f} GB")
    
    # 권장 설정 출력
    print_separator("권장 커널 설정")
    props = torch.cuda.get_device_properties(0)
    
    threads_per_block = 256  # 일반적인 권장값
    blocks_per_grid = props.multi_processor_count * 4  # SM당 4개 블록
    
    print(f"\n🎯 일반적인 권장 설정:")
    print(f"   • threads_per_block: {threads_per_block}")
    print(f"   • blocks_per_grid: {blocks_per_grid} (SM × 4)")
    print(f"   • 총 스레드 수: {threads_per_block * blocks_per_grid:,}")
    
    print("\n💡 팁:")
    print("   • Block 크기는 32의 배수로 설정하세요 (Warp 활용)")
    print("   • Block 크기가 너무 작으면 SM 활용도가 떨어집니다")
    print("   • Block 크기가 너무 크면 Occupancy가 떨어질 수 있습니다")
    print("   • 권장 Block 크기: 128, 256, 512")


def verify_cuda_kernels_compatibility() -> None:
    """CUDA 커널 실행 호환성 테스트"""
    print_separator("CUDA 커널 호환성 테스트")
    
    # 간단한 텐서 연산으로 CUDA 동작 확인
    try:
        a = torch.randn(1000, device='cuda')
        b = torch.randn(1000, device='cuda')
        c = a + b
        torch.cuda.synchronize()
        print("✅ CUDA 텐서 연산 성공!")
        print(f"   • 테스트 텐서 크기: {a.shape}")
        print(f"   • 결과 합계: {c.sum().item():.4f}")
    except Exception as e:
        print(f"❌ CUDA 텐서 연산 실패: {e}")


if __name__ == "__main__":
    print("\n" + "🔍 GPU Device Query".center(60))
    print("=" * 60)
    
    query_device_properties()
    verify_cuda_kernels_compatibility()
    
    print("\n" + "=" * 60)
    print("🎉 Device Query 완료!")
    print("=" * 60 + "\n")
