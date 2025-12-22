"""
Week 1 과제 2: cdiv (Ceiling Division) 함수 구현

문제: 총 데이터 개수 N일 때, block_size로 나누면 grid_size는 얼마?
공식: grid_size = ceil(N / block_size) = (N + block_size - 1) // block_size
"""


def cdiv(n: int, divisor: int) -> int:
    """
    Ceiling Division (올림 나눗셈)
    
    CUDA 커널 실행 시 grid size 계산에 사용된다.
    N개의 데이터를 처리할 때, 각 블록이 divisor개의 스레드를 가지면
    필요한 블록 수는 ceil(N / divisor)이다.
    
    Args:
        n: 총 데이터 개수 (또는 피제수)
        divisor: 블록당 스레드 수 (또는 제수)
    
    Returns:
        올림 나눗셈 결과 (필요한 블록 수)
    
    Examples:
        cdiv(1000, 256) = 4  # 4 * 256 = 1024 >= 1000
        cdiv(1024, 256) = 4  # 정확히 나눠떨어짐
        cdiv(1025, 256) = 5  # 5 * 256 = 1280 >= 1025
    """
    return (n + divisor - 1) // divisor


if __name__ == "__main__":
    # 테스트 케이스
    test_cases = [
        (1000, 256, 4),   # 과제에서 주어진 예시
        (1024, 256, 4),   # 정확히 나눠떨어지는 경우
        (1025, 256, 5),   # 1개 초과
        (100, 32, 4),     # 100 / 32 = 3.125 -> 4
        (0, 256, 0),      # 엣지 케이스: 데이터 없음
    ]
    
    print("=" * 50)
    print("cdiv (Ceiling Division) 테스트")
    print("=" * 50)
    
    all_passed = True
    for n, divisor, expected in test_cases:
        result = cdiv(n, divisor)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result != expected:
            all_passed = False
        print(f"cdiv({n}, {divisor}) = {result} (expected: {expected}) {status}")
    
    print("=" * 50)
    if all_passed:
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")
    
    # 실제 CUDA 커널에서의 사용 예시
    print("\n[실제 사용 예시]")
    N = 1000
    threads_per_block = 256
    blocks_per_grid = cdiv(N, threads_per_block)
    total_threads = blocks_per_grid * threads_per_block
    print(f"데이터 개수: {N}")
    print(f"블록당 스레드: {threads_per_block}")
    print(f"필요한 블록 수: {blocks_per_grid}")
    print(f"총 스레드 수: {total_threads} (>= {N} ✓)")
