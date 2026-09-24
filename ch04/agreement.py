"""04-3 일치도 지표: 정확도, Cohen's Kappa, Spearman 순위 상관."""
from collections import Counter


def accuracy(a: list, b: list) -> float:
    """두 판정 목록이 같은 비율."""
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: list, b: list) -> float:
    """우연히 일치할 확률을 뺀 일치도. 1이면 완전 일치, 0이면 우연 수준."""
    n = len(a)
    observed = accuracy(a, b)
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def _ranks(values: list[float]) -> list[float]:
    """같은 값에는 평균 순위를 줍니다."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    """두 값의 순위가 얼마나 같은 방향으로 움직이는지 (-1~1)."""
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)
