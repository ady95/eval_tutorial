"""05-2 검색 지표: 검색된 청크 목록을 정답 문서·근거 문장과 비교합니다.

retrieved 는 순위대로 정렬된 [{"doc_id": ..., "text": ...}, ...] 입니다.
"""
import math


def hit_rate(retrieved: list[dict], relevant: set[str]) -> float:
    """상위 K개 안에 정답 문서가 하나라도 있으면 1."""
    return float(any(r["doc_id"] in relevant for r in retrieved))


def recall_at_k(retrieved: list[dict], relevant: set[str]) -> float:
    """정답 문서 중 상위 K개 안에 들어온 비율."""
    found = {r["doc_id"] for r in retrieved} & relevant
    return len(found) / len(relevant)


def precision_at_k(retrieved: list[dict], relevant: set[str]) -> float:
    """상위 K개 청크 중 정답 문서에서 나온 청크의 비율."""
    return sum(r["doc_id"] in relevant for r in retrieved) / len(retrieved)


def mrr(retrieved: list[dict], relevant: set[str]) -> float:
    """첫 번째 정답 청크가 나온 순위의 역수. 1위면 1, 2위면 0.5, 없으면 0."""
    for rank, r in enumerate(retrieved, start=1):
        if r["doc_id"] in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved: list[dict], relevant: set[str]) -> float:
    """정답 문서가 위에 있을수록 높은 점수. 같은 문서의 청크는 처음 한 번만 셉니다."""
    seen, dcg = set(), 0.0
    for rank, r in enumerate(retrieved, start=1):
        if r["doc_id"] in relevant and r["doc_id"] not in seen:
            dcg += 1 / math.log2(rank + 1)
            seen.add(r["doc_id"])
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), len(retrieved)) + 1))
    return dcg / ideal


def evidence_recall(retrieved: list[dict], evidence: list[str]) -> float:
    """답에 필요한 근거 문장 중 검색된 청크 안에 들어 있는 비율."""
    text = "\n".join(r["text"] for r in retrieved)
    return sum(e in text for e in evidence) / len(evidence)


METRICS = {
    "Hit": hit_rate,
    "Recall": recall_at_k,
    "Precision": precision_at_k,
    "MRR": mrr,
    "NDCG": ndcg_at_k,
}
