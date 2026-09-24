"""06-6 Reranker: 임베딩 검색으로 넉넉히 뽑은 후보를 Cross-encoder로 다시 순위 매깁니다.

필요한 패키지: pip install torch --index-url https://download.pytorch.org/whl/cpu
              pip install sentence-transformers
처음 실행할 때 모델(BAAI/bge-reranker-v2-m3, 약 2GB)을 내려받습니다.
"""
from operator import itemgetter

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model = None


def rerank(question: str, hits: list, top_k: int) -> list:
    """hits 는 Retriever.search() 의 결과 [(Chunk, 점수), ...] 입니다. 점수를 Reranker 점수로 바꿔 돌려줍니다."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    scores = _model.predict([(question, chunk.text) for chunk, _ in hits])
    ranked = sorted(zip(hits, scores), key=itemgetter(1), reverse=True)
    return [(chunk, float(score)) for (chunk, _), score in ranked[:top_k]]
