"""05-2 실습: 평가셋 질문으로 검색만 실행해 top_k 별 검색 지표를 계산합니다. (LLM 호출 없음)

실행: python -m ch05.retrieval_eval
      python -m ch05.retrieval_eval --chunk-size 200
"""
import argparse

from ch05.rag import Retriever, load_eval_set
from ch05.retrieval_metrics import METRICS, evidence_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=400)
    args = parser.parse_args()

    # 답이 문서에 없는 질문(none)은 정답 문서가 없으므로 검색 평가에서 뺍니다
    questions = [q for q in load_eval_set() if q["docs"]]
    retriever = Retriever(args.chunk_size)
    print(f"청크 크기 {args.chunk_size}자, 청크 {len(retriever.chunks)}개, 질문 {len(questions)}개\n")

    hits = {q["id"]: retriever.search(q["question"], top_k=10) for q in questions}
    print(" K    " + "  ".join(f"{m:>9}" for m in METRICS) + "   근거포함")
    for k in (1, 3, 5, 10):
        sums = dict.fromkeys(METRICS, 0.0)
        ev = 0.0
        for q in questions:
            retrieved = [{"doc_id": c.doc_id, "text": c.text} for c, _ in hits[q["id"]][:k]]
            for name, fn in METRICS.items():
                sums[name] += fn(retrieved, set(q["docs"]))
            ev += evidence_recall(retrieved, q["evidence"])
        n = len(questions)
        print(f"{k:2d}    " + "  ".join(f"{sums[m] / n:9.3f}" for m in METRICS) + f"   {ev / n:8.3f}")


if __name__ == "__main__":
    main()
