"""06장 실습: 05장 RAG 결과를 RAGAS 지표로 채점합니다.

실행: python -m ch06.ragas_eval                                   (05-4 결과 파일)
      python -m ch06.ragas_eval --input results/ch06_rag_c200_k3.json
"""
import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
    NoiseSensitivity,
)

from ch06.setup import ragas_embeddings, ragas_llm

llm, embeddings = ragas_llm(), ragas_embeddings()

# 지표 이름: (지표 객체, 그 지표가 받는 입력 이름들)
METRICS = {
    "context_precision": (ContextPrecision(llm=llm), ("user_input", "reference", "retrieved_contexts")),
    "context_recall": (ContextRecall(llm=llm), ("user_input", "retrieved_contexts", "reference")),
    "faithfulness": (Faithfulness(llm=llm), ("user_input", "response", "retrieved_contexts")),
    "answer_relevancy": (AnswerRelevancy(llm=llm, embeddings=embeddings), ("user_input", "response")),
    "factual_correctness": (FactualCorrectness(llm=llm), ("response", "reference")),
    "noise_sensitivity": (NoiseSensitivity(llm=llm), ("user_input", "response", "reference", "retrieved_contexts")),
}


def make_input(row: dict, names: tuple) -> dict:
    """05장 결과 파일의 필드를 RAGAS 입력 이름으로 바꿔, 지표가 받는 것만 골라 넘깁니다."""
    values = {
        "user_input": row["question"],
        "response": row["answer"],
        "reference": row["reference"],
        "retrieved_contexts": [c["text"] for c in row["retrieved"]],
    }
    return {name: values[name] for name in names}


semaphore = asyncio.Semaphore(4)  # 서버에 한꺼번에 보내는 요청 수를 제한합니다


async def score_one(name: str, row: dict) -> float | None:
    metric, names = METRICS[name]
    async with semaphore:
        try:
            result = await metric.ascore(**make_input(row, names))
            return None if math.isnan(result.value) else round(float(result.value), 4)
        except Exception as e:  # 한 문항의 실패로 전체가 멈추지 않게 합니다
            row.setdefault("errors", {})[name] = f"{type(e).__name__}: {str(e)[:120]}"
            return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/ch05_evaluate_rag_c400_k3.json")
    parser.add_argument("--metrics", default=",".join(METRICS))
    args = parser.parse_args()
    names = args.metrics.split(",")

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    start = time.time()
    for name in names:
        scores = await asyncio.gather(*(score_one(name, r) for r in rows))
        for r, s in zip(rows, scores):
            r.setdefault("ragas", {})[name] = s
        valid = [s for s in scores if s is not None]
        print(f"{name:<20} 평균 {sum(valid) / len(valid):.3f}  (채점 {len(valid)}/{len(rows)}, 누적 {time.time() - start:.0f}초)")

    stem = Path(args.input).stem.split("_")[-2:]  # 예: ch05_evaluate_rag_c400_k3 → c400_k3
    out = Path("results") / f"ch06_ragas_{'_'.join(stem)}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    asyncio.run(main())
