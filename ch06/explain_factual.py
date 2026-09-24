"""06-3 실습: RAGAS Factual Correctness가 주장을 어떻게 쪼개고 판정했는지 양방향으로 꺼내 봅니다.

실행: python -m ch06.explain_factual --id 25
"""
import argparse
import asyncio
import json
from pathlib import Path

from ragas.metrics.collections import FactualCorrectness

from ch06.setup import ragas_llm


async def show(metric: FactualCorrectness, title: str, text: str, against: str):
    claims = await metric._decompose_claims(text)
    print(f"{title}: 주장 {len(claims)}개")
    if not claims:
        return
    verdicts = await metric._verify_claims(claims, against)
    for s in verdicts.statements:
        print(f"  verdict={s.verdict}  {s.statement}\n      └ {s.reason}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=25)
    args = parser.parse_args()
    row = {r["id"]: r for r in json.loads((Path("results") / "ch05_evaluate_rag_c400_k3.json").read_text(encoding="utf-8"))}[args.id]
    print(f"질문: {row['question']}\n답변: {row['answer']}\n정답: {row['reference']}\n")

    metric = FactualCorrectness(llm=ragas_llm())
    await show(metric, "① 답변의 주장 → 정답으로 확인 (정밀도)", row["answer"], row["reference"])
    await show(metric, "② 정답의 주장 → 답변으로 확인 (재현율)", row["reference"], row["answer"])
    result = await metric.ascore(response=row["answer"], reference=row["reference"])
    print(f"\n이번 실행의 Factual Correctness = {result.value}")


asyncio.run(main())
