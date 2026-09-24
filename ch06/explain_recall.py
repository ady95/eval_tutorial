"""06-1 실습: RAGAS Context Recall이 정답 문장마다 내린 판정과 이유를 꺼내 봅니다.

ContextRecall.ascore() 는 점수만 돌려주므로, 같은 프롬프트를 직접 호출해 판정 이유를 확인합니다.

실행: python -m ch06.explain_recall --id 12 --n 3
"""
import argparse
import asyncio
import json
from pathlib import Path

from ragas.metrics.collections.context_recall.util import (
    ContextRecallInput,
    ContextRecallOutput,
    ContextRecallPrompt,
)

from ch06.setup import ragas_llm


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=12)
    parser.add_argument("--n", type=int, default=3, help="반복 횟수")
    args = parser.parse_args()

    rows = {r["id"]: r for r in json.loads((Path("results") / "ch05_evaluate_rag_c400_k3.json").read_text(encoding="utf-8"))}
    row = rows[args.id]
    context = "\n".join(c["text"] for c in row["retrieved"])  # RAGAS 와 같은 방식으로 이어 붙입니다
    prompt = ContextRecallPrompt().to_string(
        ContextRecallInput(question=row["question"], context=context, answer=row["reference"]))

    llm = ragas_llm()
    for i in range(1, args.n + 1):
        output = await llm.agenerate(prompt, ContextRecallOutput)
        for c in output.classifications:
            print(f"[{i}회] attributed={c.attributed}  {c.statement}\n      └ {c.reason}")


asyncio.run(main())
