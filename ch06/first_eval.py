"""06-1 실습: 05장에서 만든 RAG 결과 세 개를 RAGAS로 채점해 봅니다.

실행: python -m ch06.first_eval      (먼저 python -m ch05.evaluate_rag 실행)
"""
import asyncio
import json
from pathlib import Path

from ragas.metrics.collections import ContextRecall, Faithfulness

from ch06.setup import ragas_llm

rows = json.loads((Path("results") / "ch05_evaluate_rag_c400_k3.json").read_text(encoding="utf-8"))
samples = [r for r in rows if r["id"] in (12, 19, 29)]

llm = ragas_llm()
faithfulness = Faithfulness(llm=llm)
context_recall = ContextRecall(llm=llm)


async def main():
    for r in samples:
        contexts = [c["text"] for c in r["retrieved"]]
        f = await faithfulness.ascore(user_input=r["question"], response=r["answer"], retrieved_contexts=contexts)
        c = await context_recall.ascore(user_input=r["question"], retrieved_contexts=contexts, reference=r["reference"])
        print(f"[{r['id']}] {r['question']}")
        print(f"  Faithfulness   {f.value:.2f}")
        print(f"  Context Recall {c.value:.2f}\n")


asyncio.run(main())
