"""06-5, 06-6 실습: RAG 설정을 바꿔 가며 같은 평가셋을 실행하고 품질과 지연 시간을 비교합니다.

실행: python -m ch06.sweep --chunk-size 400 --top-k 3
      python -m ch06.sweep --chunk-size 400 --top-k 3 --embedding-model qwen3-embedding:0.6b --embedding-base-url http://localhost:11434/v1
      python -m ch06.sweep --chunk-size 400 --top-k 3 --rerank 10
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--chunk-size", type=int, default=400)
parser.add_argument("--top-k", type=int, default=3)
parser.add_argument("--embedding-model", default=None, help="임베딩 모델을 바꿀 때 (.env 값 대신)")
parser.add_argument("--embedding-base-url", default=None)
parser.add_argument("--rerank", type=int, default=0, help="0보다 크면 이 개수만큼 검색한 뒤 Reranker로 top_k개를 고릅니다")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

# common.llm 을 가져오기 전에 환경 변수를 바꿔야 임베딩 설정이 바뀝니다 (.env 보다 우선)
if args.embedding_model:
    os.environ["EMBEDDING_MODEL"] = args.embedding_model
if args.embedding_base_url:
    os.environ["EMBEDDING_BASE_URL"] = args.embedding_base_url

import asyncio  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from ragas.metrics.collections import ContextRecall, Faithfulness  # noqa: E402

from ch05.rag import Retriever, generate, load_eval_set  # noqa: E402
from ch05.rag_judges import judge_correct_with_docs  # noqa: E402
from ch05.retrieval_metrics import evidence_recall, recall_at_k  # noqa: E402
from ch06.setup import ragas_llm  # noqa: E402
from common.llm import EMBEDDING_MODEL  # noqa: E402

DISTRACTORS = {"21-domestic-travel-2025", "22-remote-work-2024", "23-affiliate-welfare"}


def run_rag_timed(questions, retriever, top_k, reranker=None, pool=0):
    """질문마다 검색과 생성을 순서대로 실행하며 걸린 시간을 잽니다."""
    rows = []
    for q in questions:
        t0 = time.time()
        hits = retriever.search(q["question"], pool or top_k)
        if reranker:
            hits = reranker(q["question"], hits, top_k)
        t1 = time.time()
        contexts = [c.text for c, _ in hits]
        answer = generate(q["question"], contexts)
        t2 = time.time()
        rows.append({**q, "answer": answer,
                     "retrieved": [{"doc_id": c.doc_id, "index": c.index, "score": round(s, 4), "text": c.text}
                                   for c, s in hits],
                     "search_sec": round(t1 - t0, 3), "generate_sec": round(t2 - t1, 2),
                     "context_chars": sum(len(t) for t in contexts)})
    return rows


async def ragas_scores(rows):
    llm = ragas_llm()
    recall, faith = ContextRecall(llm=llm), Faithfulness(llm=llm)
    sem = asyncio.Semaphore(4)

    async def one(r):
        ctx = [c["text"] for c in r["retrieved"]]
        async with sem:
            out = {}
            for name, coro in (("ragas_context_recall", recall.ascore(user_input=r["question"], retrieved_contexts=ctx,
                                                                       reference=r["reference"])),
                               ("ragas_faithfulness", faith.ascore(user_input=r["question"], response=r["answer"],
                                                                   retrieved_contexts=ctx))):
                try:
                    v = (await coro).value
                    out[name] = None if math.isnan(v) else round(float(v), 4)
                except Exception:  # 주장이 없는 답변 등은 점수 없이 넘어갑니다
                    out[name] = None
            r.update(out)

    await asyncio.gather(*(one(r) for r in rows))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def main():
    reranker = None
    if args.rerank:
        from ch06.rerank import rerank
        reranker = rerank
    name = f"c{args.chunk_size}_k{args.top_k}" + (f"_{EMBEDDING_MODEL.replace(':', '_')}" if args.embedding_model else "") \
        + (f"_rr{args.rerank}" if args.rerank else "")
    print(f"설정: 청크 {args.chunk_size}자, top_k {args.top_k}, 임베딩 {EMBEDDING_MODEL}"
          + (f", Reranker(후보 {args.rerank}개)" if args.rerank else ""))

    questions = load_eval_set()
    t = time.time()
    retriever = Retriever(args.chunk_size)
    print(f"색인: 청크 {len(retriever.chunks)}개, {time.time() - t:.1f}초")
    rows = run_rag_timed(questions, retriever, args.top_k, reranker, args.rerank)

    asyncio.run(ragas_scores(rows))
    with ThreadPoolExecutor(args.workers) as ex:
        verdicts = list(ex.map(lambda r: judge_correct_with_docs(
            r["question"], r["answer"], r["reference"], [c["text"] for c in r["retrieved"]]), rows))
    for r, v in zip(rows, verdicts):
        r["correct"] = v.get("result")
        r["recall"] = recall_at_k(r["retrieved"], set(r["docs"])) if r["docs"] else None
        r["evidence_recall"] = evidence_recall(r["retrieved"], r["evidence"]) if r["evidence"] else None
        r["distractors"] = sum(c["doc_id"] in DISTRACTORS for c in r["retrieved"])

    summary = {
        "Recall": mean(r["recall"] for r in rows),
        "근거포함": mean(r["evidence_recall"] for r in rows),
        "헷갈리는문서": mean(r["distractors"] for r in rows),
        "CtxRecall": mean(r["ragas_context_recall"] for r in rows if r["docs"]),
        "Faith": mean(r["ragas_faithfulness"] for r in rows),
        "정답률": sum(r["correct"] == "PASS" for r in rows) / len(rows),
        "검색ms": mean(r["search_sec"] * 1000 for r in rows),
        "생성초": mean(r["generate_sec"] for r in rows),
        "문서길이": mean(r["context_chars"] for r in rows),
    }
    print("  ".join(f"{k} {v:.2f}" if k not in ("검색ms", "문서길이") else f"{k} {v:.0f}" for k, v in summary.items()))
    fails = [r["id"] for r in rows if r["correct"] != "PASS"]
    print(f"FAIL {len(fails)}개: {fails}")

    out = Path("results") / f"ch06_sweep_{name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
