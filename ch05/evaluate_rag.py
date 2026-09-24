"""05-4 실습: 평가용 RAG를 실행하고 검색·생성을 나눠 직접 채점합니다.

실행: python -m ch05.evaluate_rag                         (청크 400자, top_k 3)
      python -m ch05.evaluate_rag --chunk-size 200 --top-k 5
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from ch04.human_agreement import judge_binary_v2
from ch05.rag import load_eval_set, run_rag
from ch05.rag_judges import context_precision, faithfulness, judge_relevance
from ch05.retrieval_metrics import evidence_recall, mrr, recall_at_k


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(v):
    return "  -  " if v is None else f"{v:5.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    rows = run_rag(load_eval_set(), args.chunk_size, args.top_k)
    print(f"청크 {args.chunk_size}자, top_k {args.top_k}\n")
    print(" id 유형     Recall  근거포함  CtxPrec  Faith  정답  답변")
    for r in rows:
        contexts = [c["text"] for c in r["retrieved"]]
        relevant = set(r["docs"])
        # 검색 평가 (정답 문서가 없는 none 유형은 뺍니다)
        r["recall"] = recall_at_k(r["retrieved"], relevant) if relevant else None
        r["mrr"] = mrr(r["retrieved"], relevant) if relevant else None
        r["evidence_recall"] = evidence_recall(r["retrieved"], r["evidence"]) if r["evidence"] else None
        flags = [bool(judge_relevance(r["question"], c).get("relevant")) for c in contexts]
        r["relevance"] = flags
        r["context_precision"] = context_precision(flags) if relevant else None
        # 생성 평가
        f = faithfulness(r["answer"], contexts)
        r["faithfulness"], r["claims"], r["verdicts"] = f["score"], f["claims"], f["verdicts"]
        judged = judge_binary_v2(r["question"], r["answer"], r["reference"])
        r["correct"], r["correct_reason"] = judged.get("result"), judged.get("reason")
        print(f"{r['id']:3d} {r['type']:<7} {fmt(r['recall'])}   {fmt(r['evidence_recall'])}   {fmt(r['context_precision'])}"
              f"  {fmt(r['faithfulness'])}  {r['correct'] or '오류':4}  {r['answer'][:40]}")

    print("\n유형별 평균")
    print(" 유형     개수  Recall  근거포함  CtxPrec  Faith  정답률")
    groups = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
        groups["전체"].append(r)
    for t in ["single", "multi", "calc", "none", "전체"]:
        g = groups[t]
        acc = sum(r["correct"] == "PASS" for r in g) / len(g)
        print(f" {t:<7} {len(g):4d}  {fmt(mean(r['recall'] for r in g))}   {fmt(mean(r['evidence_recall'] for r in g))}"
              f"   {fmt(mean(r['context_precision'] for r in g))}  {fmt(mean(r['faithfulness'] for r in g))}  {acc:6.2f}")

    out = Path("results") / f"ch05_evaluate_rag_c{args.chunk_size}_k{args.top_k}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
