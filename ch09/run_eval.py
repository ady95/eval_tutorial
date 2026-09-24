"""09-2 실습: 설정 하나로 평가셋을 실행·채점하고 결과를 저장합니다. 회귀 평가 한 번에 해당합니다.

실행: python -m ch09.run_eval --run baseline-1                  (05장 설정: 청크 400, top_k 3, 05장 프롬프트)
      python -m ch09.run_eval --run friendly --prompt friendly   (프롬프트 변경)
      python -m ch09.run_eval --run qwen --model qwen3.5:9b --base-url http://localhost:11434/v1   (모델 교체)
      python -m ch09.run_eval --run smoke-1 --smoke              (CI 용 10문항)
"""
import argparse
import json
from pathlib import Path
from statistics import median

from ch05.rag import load_eval_set
from ch05.rag_judges import judge_correct_with_docs
from ch09.rag_app import PROMPTS, SMOKE_IDS, RagApp, RagConfig

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="결과 파일 이름 (results/ch09_run_<이름>.json)")
parser.add_argument("--chunk-size", type=int, default=400)
parser.add_argument("--top-k", type=int, default=3)
parser.add_argument("--rerank", action="store_true")
parser.add_argument("--prompt", default="v1", choices=list(PROMPTS))
parser.add_argument("--model", default=None)
parser.add_argument("--base-url", default="")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

config = RagConfig(chunk_size=args.chunk_size, top_k=args.top_k, rerank=args.rerank, prompt=args.prompt,
                   base_url=args.base_url, **({"model": args.model} if args.model else {}))
questions = [q for q in load_eval_set() if not args.smoke or q["id"] in SMOKE_IDS]
app = RagApp(config)

rows = []
for q in questions:
    try:
        out = app.answer(q["question"])
    except Exception as e:  # 운영에서처럼 호출 실패도 결과로 남깁니다 (Reliability)
        rows.append({**q, "answer": None, "correct": "ERROR", "reason": f"{type(e).__name__}: {e}"})
        continue
    contexts = [c["text"] for c in out["retrieved"]]
    judged = judge_correct_with_docs(q["question"], out["answer"], q["reference"], contexts)
    found = {c["doc_id"] for c in out["retrieved"]}
    rows.append({**q, **out, "correct": judged.get("result", "ERROR"), "reason": judged.get("reason"),
                 "doc_recall": sum(d in found for d in q["docs"]) / len(q["docs"]) if q["docs"] else None})
    print(f"  {q['id']:2d} {rows[-1]['correct']:<5} {out['search_ms'] + out['rerank_ms'] + out['llm_ms']:6d}ms  {out['answer'][:50]}")

ok = [r for r in rows if r["correct"] != "ERROR"]
latency = [r["search_ms"] + r["rerank_ms"] + r["llm_ms"] for r in ok]
summary = {
    "config": config.name(), "n": len(rows),
    "pass_rate": round(sum(r["correct"] == "PASS" for r in rows) / len(rows), 3),
    "by_type": {t: round(sum(r["correct"] == "PASS" for r in rows if r["type"] == t)
                         / sum(r["type"] == t for r in rows), 3) for t in sorted({r["type"] for r in rows})},
    "errors": len(rows) - len(ok),
    "prompt_tokens": round(sum(r["prompt_tokens"] for r in ok) / len(ok)),
    "completion_tokens": round(sum(r["completion_tokens"] for r in ok) / len(ok)),
    "latency_p50_ms": round(median(latency)),
    "latency_p95_ms": sorted(latency)[max(0, round(len(latency) * 0.95) - 1)],
    "search_ms": round(sum(r["search_ms"] + r["rerank_ms"] for r in ok) / len(ok)),
    "llm_ms": round(sum(r["llm_ms"] for r in ok) / len(ok)),
    "fail_ids": [r["id"] for r in rows if r["correct"] != "PASS"],
}
out_path = Path("results") / f"ch09_run_{args.run}.json"
out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
