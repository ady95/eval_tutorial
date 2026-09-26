"""09-2 실습: 설정 하나로 평가셋을 실행·채점하고 결과를 저장합니다. 회귀 평가 한 번에 해당합니다.

실행: python -m ch09.run_eval --run baseline-1                  (05장 설정: 청크 400, top_k 3, 05장 프롬프트)
      python -m ch09.run_eval --run friendly --prompt friendly   (프롬프트 변경)
      python -m ch09.run_eval --run qwen --model qwen3.5:9b --base-url http://localhost:11434/v1   (모델 교체)
      python -m ch09.run_eval --run smoke-1 --smoke              (CI 용 10문항)
"""
import argparse
import json
import sys
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
try:
    app = RagApp(config)
except Exception as e:  # 임베딩 서버 연결 같은 준비 단계의 실패도 문항별 오류로 남겨 결과 파일을 만듭니다
    app, setup_error = None, f"준비 실패 {type(e).__name__}: {e}"

rows = []
for q in questions:
    if app is None:
        rows.append({**q, "answer": None, "correct": "ERROR", "reason": setup_error})
        continue
    try:
        out = app.answer(q["question"])
    except Exception as e:  # 운영에서처럼 호출 실패도 결과로 남깁니다 (Reliability)
        rows.append({**q, "answer": None, "correct": "ERROR", "reason": f"{type(e).__name__}: {e}"})
        continue
    contexts = [c["text"] for c in out["retrieved"]]
    try:
        judged = judge_correct_with_docs(q["question"], out["answer"], q["reference"], contexts)
    except Exception as e:  # Judge 호출이 실패해도 문항별 오류로 남기고 계속합니다
        judged = {"error": f"{type(e).__name__}: {e}"}
    found = {c["doc_id"] for c in out["retrieved"]}
    rows.append({**q, **out, "correct": judged.get("result", "ERROR"), "reason": judged.get("reason", judged.get("error")),
                 "doc_recall": sum(d in found for d in q["docs"]) / len(q["docs"]) if q["docs"] else None})
    print(f"  {q['id']:2d} {rows[-1]['correct']:<5} {out['search_ms'] + out['rerank_ms'] + out['llm_ms']:6d}ms  {out['answer'][:50]}")

ok = [r for r in rows if r["correct"] != "ERROR"]
latency = sorted(r["search_ms"] + r["rerank_ms"] + r["llm_ms"] for r in ok)


def avg(key: str) -> int | None:
    """정상 행의 평균. 정상 행이 하나도 없으면 None 입니다."""
    return round(sum(r[key] for r in ok) / len(ok)) if ok else None


summary = {
    "config": config.name(), "n": len(rows),
    "pass_rate": round(sum(r["correct"] == "PASS" for r in rows) / len(rows), 3),
    "by_type": {t: round(sum(r["correct"] == "PASS" for r in rows if r["type"] == t)
                         / sum(r["type"] == t for r in rows), 3) for t in sorted({r["type"] for r in rows})},
    "errors": len(rows) - len(ok),
    "prompt_tokens": avg("prompt_tokens"),
    "completion_tokens": avg("completion_tokens"),
    "latency_p50_ms": round(median(latency)) if latency else None,
    "latency_p95_ms": latency[max(0, round(len(latency) * 0.95) - 1)] if latency else None,
    "search_ms": round(sum(r["search_ms"] + r["rerank_ms"] for r in ok) / len(ok)) if ok else None,
    "llm_ms": avg("llm_ms"),
    "fail_ids": [r["id"] for r in rows if r["correct"] != "PASS"],
}
out_path = Path("results") / f"ch09_run_{args.run}.json"
out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not ok:  # 모든 문항이 오류면 결과를 저장한 뒤 실패로 끝냅니다 (CI 가 이 실행을 성공으로 보지 않게)
    sys.exit(1)
