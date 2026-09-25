"""10-1 프로젝트 1: 사내 문서 RAG 평가 시스템. 실행 → 채점(Judge + RAGAS) → 리포트를 한 번에 만듭니다.

실행: python -m ch10.rag_project --run before-1 --reuse baseline-1          (09-2 에서 만든 답변을 다시 씁니다)
      python -m ch10.rag_project --run after-1 --top-k 5 --prompt v2          (개선안을 새로 실행합니다)
"""
import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean

from ch05.rag import load_eval_set
from ch05.rag_judges import judge_correct_with_docs
from ch06.ragas_eval import score_one
from ch09.rag_app import PROMPTS, RagApp, RagConfig

RAGAS_METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--reuse", help="09-2 의 실행 이름. 답변과 채점을 새로 만들지 않고 가져옵니다")
parser.add_argument("--top-k", type=int, default=3)
parser.add_argument("--prompt", default="v1", choices=list(PROMPTS))
args = parser.parse_args()

# 1) 실행: 평가셋의 질문마다 RAG 로 답하고, 05-4 의 문서 참고 Judge 로 채점합니다
if args.reuse:
    rows = json.loads((Path("results") / f"ch09_run_{args.reuse}.json").read_text(encoding="utf-8"))["rows"]
    config_name = f"{args.reuse} 의 답변"
else:
    app = RagApp(RagConfig(top_k=args.top_k, prompt=args.prompt))
    config_name = app.config.name()
    rows = []
    for q in load_eval_set():
        out = app.answer(q["question"])
        judged = judge_correct_with_docs(q["question"], out["answer"], q["reference"],
                                         [c["text"] for c in out["retrieved"]])
        rows.append({**q, **out, "correct": judged.get("result", "ERROR"), "reason": judged.get("reason")})


# 2) 채점: 06장의 RAGAS 지표 네 개를 더합니다
async def add_ragas():
    for name in RAGAS_METRICS:
        scores = await asyncio.gather(*(score_one(name, r) for r in rows))
        for r, s in zip(rows, scores):
            r.setdefault("ragas", {})[name] = s

asyncio.run(add_ragas())


# 3) 리포트: 요약, 유형별 정답률, 실패 목록을 Markdown 으로 씁니다
def avg(values: list) -> str:
    valid = [v for v in values if v is not None]
    return f"{mean(valid):.3f}" if valid else "-"


summary = {"정답률": avg([float(r["correct"] == "PASS") for r in rows]),
           **{name: avg([r["ragas"][name] for r in rows]) for name in RAGAS_METRICS}}
lines = [f"# RAG 평가 리포트: {args.run}", "", f"- 설정: {config_name}", f"- 문항: {len(rows)}개", "",
         "| 지표 | 값 |", "|---|---|", *[f"| {k} | {v} |" for k, v in summary.items()], "",
         "| 유형 | 문항 | 정답률 | Faithfulness | Context Recall |", "|---|---|---|---|---|"]
for t in sorted({r["type"] for r in rows}):
    group = [r for r in rows if r["type"] == t]
    lines.append(f"| {t} | {len(group)} | {avg([float(r['correct'] == 'PASS') for r in group])} | "
                 f"{avg([r['ragas']['faithfulness'] for r in group])} | {avg([r['ragas']['context_recall'] for r in group])} |")
lines += ["", "## 실패한 문항", ""]
lines += [f"- {r['id']}번 ({r['type']}) {r['question']}\n  - 답변: {r['answer']}\n  - 판정 근거: {r['reason']}"
          for r in rows if r["correct"] != "PASS"]

out = Path("results") / f"ch10_rag_{args.run}"
out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
out.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
print("\n".join(lines[:lines.index("## 실패한 문항") - 1]))  # 실패 목록은 리포트 파일에서 봅니다
print(f"\n리포트: {out.with_suffix('.md')}")
