"""10-1 프로젝트 1: 개선 전(before-1~3)과 개선 후(after-1~3)를 지표와 문항 단위로 비교합니다. LLM 을 부르지 않습니다.

실행: python -m ch10.compare_rag
"""
import json
from pathlib import Path
from statistics import mean

METRICS = ["정답률", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]


def load(run: str) -> dict:
    return json.loads((Path("results") / f"ch10_rag_{run}.json").read_text(encoding="utf-8"))


groups = {name: [load(f"{name}-{n}") for n in (1, 2, 3)] for name in ("before", "after")}

print(f"{'지표':<20}{'개선 전(3회 평균, 범위)':<26}개선 후(3회 평균, 범위)")
for metric in METRICS + ["입력 토큰"]:
    cells = []
    for runs in groups.values():
        if metric == "입력 토큰":
            values = [mean(r["prompt_tokens"] for r in run["rows"]) for run in runs]
        else:
            values = [float(run["summary"][metric]) for run in runs]
        cells.append(f"{mean(values):.3f} ({min(values):.3f}~{max(values):.3f})")
    print(f"{metric:<20}{cells[0]:<32}{cells[1]}")


def passes(runs: list[dict], qid: int) -> int:
    """세 번 중 몇 번 PASS 했는가."""
    return sum(next(r for r in run["rows"] if r["id"] == qid)["correct"] == "PASS" for run in runs)


questions = {r["id"]: r for r in groups["before"][0]["rows"]}
fixed = [i for i in questions if passes(groups["before"], i) <= 1 and passes(groups["after"], i) >= 2]
broken = [i for i in questions if passes(groups["before"], i) >= 2 and passes(groups["after"], i) <= 1]
print("\n세 번 중 PASS 횟수가 과반 기준으로 바뀐 문항")
for label, ids in (("고쳐짐", fixed), ("깨짐", broken)):
    for i in ids:
        print(f"  {label} {i:2d}번 ({questions[i]['type']}) 개선 전 {passes(groups['before'], i)}/3 → 개선 후 "
              f"{passes(groups['after'], i)}/3  {questions[i]['question']}")
still = [i for i in questions if passes(groups["after"], i) <= 1]
print(f"  개선 후에도 과반 실패: {still}")
