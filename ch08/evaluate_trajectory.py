"""08-4 실습: 저장한 Trace 로 실행 경로와 계획을 평가합니다. LLM 을 부르지 않습니다.

실행: python -m ch08.evaluate_trajectory                    (v0, v1 각 3회 실행분)
      python -m ch08.evaluate_trajectory --plan             (계획 먼저 세우기 실행분)
"""
import argparse
import json
from pathlib import Path

from ch08.agent_metrics import tool_calls
from ch08.trajectory import (
    error_outcomes,
    loops,
    plan_adherence,
    plan_quality,
    step_efficiency,
    trajectory_match,
)

parser = argparse.ArgumentParser()
parser.add_argument("--plan", action="store_true")
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def load(name: str) -> list[dict]:
    return json.loads((Path("results") / name).read_text(encoding="utf-8"))


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    match = [trajectory_match(r["expected_tools"], r["trace"]) for r in rows]
    errors = [error_outcomes(r["trace"]) for r in rows]
    efficiency = sum(step_efficiency(r["expected_tools"], r["trace"]) for r in rows) / n
    calls = sum(len(tool_calls(r["trace"])) for r in rows)
    return (f"{sum(m['strict'] for m in match):2d}/{n}  {sum(m['unordered'] for m in match):2d}/{n}  "
            f"{sum(m['superset'] for m in match):2d}/{n}   {efficiency:.2f}   {calls:3d}   "
            f"{sum(loops(r['trace']) for r in rows):2d}   {sum(e['recovered'] for e in errors):2d}   "
            f"{sum(e['dead_ends'] for e in errors):2d}")


if not args.plan:
    print("버전 회차  경로 일치(정확/순서무시/포함)  효율  호출 수  반복  회복  막다른 길")
    for version in ("v0", "v1"):
        for run in range(1, args.runs + 1):
            print(f" {version}  {run}회   {summarize(load(f'ch08_traces_{version}_run{run}.json'))}")
    # 경로가 기대와 다른 과제를 하나씩 봅니다 (v0 1회차)
    print("\n[v0 1회차] 기대 경로와 다른 과제")
    for r in load("ch08_traces_v0_run1.json"):
        if not trajectory_match(r["expected_tools"], r["trace"])["strict"]:
            actual = " → ".join(f"{c['name']}{'(오류)' * c['output'].startswith('오류')}" for c in tool_calls(r["trace"]))
            expected = " → ".join(t["name"] for t in r["expected_tools"])
            print(f"  {r['id']:2d} 기대: {expected}\n     실제: {actual}")
else:
    print("버전 회차  계획 품질  계획 준수  경로 일치(정확)  효율")
    for version, run in [(v, n) for v in ("v0", "v1") for n in range(1, args.runs + 1)]:
        rows = load(f"ch08_traces_{version}_plan_run{run}.json")
        quality = [plan_quality(r["expected_tools"], r["trace"]["plan"]) for r in rows]
        adherence = [plan_adherence(r["trace"]["plan"], r["trace"]) for r in rows]
        strict = [trajectory_match(r["expected_tools"], r["trace"])["strict"] for r in rows]
        efficiency = sum(step_efficiency(r["expected_tools"], r["trace"]) for r in rows) / len(rows)
        print(f" {version}  {run}회    {sum(quality):2d}/{len(rows)}      {sum(adherence):2d}/{len(rows)}         "
              f"{sum(strict):2d}/{len(rows)}         {efficiency:.2f}")
        for r, q, a in zip(rows, quality, adherence):
            if not (q and a):
                planned = " → ".join(str(s.get("tool")) for s in r["trace"]["plan"]) or "(도구 없음)"
                actual = " → ".join(c["name"] for c in tool_calls(r["trace"])) or "(도구 없음)"
                print(f"     {r['id']:2d} 계획: {planned}\n        실제: {actual}")
