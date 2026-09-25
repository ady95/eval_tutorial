"""10-2 프로젝트 2: 업무 Agent 평가 시스템. 과제 실행 → Trace → 결과·도구·경로 평가 → 리포트.

실행: python -m ch10.agent_project --tasks data/agent_tasks_new.json --run new-1
      python -m ch10.agent_project --tasks data/agent_regression.json --run regression-1
실패한 과제를 회귀 평가셋에 넣는 일은 ch10.add_to_regression 이 맡습니다.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

from ch08.agent import run_agent
from ch08.agent_metrics import argument_check, judge_query, judge_task_completion, tool_calls, tool_selection
from ch08.trajectory import error_outcomes, step_efficiency, trajectory_match

parser = argparse.ArgumentParser()
parser.add_argument("--tasks", default="data/agent_tasks.json")
parser.add_argument("--version", default="v1", choices=["v0", "v1"])
parser.add_argument("--run", required=True)
args = parser.parse_args()


def evaluate(task: dict, trace: dict) -> dict:
    """08-3 · 08-4 의 지표를 한 과제에 모두 적용합니다."""
    arguments = {}
    for expected in task["expected_tools"]:
        ok = argument_check(task, expected, trace)
        if ok == "judge":
            queries = [c["arguments"].get("query", "") for c in tool_calls(trace)
                       if c["name"] == expected["name"] and c["arguments"]]
            ok = any(judge_query(task["task"], q).get("result") == "PASS" for q in queries)
        if ok is not None:
            arguments[expected["name"]] = ok
    selection = tool_selection(task["expected_tools"], trace)
    completion = judge_task_completion(task, trace)
    return {
        "completed": completion.get("result") == "PASS", "reason": completion.get("reason"),
        "tool_correct": selection["correct"], "missing": selection["missing"], "unnecessary": selection["unnecessary"],
        "arguments_ok": all(arguments.values()), "arguments": arguments,
        "path_strict": trajectory_match(task["expected_tools"], trace)["strict"],
        "efficiency": step_efficiency(task["expected_tools"], trace),
        "dead_ends": error_outcomes(trace)["dead_ends"],
    }


tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
rows = []
for task in tasks:
    trace = run_agent(task["task"], args.version)
    rows.append({**task, "trace": trace, "eval": evaluate(task, trace)})

# 리포트
n = len(rows)
ok = [r for r in rows if r["eval"]["completed"] and r["eval"]["tool_correct"] and r["eval"]["arguments_ok"]]
lines = [f"# Agent 평가 리포트: {args.run} ({args.version}, {args.tasks})", "",
         "| 지표 | 값 |", "|---|---|",
         f"| 과제 달성 | {sum(r['eval']['completed'] for r in rows)}/{n} |",
         f"| 도구 선택 | {sum(r['eval']['tool_correct'] for r in rows)}/{n} |",
         f"| 도구 인자 | {sum(r['eval']['arguments_ok'] for r in rows)}/{n} |",
         f"| 경로 일치(strict) | {sum(r['eval']['path_strict'] for r in rows)}/{n} |",
         f"| Step Efficiency 평균 | {mean(r['eval']['efficiency'] for r in rows):.2f} |",
         f"| 막다른 길 | {sum(r['eval']['dead_ends'] for r in rows)} |",
         f"| 토큰 합계 | {sum(r['trace']['total_tokens'] for r in rows):,} |",
         f"| 모두 통과한 과제 | {len(ok)}/{n} |", "", "## 실패한 과제", ""]
failures = [r for r in rows if r not in ok]
for r in failures:
    e = r["eval"]
    calls = " → ".join(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})" for c in tool_calls(r["trace"]))
    lines += [f"- {r['id']}번 {r['task']}",
              f"  - 달성 {'OK' if e['completed'] else '불가'}, 도구 {'OK' if e['tool_correct'] else '불가'}"
              f"(누락 {e['missing']}, 불필요 {e['unnecessary']}), 인자 {'OK' if e['arguments_ok'] else '불가'}",
              f"  - 호출: {calls or '(없음)'}", f"  - 답변: {r['trace']['final_answer']}", f"  - 판정 근거: {e['reason']}"]

out = Path("results") / f"ch10_agent_{args.run}"
out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
out.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n".join(lines))
