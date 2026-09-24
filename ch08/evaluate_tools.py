"""08-3 실습: 08-2 에서 저장한 Trace 로 과제 달성, 도구 선택, 도구 인자를 평가합니다.

실행: python -m ch08.evaluate_tools                 (v0 Agent, 3회 실행분)
      python -m ch08.evaluate_tools --version v1    (08-5 에서 고친 Agent)
"""
import argparse
import json
from pathlib import Path

from ch08.agent_metrics import (
    argument_check,
    judge_query,
    judge_task_completion,
    tool_calls,
    tool_selection,
)

parser = argparse.ArgumentParser()
parser.add_argument("--version", default="v0", choices=["v0", "v1"])
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def evaluate_trace(row: dict) -> dict:
    trace = row["trace"]
    selection = tool_selection(row["expected_tools"], trace)
    arguments, query_judgments = {}, []
    for expected in row["expected_tools"]:
        ok = argument_check(row, expected, trace)
        if ok == "judge":  # 검색어: 한 번이라도 적절한 검색어로 찾았는지 Judge 에게 묻습니다
            for call in tool_calls(trace):
                if call["name"] == expected["name"] and call["arguments"]:
                    query = call["arguments"].get("query", "")
                    query_judgments.append({"query": query, **judge_query(row["task"], query)})
            ok = any(j.get("result") == "PASS" for j in query_judgments)
        if ok is not None:  # 부르지 않은 도구는 인자 평가에서 뺍니다
            arguments[expected["name"]] = ok
    completion = judge_task_completion(row, trace)
    return {"id": row["id"], "completed": completion.get("result") == "PASS", "reason": completion.get("reason"),
            "tool_correct": selection["correct"], "missing": selection["missing"],
            "unnecessary": selection["unnecessary"], "arguments": arguments,
            "arguments_ok": all(arguments.values()), "query_judgments": query_judgments}


def mark(ok: bool) -> str:
    return "OK  " if ok else "불가"  # 표의 칸을 맞추려고 OK 뒤에 공백 두 칸을 둡니다


all_runs = []
for run in range(1, args.runs + 1):
    rows = json.loads((Path("results") / f"ch08_traces_{args.version}_run{run}.json").read_text(encoding="utf-8"))
    results = [evaluate_trace(r) for r in rows]
    all_runs.append(results)
    if run == 1:
        print(f"[{args.version} 1회차]\n id  달성  도구  인자  비고")
        for r in results:
            notes = [f"누락 {r['missing']}"] * bool(r["missing"]) + [f"불필요 {r['unnecessary']}"] * bool(r["unnecessary"])
            notes += [f"인자 오류 {name}" for name, ok in r["arguments"].items() if not ok]
            print(f" {r['id']:2d}  {mark(r['completed'])}  {mark(r['tool_correct'])}  {mark(r['arguments_ok'])}  {', '.join(notes)}")

print(f"\n[{args.version}] 회차  과제 달성  도구 선택  도구 인자  답은 맞고 과정이 틀림 / 과정은 맞고 답이 틀림")
for run, results in enumerate(all_runs, 1):
    n = len(results)
    process_ok = [r["tool_correct"] and r["arguments_ok"] for r in results]
    right_answer_wrong_process = [r["id"] for r, p in zip(results, process_ok) if r["completed"] and not p]
    wrong_answer_right_process = [r["id"] for r, p in zip(results, process_ok) if not r["completed"] and p]
    print(f"      {run}회     {sum(r['completed'] for r in results):2d}/{n}      {sum(r['tool_correct'] for r in results):2d}/{n}"
          f"      {sum(r['arguments_ok'] for r in results):2d}/{n}     {right_answer_wrong_process} / {wrong_answer_right_process}")

out = Path("results") / f"ch08_eval_{args.version}.json"
out.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n결과 저장: {out}")
