"""08-2 실습: Agent 평가 과제 20개를 실행하고 Trace 를 저장합니다.

실행: python -m ch08.run_agent                  (처음 만든 v0 Agent)
      python -m ch08.run_agent --version v1     (08-5 에서 고친 Agent)
      python -m ch08.run_agent --version v1 --plan   (08-4 계획 먼저 세우기)
      python -m ch08.run_agent --id 15          (과제 하나의 Trace 를 자세히 봅니다)
"""
import argparse
import json
from pathlib import Path

from ch08.agent import run_agent
from ch08.planner import run_planned_agent

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="과제 하나만 실행하고 Trace 전체를 출력합니다")
parser.add_argument("--version", default="v0", choices=["v0", "v1"])
parser.add_argument("--plan", action="store_true", help="계획을 먼저 세우고 실행합니다 (08-4)")
parser.add_argument("--run", type=int, default=1, help="반복 실행 번호. 결과 파일 이름에 붙습니다")
args = parser.parse_args()

agent = run_planned_agent if args.plan else run_agent
tasks = json.loads(Path("data/agent_tasks.json").read_text(encoding="utf-8"))

if args.id:
    task = next(t for t in tasks if t["id"] == args.id)
    print(json.dumps(agent(task["task"], args.version), ensure_ascii=False, indent=2))
    raise SystemExit

results = []
print(f" id 유형                 LLM  도구 호출 순서                                  시간(초)")
for task in tasks:
    trace = agent(task["task"], args.version)
    results.append({**task, "trace": trace})
    llm_calls = sum(1 for s in trace["steps"] if s["type"] in ("llm", "plan"))
    tools = " → ".join(s["name"] for s in trace["steps"] if s["type"] == "tool") or "(없음)"
    print(f" {task['id']:2d} {task['type']:<20} {llm_calls:3d}  {tools:<46} {trace['total_latency_ms'] / 1000:5.1f}")

out = Path("results") / f"ch08_traces_{args.version}{'_plan' * args.plan}_run{args.run}.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n결과 저장: {out}")
