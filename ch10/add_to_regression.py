"""10-2 프로젝트 2: 여러 번 실행한 결과에서 한 번이라도 실패한 과제를 회귀 평가셋(data/agent_regression.json)에 추가합니다.

실행: python -m ch10.add_to_regression new-1 new-2 new-3
처음 실행하면 08장의 과제 20개로 회귀 평가셋을 시작합니다. 이미 들어 있는 과제는 다시 넣지 않습니다.
"""
import json
import sys
from pathlib import Path

REGRESSION = Path("data/agent_regression.json")
runs = sys.argv[1:]

regression = json.loads((REGRESSION if REGRESSION.exists() else Path("data/agent_tasks.json")).read_text(encoding="utf-8"))
known = {t["id"] for t in regression}
failed = {}  # 과제 번호: 실패한 실행 이름들
tasks = {}
for run in runs:
    for row in json.loads((Path("results") / f"ch10_agent_{run}.json").read_text(encoding="utf-8")):
        e = row["eval"]
        if not (e["completed"] and e["tool_correct"] and e["arguments_ok"]):
            failed.setdefault(row["id"], []).append(run)
            tasks[row["id"]] = {k: v for k, v in row.items() if k not in ("trace", "eval")}

added = []
for task_id, where in sorted(failed.items()):
    print(f"  {task_id}번 실패 {len(where)}/{len(runs)}회 ({', '.join(where)})  {tasks[task_id]['task']}")
    if task_id not in known:
        added.append({**tasks[task_id], "source": f"10-2 운영 과제, {len(where)}/{len(runs)}회 실패"})

REGRESSION.write_text(json.dumps(regression + added, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"회귀 평가셋에 {len(added)}개 추가 → {REGRESSION} (모두 {len(regression) + len(added)}개)")
