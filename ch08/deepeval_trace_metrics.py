"""08-5 실습 2: Agent 를 DeepEval 로 관측하며 실행하고, Trace 전체를 보는 지표로 채점합니다.

실행: python -m ch08.deepeval_trace_metrics                   (v0 Agent)
      python -m ch08.deepeval_trace_metrics --version v1
      python -m ch08.deepeval_trace_metrics --version v1 --plan
      python -m ch08.deepeval_trace_metrics --context      (시스템 프롬프트도 Trace 에 남깁니다)
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS", "1800")  # 07-1 의 대기 한도 문제

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import PlanAdherenceMetric, PlanQualityMetric, StepEfficiencyMetric, TaskCompletionMetric

from ch07.judge_model import OpenAICompatibleJudge
from ch08.agent_metrics import judge_task_completion
from ch08.observed_agent import run_observed
from ch08.trajectory import plan_adherence, plan_quality, step_efficiency

parser = argparse.ArgumentParser()
parser.add_argument("--version", default="v0", choices=["v0", "v1"])
parser.add_argument("--plan", action="store_true")
parser.add_argument("--context", action="store_true")
args = parser.parse_args()
name = f"{args.version}{'_plan' * args.plan}{'_context' * args.context}"

judge = OpenAICompatibleJudge()
metrics = [TaskCompletionMetric(model=judge), StepEfficiencyMetric(model=judge),
           PlanQualityMetric(model=judge), PlanAdherenceMetric(model=judge)]
tasks = json.loads(Path("data/agent_tasks.json").read_text(encoding="utf-8"))
dataset = EvaluationDataset(goldens=[Golden(input=t["task"]) for t in tasks])
folder = Path("results") / f"ch08_deepeval_{name}"

traces = []
for golden in dataset.evals_iterator(
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=4),
    display_config=DisplayConfig(print_results=False, show_indicator=False, inspect_after_run=False,
                                 results_folder=str(folder)),  # 채점 결과를 이 폴더에 JSON 으로 저장합니다
    cache_config=CacheConfig(write_cache=False),
):
    traces.append(run_observed(golden.input, args.version, plan=args.plan, context=args.context))

# 저장된 채점 결과를 읽어 과제 순서대로 맞춥니다
saved = json.loads(max(folder.glob("test_run_*.json")).read_text(encoding="utf-8"))
scores = {tc["input"].rsplit("[사용자 요청]\n", 1)[-1]: {m["name"]: m["score"] for m in tc["metricsData"]}
          for tc in saved["testCases"]}

print(f"[{name}] id  달성(08-3)  Task Completion  효율(08-4)  Step Efficiency  계획 품질·준수(08-4)  Plan Quality  Plan Adherence")
rows = []
for task, trace in zip(tasks, traces):
    s = scores.get(task["task"], {})
    ours_done = judge_task_completion(task, trace).get("result") == "PASS"
    plan = trace.get("plan")
    ours_plan = (f"{'OK  ' if plan_quality(task['expected_tools'], plan) else '불가'}·"
                 f"{'OK  ' if plan_adherence(plan, trace) else '불가'}") if plan is not None else "(계획 없음)"
    print(f"        {task['id']:2d}  {'OK  ' if ours_done else '불가'}        {s.get('Task Completion', 0):.2f}"
          f"             {step_efficiency(task['expected_tools'], trace):.2f}        {s.get('Step Efficiency', 0):.2f}"
          f"             {ours_plan:<14}        {s.get('Plan Quality', 0):.2f}          {s.get('Plan Adherence', 0):.2f}")
    rows.append({**task, "trace": trace, "deepeval": s, "completed": ours_done})

out = Path("results") / f"ch08_deepeval_{name}.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n결과 저장: {out}")
