"""08-5 실습 3: v0 와 v1 을 08-3 · 08-4 의 지표와 DeepEval 지표로 나란히 비교합니다. LLM 을 부르지 않습니다.

실행: python -m ch08.compare_versions
      (먼저 ch08.evaluate_tools 를 v0 · v1 로, ch08.deepeval_trace_metrics 를 v0 · v1 에 --context 로 실행)
"""
import json
import unicodedata
from pathlib import Path
from statistics import mean

from ch08.agent_metrics import tool_calls
from ch08.trajectory import error_outcomes, trajectory_match


def load(name: str):
    return json.loads((Path("results") / name).read_text(encoding="utf-8"))


def pad(text: str, width: int) -> str:
    """한글처럼 두 칸을 차지하는 글자를 세어 표의 칸을 맞춥니다."""
    used = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - used, 0)


def spread(values: list) -> str:
    """3회 실행 값을 '최소~최대' 로 나타냅니다."""
    low, high = min(values), max(values)
    return f"{low}" if low == high else f"{low}~{high}"


rows = []
for version in ("v0", "v1"):
    evals = load(f"ch08_eval_{version}.json")  # 08-3 의 3회 채점 결과
    traces = [load(f"ch08_traces_{version}_run{n}.json") for n in (1, 2, 3)]
    deepeval = load(f"ch08_deepeval_{version}_context.json")  # 08-5 실습 2 (시스템 프롬프트 포함)
    rows.append({
        "과제 달성(08-3)": spread([sum(r["completed"] for r in run) for run in evals]),
        "도구 선택(08-3)": spread([sum(r["tool_correct"] for r in run) for run in evals]),
        "도구 인자(08-3)": spread([sum(r["arguments_ok"] for r in run) for run in evals]),
        "답은 맞고 과정이 틀림": spread([sum(r["completed"] and not (r["tool_correct"] and r["arguments_ok"])
                                    for r in run) for run in evals]),
        "경로 일치 strict(08-4)": spread([sum(trajectory_match(r["expected_tools"], r["trace"])["strict"] for r in run)
                                     for run in traces]),
        "막다른 길(08-4)": spread([sum(error_outcomes(r["trace"])["dead_ends"] for r in run) for run in traces]),
        "도구 호출 수": spread([sum(len(tool_calls(r["trace"])) for r in run) for run in traces]),
        "토큰 합계": spread([sum(r["trace"]["total_tokens"] for r in run) for run in traces]),
        **{f"DeepEval {k}": f"{mean(r['deepeval'].get(k, 0) for r in deepeval):.2f}"
           for k in ("Task Completion", "Step Efficiency", "Plan Quality", "Plan Adherence")},
    })

print(f"{pad('지표', 30)}{pad('v0', 16)}v1")
for key in rows[0]:
    print(f"{pad(key, 30)}{pad(rows[0][key], 16)}{rows[1][key]}")
