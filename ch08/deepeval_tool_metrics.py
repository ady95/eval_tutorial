"""08-5 실습 1: 08-2 에서 저장한 Trace 를 DeepEval 의 도구 지표로 채점하고, 08-3 의 규칙 기반 결과와 비교합니다.

실행: python -m ch08.deepeval_tool_metrics      (v0, v1 의 1회차 Trace)
"""
import json
import os
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS", "1800")  # 07-1 의 대기 한도 문제

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import ArgumentCorrectnessMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

from ch07.judge_model import OpenAICompatibleJudge
from ch08.agent_metrics import argument_check, tool_calls, tool_selection

judge = OpenAICompatibleJudge()
METRICS = {
    "도구": ToolCorrectnessMetric(),  # 도구 이름만 비교합니다 (LLM 을 쓰지 않습니다)
    "도구+인자": ToolCorrectnessMetric(evaluation_params=[ToolCallParams.INPUT_PARAMETERS]),  # 인자 값까지 비교
    "인자(Judge)": ArgumentCorrectnessMetric(model=judge),  # 요청에 맞는 인자인지 LLM 이 판단합니다
}


def to_test_case(row: dict) -> LLMTestCase:
    return LLMTestCase(
        input=row["task"],
        actual_output=row["trace"]["final_answer"],
        tools_called=[ToolCall(name=c["name"], input_parameters=c["arguments"] or {}, output=c["output"])
                      for c in tool_calls(row["trace"])],
        expected_tools=[ToolCall(name=t["name"], input_parameters=t["arguments"]) for t in row["expected_tools"]],
        name=str(row["id"]),
    )


def ours(row: dict) -> tuple[bool, bool]:
    """08-3 의 규칙 기반 판정 (검색어는 규칙으로 잴 수 없으므로 여기서는 뺍니다)."""
    checks = [argument_check(row, t, row["trace"]) for t in row["expected_tools"]]
    return tool_selection(row["expected_tools"], row["trace"])["correct"], all(c for c in checks if c in (True, False))


def fmt(score: float | None) -> str:
    return "  - " if score is None else f"{score:.2f}"  # 오류로 점수가 없으면 - 로 표시합니다


for version in ("v0", "v1"):
    rows = json.loads((Path("results") / f"ch08_traces_{version}_run1.json").read_text(encoding="utf-8"))
    result = evaluate(
        [to_test_case(r) for r in rows if r["expected_tools"]],  # 도구가 필요 없는 과제(18번)는 뺍니다
        list(METRICS.values()),
        async_config=AsyncConfig(max_concurrent=4),
        display_config=DisplayConfig(print_results=False, show_indicator=False, inspect_after_run=False),
        error_config=ErrorConfig(ignore_errors=True),
        cache_config=CacheConfig(write_cache=False),
    )
    scores = {int(tr.name): [md.score for md in tr.metrics_data] for tr in result.test_results}
    print(f"[{version} 1회차] id  08-3 도구  08-3 인자  |  DeepEval 도구  도구+인자  인자(Judge)")
    for r in rows:
        if r["id"] not in scores:
            continue
        tool_ok, args_ok = ours(r)
        tool, tool_args, judged = scores[r["id"]]
        print(f"            {r['id']:2d}  {'OK  ' if tool_ok else '불가'}       {'OK  ' if args_ok else '불가'}       |"
              f"  {fmt(tool)}           {fmt(tool_args)}       {fmt(judged)}")
    print()
