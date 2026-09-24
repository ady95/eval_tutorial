"""08-4 계획 먼저 세우기(Plan-and-Execute): 도구를 부르기 전에 계획을 따로 세우고, 그 계획을 보여 준 채 실행합니다.

계획을 Trace 에 남겨 두면 계획의 품질과, 계획대로 움직였는지를 평가할 수 있습니다.
"""
import json
import time

from ch08.agent import VERSIONS, elapsed_ms, run_agent
from common.llm import MODEL, client

PLAN_INSTRUCTION = """지금은 도구를 부르지 말고 계획만 세우세요.
쓸 수 있는 도구는 search_document(사내 규정 검색), calculator(사칙연산), get_weather(도시·날짜별 날씨)입니다.
요청을 처리할 단계를 순서대로 JSON 으로만 쓰세요. 도구가 필요 없으면 steps 를 빈 목록으로 둡니다.
{"steps": [{"tool": "도구 이름", "purpose": "이 단계에서 할 일"}]}"""


def make_plan(task: str, version: str) -> tuple[list[dict], dict]:
    system, _ = VERSIONS[version]
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": f"{system}\n\n{PLAN_INSTRUCTION}"}, {"role": "user", "content": task}],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    try:
        steps = json.loads(text).get("steps", [])
    except json.JSONDecodeError:
        steps = []  # 계획을 읽지 못하면 빈 계획으로 둡니다 (Trace 에 원문이 남습니다)
    step = {"type": "plan", "latency_ms": elapsed_ms(t0), "tokens": response.usage.total_tokens, "content": text}
    return steps, step


def run_planned_agent(task: str, version: str = "v1") -> dict:
    plan, plan_step = make_plan(task, version)
    plan_text = "\n".join(f"{i}. {s.get('tool')}: {s.get('purpose')}" for i, s in enumerate(plan, 1)) or "도구 없이 답합니다."
    trace = run_agent(task, version, plan=plan_text)
    trace["plan"] = plan
    trace["steps"].insert(0, plan_step)
    trace["total_latency_ms"] += plan_step["latency_ms"]
    trace["total_tokens"] += plan_step["tokens"]
    return trace
