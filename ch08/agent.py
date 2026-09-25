"""08장 평가할 Agent: 도구 3종을 쓰는 Tool-Calling Agent 와 실행 기록(Trace).

LLM 이 도구를 부르면 실행해 결과를 돌려주고, 도구 없이 답하면 끝나는 가장 단순한 반복 구조입니다.
실행 중 일어난 일을 모두 trace 에 남겨 08-3 ~ 08-5 에서 평가합니다.
"""
import json
import os
import time

from ch08.tools import TOOL_SCHEMAS, TOOL_SCHEMAS_V0, TOOLS
from common.llm import MODEL, client

SYSTEM = """당신은 누리솔 사내 업무 도우미입니다. 오늘은 2026년 10월 5일 월요일입니다.
- 사내 규정에 관한 질문은 search_document 로 찾은 내용만 근거로 답하세요. 규정에서 찾을 수 없으면 "규정에서 찾을 수 없습니다"라고 답하세요.
- 금액이나 시간 계산이 필요하면 반드시 calculator 를 쓰세요.
- 날씨는 get_weather 로 조회하세요.
- 도구가 필요 없는 말에는 도구 없이 답하세요.
- 답변은 두세 문장으로 간결하게 쓰세요."""

# 처음 만든 버전(v0): 흔히 쓰는 짧은 프롬프트입니다. 08-5 에서 위의 SYSTEM(v1)으로 고칩니다
SYSTEM_V0 = "당신은 누리솔 사내 업무 도우미입니다. 필요하면 도구를 사용해 질문에 답하세요."

VERSIONS = {"v0": (SYSTEM_V0, TOOL_SCHEMAS_V0), "v1": (SYSTEM, TOOL_SCHEMAS)}

# OpenAI 공식 API 의 gpt-6-luna 는 도구 호출을 쓸 때 추론 강도(reasoning_effort)를 none 으로 지정해야 합니다(공식 문서 기준).
# .env 의 AGENT_REASONING_EFFORT 로 정하고, 이 인자를 받지 않는 모델이면 비워 둡니다.
AGENT_OPTIONS = {"reasoning_effort": os.environ["AGENT_REASONING_EFFORT"]} if os.getenv("AGENT_REASONING_EFFORT") else {}


def elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def call_llm(messages: list[dict], tool_schemas: list[dict]):
    """LLM 호출. 도구 호출과 함께 함수로 떼어 두면 08-5 에서 관측(tracing)을 붙이기 쉽습니다."""
    return client.chat.completions.create(model=MODEL, messages=messages, tools=tool_schemas, **AGENT_OPTIONS)


def call_tool(name: str, arguments: str) -> tuple[dict | None, str]:
    """LLM 이 만든 인자(JSON 문자열)로 도구를 실행합니다. 실패해도 예외를 내지 않고 오류 문자열을 돌려줍니다."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return None, "오류: 인자를 JSON 으로 읽을 수 없습니다"
    if name not in TOOLS:
        return args, f"오류: '{name}' 도구는 없습니다"
    try:
        return args, TOOLS[name](**args)
    except TypeError as e:
        return args, f"오류: 인자가 맞지 않습니다 ({e})"


def run_agent(task: str, version: str = "v0", max_steps: int = 8, plan: str | None = None) -> dict:
    system, tool_schemas = VERSIONS[version]
    if plan:  # 08-4 의 계획 먼저 세우기: 미리 세운 계획을 함께 알려 줍니다
        system += f"\n\n다음 계획에 따라 진행하세요.\n{plan}"
    trace = {"task": task, "version": version, "steps": [], "final_answer": None, "stop_reason": "max_steps"}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    start = time.perf_counter()
    for _ in range(max_steps):
        t0 = time.perf_counter()
        response = call_llm(messages, tool_schemas)
        message = response.choices[0].message
        calls = message.tool_calls or []
        trace["steps"].append({
            "type": "llm", "latency_ms": elapsed_ms(t0), "tokens": response.usage.total_tokens,
            "content": message.content, "tool_calls": [c.function.name for c in calls],
        })
        if not calls:  # 도구를 부르지 않으면 최종 답변입니다
            trace["final_answer"] = message.content
            trace["stop_reason"] = "answer"
            break
        messages.append({"role": "assistant", "content": message.content, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in calls]})
        for call in calls:  # 한 번에 여러 도구를 부르기도 합니다
            t0 = time.perf_counter()
            args, output = call_tool(call.function.name, call.function.arguments)
            trace["steps"].append({"type": "tool", "name": call.function.name, "arguments": args,
                                   "output": output, "latency_ms": elapsed_ms(t0)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
    trace["total_latency_ms"] = elapsed_ms(start)
    trace["total_tokens"] = sum(s["tokens"] for s in trace["steps"] if s["type"] == "llm")
    return trace
