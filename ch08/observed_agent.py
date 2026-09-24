"""08-5 08-2 의 Agent 에 DeepEval 관측(tracing)을 붙입니다. Agent 코드는 고치지 않습니다.

LLM 호출(call_llm), 도구 호출(call_tool), 계획 세우기(make_plan)를 @observe 로 감싸 바꿔 끼우면,
실행할 때마다 DeepEval 이 Span 을 기록하고 Trace 를 만듭니다.
Confident AI 에 로그인하지 않았으므로 Trace 는 외부로 보내지 않고 이 컴퓨터 안에서만 씁니다.
"""
import json

from deepeval.tracing import observe, update_current_span, update_current_trace

import ch08.agent as agent
import ch08.planner as planner
from common.llm import MODEL

_call_llm, _call_tool, _make_plan = agent.call_llm, agent.call_tool, planner.make_plan


@observe(type="llm", model=MODEL)
def traced_llm(messages: list[dict], tool_schemas: list[dict]):
    response = _call_llm(messages, tool_schemas)
    message = response.choices[0].message
    calls = [f"{c.function.name}({c.function.arguments})" for c in message.tool_calls or []]
    update_current_span(input=messages[-1]["content"], output=message.content or f"도구 호출: {', '.join(calls)}")
    return response


@observe(type="tool")
def traced_tool(name: str, arguments: str):
    args, output = _call_tool(name, arguments)
    update_current_span(name=name, input=args, output=output)  # Span 이름을 도구 이름으로 바꿉니다
    return args, output


@observe(name="plan")
def traced_plan(task: str, version: str):
    steps, step = _make_plan(task, version)
    update_current_span(input=task, output=json.dumps({"plan": steps}, ensure_ascii=False))
    return steps, step


agent.call_llm, agent.call_tool, planner.make_plan = traced_llm, traced_tool, traced_plan


@observe(type="agent", available_tools=["search_document", "calculator", "get_weather"])
def run_observed(task: str, version: str, plan: bool = False, context: bool = False) -> dict:
    trace = planner.run_planned_agent(task, version) if plan else agent.run_agent(task, version)
    # context=True 면 Agent 가 받은 시스템 프롬프트(오늘 날짜 등)도 Trace 에 남겨 Judge 가 볼 수 있게 합니다
    request = f"[시스템 프롬프트]\n{agent.VERSIONS[version][0]}\n\n[사용자 요청]\n{task}" if context else task
    update_current_span(input=request, output=trace["final_answer"])
    update_current_trace(input=request, output=trace["final_answer"])
    return trace
