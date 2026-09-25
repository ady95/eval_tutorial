"""08-3 Agent 평가 지표: 도구 선택과 인자는 규칙으로, 과제 달성과 검색어는 Judge 로 잽니다."""
from ch03.judges import call_judge
from ch08.tools import calculator


def tool_calls(trace: dict) -> list[dict]:
    return [s for s in trace["steps"] if s["type"] == "tool"]


# ---------------------------------------------------------------- 규칙 기반
def tool_selection(expected_tools: list[dict], trace: dict) -> dict:
    """기대한 도구를 빠짐없이, 쓸데없는 도구 없이 불렀는가. 호출 순서와 반복은 08-4 에서 봅니다."""
    expected = [t["name"] for t in expected_tools]
    called = [c["name"] for c in tool_calls(trace)]
    missing = [name for name in expected if name not in called]
    unnecessary = [name for name in called if name not in expected]
    recall = (len(expected) - len(missing)) / len(expected) if expected else float(not called)
    return {"correct": not missing and not unnecessary, "recall": recall,
            "missing": missing, "unnecessary": unnecessary}


def argument_check(task: dict, expected: dict, trace: dict) -> bool | str | None:
    """기대 도구 하나의 인자를 규칙으로 확인합니다. 규칙으로 잴 수 없는 도구는 "judge" 를 돌려줍니다.

    같은 도구를 여러 번 불렀다면 한 번이라도 맞은 인자가 있으면 맞은 것으로 봅니다.
    """
    calls = [c for c in tool_calls(trace) if c["name"] == expected["name"] and c["arguments"]]
    if not calls:
        return None  # 부르지 않은 도구는 인자를 따질 수 없습니다 (도구 누락으로 셉니다)
    if expected["name"] == "get_weather":  # 도시와 날짜는 정답이 하나뿐입니다
        return any(c["arguments"].get("city") == expected["arguments"]["city"]
                   and c["arguments"].get("date") == expected["arguments"]["date"] for c in calls)
    if expected["name"] == "calculator":  # 식의 모양이 아니라 계산한 값으로 비교합니다
        values = task["expected_value"] if isinstance(task["expected_value"], list) else [task["expected_value"]]
        return any(calculator(str(c["arguments"].get("expression", ""))) in {str(v) for v in values}
                   for c in calls)  # 맞는 값이 여럿이면(예: 1박 한도와 2박 합계) 목록으로 적습니다
    return "judge"  # search_document 의 검색어는 정답이 하나가 아닙니다 → Judge


# ---------------------------------------------------------------- Judge 기반
QUERY_JUDGE = """당신은 사내 규정 검색어를 평가하는 평가자입니다.
사용자의 요청과 Agent 가 만든 검색어를 보고, 이 검색어로 요청에 답하는 데 필요한 규정을 찾을 수 있는지 판단하세요.
- 요청의 날씨는 get_weather, 계산은 calculator 도구가 따로 맡습니다. 검색어는 사내 규정에 관한 부분만 찾으면 됩니다.
- 요청의 핵심 주제(예: 해외 출장 일비, 법인카드 한도)가 검색어에 들어 있으면 PASS 입니다.
- 표현이 달라도 뜻이 같으면 PASS 입니다. 검색어가 길거나 단어가 더 붙어 있는 것만으로 FAIL 하지 마세요.
- 핵심 주제가 빠졌거나 엉뚱한 주제를 찾으면 FAIL 입니다.
JSON 으로만 답하세요: {"reason": "판정 근거 한 문장", "result": "PASS 또는 FAIL"}"""

TASK_JUDGE = """당신은 업무 도우미 Agent 의 최종 답변을 평가하는 평가자입니다.
사용자의 요청, 기대 결과, Agent 의 최종 답변을 보고 사용자의 목표를 이뤘는지 판단하세요.
- 요청이 묻는 것에 모두 답했고, 금액·날짜·날씨 같은 사실이 기대 결과와 맞으면 PASS 입니다.
- 기대 결과에 없는 내용이 덧붙어 있어도, 요청에 대한 답이 맞으면 PASS 입니다.
- 요청의 일부에만 답했거나, 사실이 틀렸거나, 조회할 수 있는데도 못 했다고 답하면 FAIL 입니다.
- 기대 결과가 "찾을 수 없다" 또는 "조회할 수 없다"라면, 그렇게 안내한 답이 PASS 입니다.
JSON 으로만 답하세요: {"reason": "판정 근거 한두 문장", "result": "PASS 또는 FAIL"}"""


def judge_query(task: str, query: str) -> dict:
    return call_judge(QUERY_JUDGE, f"요청: {task}\n검색어: {query}", {"reason": None, "result": ["PASS", "FAIL"]})


def judge_task_completion(task: dict, trace: dict) -> dict:
    user = f"요청: {task['task']}\n기대 결과: {task['expected_result']}\n최종 답변: {trace['final_answer']}"
    return call_judge(TASK_JUDGE, user, {"reason": None, "result": ["PASS", "FAIL"]})
