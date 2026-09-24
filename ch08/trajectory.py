"""08-4 Trajectory(실행 경로)와 Plan(계획) 지표. 모두 규칙으로 계산합니다."""
from ch08.agent_metrics import tool_calls


def trajectory_match(expected_tools: list[dict], trace: dict) -> dict:
    """도구 호출 순서를 기대 경로와 세 가지 기준으로 비교합니다."""
    expected = [t["name"] for t in expected_tools]
    actual = [c["name"] for c in tool_calls(trace)]
    return {
        "strict": actual == expected,                              # 순서와 횟수까지 같다
        "unordered": sorted(set(actual)) == sorted(set(expected)),  # 순서와 반복은 무시하고 같은 도구를 썼다
        "superset": set(expected) <= set(actual),                  # 기대한 도구를 빠짐없이 썼다 (더 써도 됨)
    }


def step_efficiency(expected_tools: list[dict], trace: dict) -> float:
    """최소로 필요한 도구 호출 수 / 실제 도구 호출 수. 기대보다 적게 부른 경우는 도구 누락(08-3)에서 봅니다."""
    optimal, actual = len(expected_tools), len(tool_calls(trace))
    return 1.0 if actual <= optimal else optimal / actual


def loops(trace: dict) -> int:
    """같은 도구를 같은 인자로 다시 부른 횟수."""
    seen, repeated = set(), 0
    for call in tool_calls(trace):
        key = (call["name"], str(sorted((call["arguments"] or {}).items())))
        repeated += key in seen
        seen.add(key)
    return repeated


def error_outcomes(trace: dict) -> dict:
    """도구 오류 뒤에 같은 도구를 다시 불러 성공했으면 회복(recovered), 아니면 막다른 길(dead end)입니다."""
    calls = tool_calls(trace)
    recovered, dead_ends = 0, 0
    for i, call in enumerate(calls):
        if call["output"].startswith("오류"):
            later = [c for c in calls[i + 1:] if c["name"] == call["name"]]
            if any(not c["output"].startswith("오류") for c in later):
                recovered += 1
            else:
                dead_ends += 1
    return {"recovered": recovered, "dead_ends": dead_ends}


def plan_quality(expected_tools: list[dict], plan: list[dict]) -> bool:
    """계획에 필요한 도구가 모두 들어 있고, 필요 없는 도구가 없는가."""
    planned = {step.get("tool") for step in plan}
    return planned == {t["name"] for t in expected_tools}


def plan_adherence(plan: list[dict], trace: dict) -> bool:
    """계획한 도구를 계획한 순서대로 불렀는가."""
    return [step.get("tool") for step in plan] == [c["name"] for c in tool_calls(trace)]
