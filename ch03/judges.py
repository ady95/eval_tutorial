"""03장 LLM Judge: PASS/FAIL, 점수(Rubric), Pairwise."""
import json

from common.llm import JUDGE_MODEL, client

# Judge 호출 설정. 04-4에서 use_judge()로 로컬 모델로 바꿔 끼웁니다
JUDGE = {"client": client, "model": JUDGE_MODEL, "options": {}}


def use_judge(judge_client, model: str, **options) -> None:
    """이후 모든 Judge 호출이 이 클라이언트·모델·추가 옵션을 쓰게 합니다."""
    JUDGE.update(client=judge_client, model=model, options=options)


def call_judge(system: str, user: str, required: dict, max_retries: int = 3) -> dict:
    """Judge를 호출하고 JSON 결과를 검증합니다.

    required 는 {키: 허용값 목록} 형식입니다. 허용값이 None 이면 키가 있는지만 확인합니다.
    형식이 맞지 않으면 max_retries 번까지 다시 호출합니다.
    """
    last_error = ""
    for attempt in range(1, max_retries + 1):
        response = JUDGE["client"].chat.completions.create(
            model=JUDGE["model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            **JUDGE["options"],
        )
        text = response.choices[0].message.content or ""
        try:
            result = json.loads(text)
            if not isinstance(result, dict):  # 123, null, [] 도 올바른 JSON 이지만 판정 결과가 아닙니다
                raise ValueError(f"JSON 객체가 아님: {type(result).__name__}")
            for key, allowed in required.items():
                if key not in result:
                    raise ValueError(f"'{key}' 항목이 없음")
                if allowed is not None and result[key] not in allowed:
                    raise ValueError(f"'{key}' 값이 허용 범위 밖: {result[key]!r}")
            result["attempts"] = attempt
            return result
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e} / 원문: {text[:80]!r}"
    return {"error": last_error, "attempts": max_retries}


# ---------------------------------------------------------------- 03-2 PASS/FAIL

BINARY_SYSTEM = """당신은 답변의 정확성을 채점하는 평가자입니다.
질문, 정답(참고 답안), 평가할 답변이 주어집니다.

판정 기준:
- 답변이 정답과 같은 사실을 말하면 PASS입니다. 표현, 말투, 길이의 차이는 무시합니다.
- 답변에 사실과 다른 내용이 하나라도 있으면 FAIL입니다. 정답에 없는 부연 설명이라도 틀렸다면 FAIL입니다.
- 질문이 요구한 핵심 내용이 빠졌으면 FAIL입니다.

다음 JSON 형식으로만 답하세요. reason을 먼저 쓰고 그 근거로 result를 정하세요.
{"reason": "판정 근거 한두 문장", "result": "PASS 또는 FAIL"}"""


def judge_binary(question: str, answer: str, reference: str) -> dict:
    user = f"질문: {question}\n정답: {reference}\n답변: {answer}"
    return call_judge(BINARY_SYSTEM, user, {"reason": None, "result": ["PASS", "FAIL"]})


# ---------------------------------------------------------------- 03-3 점수·Rubric

RUBRIC = """5점: 정답과 같은 사실을 빠짐없이 말하고, 틀린 내용이 없다
4점: 핵심은 맞고 틀린 내용도 없지만, 사소한 세부가 빠졌다
3점: 핵심은 맞지만, 틀린 부연 설명이 섞여 있거나 중요한 세부가 빠졌다
2점: 일부만 맞고 핵심에 오류가 있다
1점: 핵심이 틀렸거나 질문과 관계없다"""


def judge_score(question: str, answer: str, reference: str | None = None, rubric: bool = True) -> dict:
    """1~5점으로 채점합니다. reference 가 없으면 Judge 자신의 지식으로 판단합니다."""
    system = "당신은 답변의 정확성을 채점하는 평가자입니다.\n"
    if rubric:
        system += f"다음 기준에 따라 1~5점 중 하나로 채점하세요.\n{RUBRIC}\n"
    else:
        system += "답변의 품질을 1~5점 중 하나로 채점하세요.\n"
    if reference is None:
        system += "정답은 주어지지 않습니다. 당신이 아는 사실을 기준으로 판단하세요.\n"
    system += '\n다음 JSON 형식으로만 답하세요. reason을 먼저 쓰세요.\n{"reason": "채점 근거 한두 문장", "score": 1~5 사이 정수}'

    user = f"질문: {question}\n"
    if reference is not None:
        user += f"정답: {reference}\n"
    user += f"답변: {answer}"
    return call_judge(system, user, {"reason": None, "score": [1, 2, 3, 4, 5]})


# ---------------------------------------------------------------- 03-4 Pairwise

PAIRWISE_SYSTEM = """당신은 두 답변을 비교하는 평가자입니다.
질문과 답변 A, 답변 B가 주어집니다. 정답(참고 답안)이 주어지면 그것을 사실의 기준으로 삼습니다.

{criterion}
두 답변이 똑같이 좋거나 똑같이 나쁘면 TIE를 고르세요.

다음 JSON 형식으로만 답하세요. reason을 먼저 쓰세요.
{{"reason": "비교 근거 한두 문장", "winner": "A, B, TIE 중 하나"}}"""

ACCURACY_CRITERION = "질문에 더 정확하게 답한 쪽을 고르세요. 길이나 말투는 고려하지 마세요."


def judge_pairwise(question: str, answer_a: str, answer_b: str,
                   reference: str | None = None, criterion: str = ACCURACY_CRITERION) -> dict:
    user = f"질문: {question}\n"
    if reference is not None:
        user += f"정답: {reference}\n"
    user += f"답변 A: {answer_a}\n답변 B: {answer_b}"
    return call_judge(PAIRWISE_SYSTEM.format(criterion=criterion), user,
                      {"reason": None, "winner": ["A", "B", "TIE"]})
