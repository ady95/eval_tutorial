"""05장 RAG 평가용 Judge: 검색 문서의 관련성, 답변의 주장 분해와 근거 확인(Faithfulness)."""
from ch03.judges import call_judge

# ---------------------------------------------------------------- 05-2 Context Relevance

RELEVANCE_SYSTEM = """당신은 검색 결과를 평가하는 평가자입니다.
질문과 검색된 문서 조각 하나가 주어집니다.
이 조각이 질문에 답하는 데 필요한 정보를 담고 있으면 relevant를 true, 아니면 false로 판정하세요.
주제가 비슷해도 질문의 답을 찾는 데 쓸 수 없는 내용이면 false입니다.

다음 JSON 형식으로만 답하세요. reason을 먼저 쓰세요.
{"reason": "판정 근거 한 문장", "relevant": true 또는 false}"""


def judge_relevance(question: str, chunk: str) -> dict:
    user = f"질문: {question}\n\n문서 조각:\n{chunk}"
    return call_judge(RELEVANCE_SYSTEM, user, {"reason": None, "relevant": [True, False]})


def context_precision(relevant_flags: list[bool]) -> float:
    """관련 있는 조각이 위쪽에 있을수록 높은 점수. 관련 조각이 나온 순위마다의 정밀도를 평균합니다."""
    hits, total = 0, 0.0
    for rank, flag in enumerate(relevant_flags, start=1):
        if flag:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


# ---------------------------------------------------------------- 05-3 Faithfulness

CLAIMS_SYSTEM = """당신은 답변을 분석하는 도우미입니다.
주어진 답변을 더 쪼갤 수 없는 사실 주장(claim)들로 나누세요.
- 각 주장은 다른 주장을 보지 않아도 뜻이 통하도록 주어와 조건을 포함해 완전한 문장으로 씁니다.
- 인사말이나 "규정에서 찾을 수 없습니다"처럼 사실을 주장하지 않는 문장은 빼세요. 주장이 없으면 빈 목록을 돌려줍니다.

다음 JSON 형식으로만 답하세요.
{"claims": ["주장 1", "주장 2"]}"""

VERIFY_SYSTEM = """당신은 답변이 참고 문서에 근거하는지 확인하는 평가자입니다.
참고 문서와 주장 목록이 주어집니다. 주장마다 참고 문서로 뒷받침되는지 판정하세요.
- 참고 문서에 적혀 있거나 참고 문서의 내용만으로 바로 계산·추론할 수 있으면 supported는 true입니다.
- 참고 문서에 없는 내용이면, 그 내용이 실제로 맞는지와 상관없이 supported는 false입니다.

다음 JSON 형식으로만 답하세요. 주장의 순서와 개수를 그대로 유지하세요.
{"verdicts": [{"claim": "주장 1", "reason": "근거 한 문장", "supported": true 또는 false}]}"""


def extract_claims(answer: str) -> list[str] | None:
    result = call_judge(CLAIMS_SYSTEM, f"답변: {answer}", {"claims": None})
    return result.get("claims")


def verify_claims(claims: list[str], contexts: list[str]) -> list[dict] | None:
    docs = "\n\n---\n\n".join(contexts)
    listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    result = call_judge(VERIFY_SYSTEM, f"[참고 문서]\n{docs}\n\n[주장 목록]\n{listed}", {"verdicts": None})
    verdicts = result.get("verdicts")
    if not verdicts or len(verdicts) != len(claims):
        return None
    return verdicts


def faithfulness(answer: str, contexts: list[str]) -> dict:
    """답변의 주장 중 참고 문서로 뒷받침되는 비율. 주장이 없으면 score 는 None 입니다."""
    claims = extract_claims(answer)
    if claims is None:
        return {"score": None, "claims": None, "verdicts": None, "error": "주장 분해 실패"}
    if not claims:
        return {"score": None, "claims": [], "verdicts": []}
    verdicts = verify_claims(claims, contexts)
    if verdicts is None:
        return {"score": None, "claims": claims, "verdicts": None, "error": "근거 확인 실패"}
    supported = sum(bool(v.get("supported")) for v in verdicts)
    return {"score": supported / len(claims), "claims": claims, "verdicts": verdicts}


# ---------------------------------------------------------------- 05-4 개선한 Judge

CLAIMS_SYSTEM_V2 = """당신은 답변을 분석하는 도우미입니다.
질문과 답변이 주어집니다. 답변을 더 쪼갤 수 없는 사실 주장(claim)들로 나누세요.
- 각 주장은 다른 주장을 보지 않아도 뜻이 통하도록 주어와 조건을 포함해 완전한 문장으로 씁니다.
- "해당 결제"처럼 질문에 나온 대상을 가리키는 말은 질문을 참고해 구체적으로 풀어 씁니다.
- 인사말이나 "규정에서 찾을 수 없습니다"처럼 사실을 주장하지 않는 문장은 빼세요. 주장이 없으면 빈 목록을 돌려줍니다.

다음 JSON 형식으로만 답하세요.
{"claims": ["주장 1", "주장 2"]}"""

VERIFY_SYSTEM_V2 = """당신은 답변이 참고 문서에 근거하는지 확인하는 평가자입니다.
질문, 참고 문서, 주장 목록이 주어집니다. 주장마다 참고 문서로 뒷받침되는지 판정하세요.
- 참고 문서에 적혀 있거나, 참고 문서와 질문에 주어진 정보만으로 바로 계산·추론할 수 있으면 supported는 true입니다.
- 누구나 아는 상식(예: 파리는 프랑스의 도시)이나 문서의 분류에서 곧바로 따라 나오는 결론(예: 문서가 A와 B만 따로 정하고 나머지를 '그 밖의 경우'로 정했다면, C는 '그 밖의 경우'에 해당)은 추론으로 인정합니다.
- 같은 뜻을 다르게 표현한 것은 문제 삼지 않습니다.
- 참고 문서에 없는 규정, 금액, 기한, 절차, 조건을 새로 말하면 supported는 false입니다.
- 참고 문서마다 적용 대상이 다를 수 있으니, 주장이 말하는 대상에 맞는 문서로 확인하세요.

다음 JSON 형식으로만 답하세요. 주장의 순서와 개수를 그대로 유지하세요.
{"verdicts": [{"claim": "주장 1", "reason": "근거 한 문장", "supported": true 또는 false}]}"""


def faithfulness_v2(question: str, answer: str, contexts: list[str]) -> dict:
    """질문을 함께 보고, 상식과 분류 추론을 인정하는 Faithfulness."""
    claims = call_judge(CLAIMS_SYSTEM_V2, f"질문: {question}\n답변: {answer}", {"claims": None}).get("claims")
    if claims is None:
        return {"score": None, "claims": None, "verdicts": None, "error": "주장 분해 실패"}
    if not claims:
        return {"score": None, "claims": [], "verdicts": []}
    docs = "\n\n---\n\n".join(contexts)
    listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    user = f"[질문]\n{question}\n\n[참고 문서]\n{docs}\n\n[주장 목록]\n{listed}"
    verdicts = call_judge(VERIFY_SYSTEM_V2, user, {"verdicts": None}).get("verdicts")
    if not verdicts or len(verdicts) != len(claims):
        return {"score": None, "claims": claims, "verdicts": None, "error": "근거 확인 실패"}
    supported = sum(bool(v.get("supported")) for v in verdicts)
    return {"score": supported / len(claims), "claims": claims, "verdicts": verdicts}


CORRECTNESS_WITH_DOCS = """당신은 RAG 답변의 정확성을 채점하는 평가자입니다.
질문, 정답(참고 답안), 검색된 참고 문서, 평가할 답변이 주어집니다.

판정 기준:
- 답변이 질문의 핵심에 대해 정답과 같은 사실을 말하면 PASS입니다. 표현, 말투, 길이의 차이는 무시합니다.
- 정답에 없는 부연 설명이라도 참고 문서로 확인되는 내용이면 틀린 것으로 보지 않습니다.
- 같은 대상을 가리키는 다른 표현(예: '입사일'과 '근무 시작일')은 같은 것으로 봅니다.
- 답변에 정답이나 참고 문서와 어긋나는 내용이 있으면 FAIL입니다.
- 질문이 직접 묻는 내용이 빠졌거나, 답을 흐려 사용자가 결론을 알 수 없으면 FAIL입니다.

다음 JSON 형식으로만 답하세요. reason을 먼저 쓰세요.
{"reason": "판정 근거 한두 문장", "result": "PASS 또는 FAIL"}"""


def judge_correct_with_docs(question: str, answer: str, reference: str, contexts: list[str]) -> dict:
    docs = "\n\n---\n\n".join(contexts) if contexts else "(검색된 문서 없음)"
    user = f"[질문]\n{question}\n\n[정답]\n{reference}\n\n[참고 문서]\n{docs}\n\n[답변]\n{answer}"
    return call_judge(CORRECTNESS_WITH_DOCS, user, {"reason": None, "result": ["PASS", "FAIL"]})
