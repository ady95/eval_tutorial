"""07-2 누리솔 RAG 답변을 채점하는 G-Eval 지표들입니다. 07-3 의 pytest 테스트에서도 가져다 씁니다."""
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import SingleTurnParams

from ch07.judge_model import OpenAICompatibleJudge

judge = OpenAICompatibleJudge()

CORRECTNESS_CRITERIA = "실제 답변(actual output)이 기대 답변(expected output)과 사실 면에서 일치하는지 판단합니다."

# 05-4 에서 고친 정답 Judge 의 규칙을 평가 단계로 옮겼습니다
CORRECTNESS_STEPS = [
    "기대 답변의 핵심 사실(금액, 기한, 일수, 승인자, 가능 여부)이 실제 답변에 빠짐없이 들어 있는지 확인합니다.",
    "실제 답변에 기대 답변이나 참고 문서와 어긋나는 내용이 있으면 크게 감점합니다.",
    "기대 답변에 없는 정보라도 참고 문서로 확인되면 감점하지 않습니다.",
    "질문이 직접 묻는 내용에 결론을 내리지 않고 흐렸다면 감점합니다.",
]

POLICY_STEPS = [
    "참고 문서(retrieval context)에 질문의 답이 있는지 먼저 확인합니다.",
    "답이 없는데 추측이나 일반 상식으로 답했다면 가장 낮은 점수를 줍니다. 답이 없을 때 '규정에서 찾을 수 없습니다'라고 답했다면 정책을 지킨 것입니다.",
    "폐지된 규정이나 다른 회사의 규정을 현재 누리솔 규정처럼 안내했는지 확인합니다.",
    "'무조건', '100%', '걱정하지 마세요'처럼 규정에 없는 보장이나 단정 표현이 있는지 확인합니다.",
]

POLICY_RUBRIC = [
    Rubric(score_range=(0, 2), expected_outcome="문서에 없는 내용을 지어내거나 폐지된 규정을 현재 규정처럼 안내함"),
    Rubric(score_range=(3, 6), expected_outcome="근거는 맞지만 규정에 없는 보장·단정 표현이 섞임"),
    Rubric(score_range=(7, 10), expected_outcome="문서에 있는 내용만 안내하고, 답이 없으면 없다고 말함"),
]


def correctness_criteria(**options) -> GEval:
    """평가 기준 한 문장만 주고, 평가 단계는 DeepEval 이 만들게 합니다."""
    return GEval(
        name="Correctness",
        criteria=CORRECTNESS_CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=0.7,
        **options,
    )


def correctness_steps(**options) -> GEval:
    """평가 단계를 직접 적습니다. 참고 문서도 함께 봅니다."""
    return GEval(
        name="Correctness (steps)",
        evaluation_steps=CORRECTNESS_STEPS,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        model=judge,
        threshold=0.7,
        **options,
    )


def policy(**options) -> GEval:
    """누리솔 답변 정책 준수. 정답(expected output) 없이 채점합니다."""
    return GEval(
        name="Nurisol Policy",
        evaluation_steps=POLICY_STEPS,
        rubric=POLICY_RUBRIC,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        model=judge,
        threshold=0.7,
        **options,
    )
