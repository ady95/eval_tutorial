"""07-1 실습: 05-3 의 답변 다섯 개로 우리 Faithfulness 와 DeepEval Faithfulness 의 판정 방식을 비교합니다.

실행: python -m ch07.faithfulness_compare
"""
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from ch05.rag_judges import faithfulness
from ch07.judge_model import OpenAICompatibleJudge

# 05-3 faithfulness_demo.py 와 같은 문서와 답변입니다
CONTEXT = """[국내 출장비 규정]
## 4. 일비와 숙박비
- 일비는 1일 3만 원이며, 식비와 잡비를 포함한다. 당일 출장도 일비를 지급한다.
- 숙박비는 1박당 서울·부산 12만 원, 그 밖의 지역 10만 원까지 실비로 지급한다."""

ANSWERS = {
    "충실한 답": "부산은 1박당 12만 원까지 숙박비가 나오고, 일비는 하루 3만 원입니다.",
    "지어낸 내용": "부산은 1박당 12만 원까지 나오며, 숙박 영수증은 출장 후 7일 이내에 제출해야 합니다.",
    "문서와 반대": "부산은 그 밖의 지역에 해당해 1박당 10만 원까지 숙박비가 나옵니다.",
    "잘못된 추론": "일비는 하루 3만 원이므로 2박 3일 출장이면 일비는 모두 6만 원입니다.",
    "문서 밖 상식": "부산은 1박당 12만 원까지 나옵니다. 부산은 대한민국에서 두 번째로 큰 도시입니다.",
}

judge = OpenAICompatibleJudge()


def deepeval_faithfulness(answer: str, strict: bool) -> FaithfulnessMetric:
    metric = FaithfulnessMetric(model=judge, penalize_ambiguous_claims=strict, async_mode=False)
    metric.measure(LLMTestCase(input="출장비 규정을 알려 주세요.", actual_output=answer, retrieval_context=[CONTEXT]))
    return metric


print(f"  {'답변':<10} 05-3 구현  DeepEval  DeepEval(strict)")
details = {}
for name, answer in ANSWERS.items():
    ours = faithfulness(answer, [CONTEXT])["score"]
    default = deepeval_faithfulness(answer, strict=False)
    strict = deepeval_faithfulness(answer, strict=True)
    details[name] = (default, strict)
    print(f"  {name:<10} {ours:8.2f}  {default.score:8.2f}  {strict.score:12.2f}")

# DeepEval 은 주장마다 yes(일치) / no(모순) / borderline(문서로 판단할 수 없음) 중 하나를 붙입니다.
# 기본값은 yes 와 borderline 을 통과로 세고, strict 는 yes 만 통과로 셉니다.
# 두 채점은 주장 추출부터 따로 하므로 주장 목록이 다를 수 있습니다.
for name, metrics in details.items():
    for label, metric in zip(("기본", "strict"), metrics):
        print(f"\n[{name}] DeepEval({label}) 판정")
        for claim, verdict in zip(metric.claims, metric.verdicts):
            print(f"  {verdict.verdict:<10} {claim}")
