"""05-3 실습: 같은 참고 문서에 대해 성격이 다른 답변 다섯 개의 Faithfulness를 주장 단위로 확인합니다.

실행: python -m ch05.faithfulness_demo
"""
from ch05.rag_judges import faithfulness

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

for name, answer in ANSWERS.items():
    result = faithfulness(answer, [CONTEXT])
    print(f"[{name}] Faithfulness = {result['score']:.2f}\n  답변: {answer}")
    for v in result["verdicts"]:
        mark = "O" if v["supported"] else "X"
        print(f"  {mark} {v['claim']}\n      └ {v['reason']}")
    print()
