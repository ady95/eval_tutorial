"""06-3 실습: 정답 문장에 주어가 있느냐 없느냐에 따라 Factual Correctness가 어떻게 달라지는지 봅니다.

실행: python -m ch06.fc_reference_demo
"""
import asyncio

from ragas.metrics.collections import FactualCorrectness

from ch06.setup import ragas_llm

CASES = [
    {
        "response": "고객 개인정보는 계약 종료 후 5년 동안 보관한 뒤 파기합니다.",
        "references": {
            "주어 생략": "계약 종료 후 5년 동안 보관한 뒤 파기합니다.",
            "주어 포함": "고객 개인정보는 계약 종료 후 5년 동안 보관한 뒤 파기합니다.",
        },
    },
    {
        "response": "컴퓨터 화면은 5분 동안 입력이 없으면 자동으로 잠깁니다.",
        "references": {
            "주어 생략": "5분 동안 입력이 없으면 자동으로 잠깁니다.",
            "주어 포함": "컴퓨터 화면은 5분 동안 입력이 없으면 자동으로 잠깁니다.",
        },
    },
]


async def main():
    metric = FactualCorrectness(llm=ragas_llm())
    for case in CASES:
        print(f"답변: {case['response']}")
        for label, reference in case["references"].items():
            scores = [(await metric.ascore(response=case["response"], reference=reference)).value for _ in range(3)]
            print(f"  [{label}] {reference}\n      3회 점수: {scores}")
        print()


asyncio.run(main())
