"""07-2 실습: 평가 기준 한 문장으로 G-Eval 을 만들고, DeepEval 이 만든 평가 단계를 확인합니다.

실행: python -m ch07.geval_first      (먼저 python -m ch07.deepeval_rag 실행)
"""
import json
from pathlib import Path

from deepeval.test_case import LLMTestCase

from ch07.geval_metrics import correctness_criteria

rows = {r["id"]: r for r in json.loads((Path("results") / "ch07_deepeval_c400_k3.json").read_text(encoding="utf-8"))}
r = rows[3]  # 연차 이월 질문. 05-4 에서 처음에 FAIL, 재채점에서 PASS 가 된 문항
test_case = LLMTestCase(input=r["question"], actual_output=r["answer"], expected_output=r["reference"])
print(f"질문: {r['question']}\n답변: {r['answer']}\n정답: {r['reference']}")

# 같은 기준으로 지표를 세 번 새로 만들어 채점합니다
for i in range(1, 4):
    metric = correctness_criteria(async_mode=False)
    metric.measure(test_case)
    print(f"\n[{i}회] 점수 {metric.score:.2f}  {metric.reason}")
    for n, step in enumerate(metric.evaluation_steps, 1):
        print(f"  단계 {n}. {step}")
