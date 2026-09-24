"""02-1 실습: 문자열 기반 지표 네 가지로 같은 답변을 채점합니다.

실행: python -m ch02.string_metrics
"""
from rouge_score import rouge_scorer

from ch02.metrics import bleu, exact_match, f1, rouge_l

REFERENCE = "대한민국의 수도는 서울입니다."
PREDICTIONS = [
    "대한민국의 수도는 서울입니다",
    "서울입니다.",
    "한국의 수도는 서울특별시입니다.",
    "대한민국의 수도는 부산입니다.",
]

print(f"정답: {REFERENCE}\n")
print("  EM  F1어절 F1음절  BLEU ROUGE-L  답변")
for p in PREDICTIONS:
    print(f"{exact_match(p, REFERENCE):4.2f}  {f1(p, REFERENCE, 'word'):5.2f}  {f1(p, REFERENCE, 'char'):5.2f}"
          f"  {bleu(p, REFERENCE):4.2f}  {rouge_l(p, REFERENCE):5.2f}  {p}")

# 비교용: rouge-score 기본 토크나이저를 그대로 쓰면?
default = rouge_scorer.RougeScorer(["rougeL"])
score = default.score(REFERENCE, PREDICTIONS[0])["rougeL"].fmeasure
print(f"\n[참고] rouge-score 기본 토크나이저로 계산한 첫 답변의 ROUGE-L: {score:.2f}")
