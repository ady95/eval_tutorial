"""03-4 실습: ab_test 에서 만든 답변을 그대로 두고, 평가 기준만 '정확성'으로 바꿔 다시 판정합니다.

실행: python -m ch03.ab_rejudge      (먼저 python -m ch03.ab_test 실행)
"""
import json
from collections import Counter
from pathlib import Path

from ch03.judges import ACCURACY_CRITERION, judge_pairwise

DATASET = {d["question"]: d for d in json.loads(Path("ch03/qa_dataset.json").read_text(encoding="utf-8"))}
records = json.loads((Path("results") / "ch03_ab_test.json").read_text(encoding="utf-8"))

changed = 0
counts = Counter()
for r in records:
    verdict = judge_pairwise(r["question"], r["a"], r["b"], DATASET[r["question"]]["reference"],
                             criterion=ACCURACY_CRITERION)
    winner = verdict.get("winner", "오류")
    counts[winner] += 1
    changed += winner != r["winner"]
    print(f"[{r['winner']:>3} → {winner:>3}] {r['question']}")

n = len(records)
print(f"\nA 승 {counts['A']} / B 승 {counts['B']} / 무승부 {counts['TIE']}  (총 {n}문항)")
print(f"B의 승률 (무승부는 0.5승) = {(counts['B'] + 0.5 * counts['TIE']) / n:.2f}")
print(f"판정이 바뀐 문항: {changed}개")
