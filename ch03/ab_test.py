"""03-4 실습: 시스템 프롬프트 두 개로 답변을 만들고 Pairwise Judge로 A/B 테스트를 합니다.

실행: python -m ch03.ab_test
"""
import json
from collections import Counter
from pathlib import Path

from ch03.judges import judge_pairwise
from common.llm import MODEL, client

DATASET = json.loads(Path("ch03/qa_dataset.json").read_text(encoding="utf-8"))
QUESTIONS = [d for d in DATASET if d["id"] > 10]  # 기술 질문 10개

PROMPT_A = "질문에 한두 문장으로 간결하게 답하세요."
PROMPT_B = "초보자도 이해할 수 있도록 예시를 들어 친절하고 자세하게 답하세요."
CRITERION = "질문에 더 정확하고 도움이 되게 답한 쪽을 고르세요."


def answer(system: str, question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()


def main():
    records = []
    for item in QUESTIONS:
        a, b = answer(PROMPT_A, item["question"]), answer(PROMPT_B, item["question"])
        verdict = judge_pairwise(item["question"], a, b, item["reference"], criterion=CRITERION)
        winner = verdict.get("winner", "오류")
        records.append({"question": item["question"], "a": a, "b": b, **verdict})
        print(f"[{winner:>3}] A {len(a):4d}자 / B {len(b):4d}자  {item['question']}")

    counts = Counter(r.get("winner", "오류") for r in records)
    n = len(records)
    win_rate_b = (counts["B"] + 0.5 * counts["TIE"]) / n
    print(f"\nA 승 {counts['A']} / B 승 {counts['B']} / 무승부 {counts['TIE']}  (총 {n}문항)")
    print(f"B의 승률 (무승부는 0.5승) = {win_rate_b:.2f}")
    print(f"평균 길이: A {sum(len(r['a']) for r in records) / n:.0f}자 / B {sum(len(r['b']) for r in records) / n:.0f}자")

    out = Path("results") / "ch03_ab_test.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
