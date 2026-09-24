"""01-3 실습: 같은 질문을 여러 번 던지고 답변이 얼마나 달라지는지 봅니다.

실행: python -m ch01.repeat_prompt
      python -m ch01.repeat_prompt --n 5
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from common.llm import MODEL, chat

QUESTIONS = {
    "closed": "대한민국의 수도는 어디인가요? 한 문장으로 답하세요.",
    "open": "RAG(검색 증강 생성)가 무엇인지 두 문장으로 설명하세요.",
}
REFERENCE = "서울"  # closed 질문의 정답


def run(question: str, n: int) -> list[str]:
    answers = []
    for i in range(n):
        answer = chat(question).strip()
        answers.append(answer)
        print(f"  [{i + 1:2d}] {answer}")
    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="반복 횟수")
    args = parser.parse_args()

    print(f"모델: {MODEL}\n")
    results = {}
    for key, question in QUESTIONS.items():
        print(f"질문({key}): {question}")
        answers = run(question, args.n)
        counts = Counter(answers)
        results[key] = {"question": question, "answers": answers}
        print(f"  → 서로 다른 답변 {len(counts)}개 / {args.n}회")
        print(f"  → 가장 많이 나온 답변 {counts.most_common(1)[0][1]}회")
        if key == "closed":
            exact = sum(a == REFERENCE for a in answers)
            contains = sum(REFERENCE in a for a in answers)
            print(f"  → 완전 일치 (답변 == '{REFERENCE}') : {exact}회")
            print(f"  → 포함 검사 ('{REFERENCE}' in 답변) : {contains}회")
        print()

    out = Path("results") / "ch01_repeat_prompt.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
