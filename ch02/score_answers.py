"""02-3 실습: 사람이 정답·오답을 표시한 답변 세트를 네 가지 지표로 채점합니다.

실행: python -m ch02.score_answers
"""
import json
from pathlib import Path

from ch02.metrics import cosine_similarity, exact_match, f1, rouge_l

DATASET = [
    {
        "question": "대한민국의 수도는 어디인가요?",
        "reference": "대한민국의 수도는 서울입니다.",
        "answers": [
            ("정답", "서울입니다."),
            ("정답", "한국의 수도는 서울특별시입니다."),
            ("정답", "대한민국 수도는 서울이에요."),
            ("오답", "대한민국의 수도는 부산입니다."),
            ("오답", "대한민국의 수도는 서울이 아닙니다."),
        ],
    },
    {
        "question": "환불은 구매 후 며칠 이내에 가능한가요?",
        "reference": "환불은 구매 후 7일 이내에 가능합니다.",
        "answers": [
            ("정답", "구매일로부터 7일 안에 환불받을 수 있습니다."),
            ("정답", "물건을 산 날부터 일주일 안에는 돈을 돌려받을 수 있어요."),
            ("오답", "환불은 구매 후 30일 이내에 가능합니다."),
            ("오답", "환불은 구매 후 7일 이내에 불가능합니다."),
        ],
    },
    {
        "question": "RAG는 무엇인가요?",
        "reference": "RAG는 질문과 관련된 정보를 외부 문서에서 먼저 검색한 뒤, 그 내용을 바탕으로 답변을 생성하는 방식입니다.",
        "answers": [
            ("정답", "RAG는 답하기 전에 관련 자료를 찾아보고, 찾은 자료를 근거로 답을 만드는 기법입니다."),
            ("오답", "RAG는 질문과 관련된 데이터로 모델을 다시 학습시킨 뒤, 그 모델로 답변을 생성하는 방식입니다."),
        ],
    },
]


def f1_char(prediction: str, reference: str) -> float:
    """02-1에서 한국어에 더 관대했던 음절 단위 F1."""
    return f1(prediction, reference, "char")


METRICS = {
    "EM": exact_match,
    "F1": f1_char,
    "ROUGE-L": rouge_l,
    "코사인": cosine_similarity,
}


def main():
    rows = []
    for item in DATASET:
        print(f"질문: {item['question']}\n정답: {item['reference']}")
        print("  라벨    EM    F1  ROUGE-L  코사인  답변")
        for label, answer in item["answers"]:
            scores = {name: fn(answer, item["reference"]) for name, fn in METRICS.items()}
            rows.append({"question": item["question"], "label": label, "answer": answer, **scores})
            print(f"  {label}  {scores['EM']:4.2f}  {scores['F1']:4.2f}  {scores['ROUGE-L']:7.2f}"
                  f"  {scores['코사인']:6.3f}  {answer}")
        print()

    # 같은 질문 안에서 (정답, 오답) 쌍을 모두 만들어, 정답이 더 높은 점수를 받은 비율을 셉니다
    print("지표별 요약 (정답 평균 / 오답 평균 / 정답이 오답보다 높은 쌍)")
    for name in METRICS:
        good = [r[name] for r in rows if r["label"] == "정답"]
        bad = [r[name] for r in rows if r["label"] == "오답"]
        pairs = [(g, b) for q in DATASET
                 for g in (r[name] for r in rows if r["question"] == q["question"] and r["label"] == "정답")
                 for b in (r[name] for r in rows if r["question"] == q["question"] and r["label"] == "오답")]
        wins = sum(g > b for g, b in pairs)
        print(f"  {name:<8} {sum(good) / len(good):.3f} / {sum(bad) / len(bad):.3f} / {wins}/{len(pairs)}")

    out = Path("results") / "ch02_score_answers.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
