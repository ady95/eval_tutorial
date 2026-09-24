"""04-3 실습: 사람이 채점할 답변을 만들고 라벨링 시트(CSV)를 씁니다.

정답만 나오면 사람과 Judge를 비교할 수 없으므로, 일부러 작은 로컬 모델로 답변을 만듭니다.
로컬 모델이 없다면 이 단계는 건너뛰고 저장소에 있는 ch04/label_answers.json 을 그대로 쓰세요.

실행: python -m ch04.make_label_sheet --base-url http://localhost:11434/v1 --model qwen3.5:9b
"""
import argparse
import csv
import json
from pathlib import Path

from openai import OpenAI

QUESTIONS = json.loads(Path("ch04/label_questions.json").read_text(encoding="utf-8"))
ANSWERS = Path("ch04/label_answers.json")
SHEET = Path("ch04/human_labels.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen3.5:9b")
    args = parser.parse_args()

    local = OpenAI(base_url=args.base_url, api_key="ollama")  # Ollama 는 키를 확인하지 않습니다
    rows = []
    for q in QUESTIONS:
        response = local.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": "한국어로 한두 문장으로 간결하게 답하세요."},
                      {"role": "user", "content": q["question"]}],
            reasoning_effort="none",  # 추론(thinking) 모드를 끄고 바로 답하게 합니다
        )
        answer = response.choices[0].message.content.strip()
        rows.append({**q, "model": args.model, "answer": answer})
        print(f"{q['id']:2d} {answer}")

    ANSWERS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:  # 엑셀에서 한글이 깨지지 않도록 BOM 포함
        writer = csv.writer(f)
        writer.writerow(["id", "question", "reference", "answer", "label", "memo"])
        for r in rows:
            writer.writerow([r["id"], r["question"], r["reference"], r["answer"], "", ""])
    print(f"\n답변 저장: {ANSWERS}\n라벨링 시트: {SHEET}  (label 칸에 PASS 또는 FAIL 을 적으세요)")


if __name__ == "__main__":
    main()
