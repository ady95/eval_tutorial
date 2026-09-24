"""04-2 실습 3: 같은 답변을 여러 번 채점해 판정과 점수가 얼마나 흔들리는지 봅니다.

실행: python -m ch04.repeat_judge
      python -m ch04.repeat_judge --n 5
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from ch03.judges import judge_binary, judge_score

DATASET = {d["id"]: d for d in json.loads(Path("ch03/qa_dataset.json").read_text(encoding="utf-8"))}
# (질문 id, 라벨): 03-5 에서 판정이 애매했던 답변과, 비교용으로 분명한 답변
TARGETS = [(2, "부분"), (6, "부분"), (8, "부분"), (7, "부분"), (13, "부분"), (3, "정답"), (3, "오답")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="반복 횟수")
    args = parser.parse_args()

    records = []
    print(f"같은 답변을 {args.n}번씩 채점합니다.\n")
    print(" id 라벨   PASS/FAIL        점수 분포          평균  표준편차  답변")
    for qid, label in TARGETS:
        item = DATASET[qid]
        answer = next(a["text"] for a in item["answers"] if a["label"] == label)
        binaries, scores = [], []
        for _ in range(args.n):
            binaries.append(judge_binary(item["question"], answer, item["reference"]).get("result"))
            scores.append(judge_score(item["question"], answer, item["reference"]).get("score"))
        valid = [s for s in scores if s]
        b = Counter(binaries)
        dist = dict(sorted(Counter(valid).items()))
        mean, stdev = statistics.mean(valid), statistics.pstdev(valid)
        records.append({"id": qid, "label": label, "answer": answer, "binary": binaries, "scores": scores})
        print(f"{qid:3d} {label}  PASS {b['PASS']:2d} FAIL {b['FAIL']:2d}  {str(dist):<20} "
              f"{mean:5.2f}  {stdev:6.2f}   {answer}")

    out = Path("results") / "ch04_repeat_judge.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
