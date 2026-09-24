"""04-2 실습 2: 내용은 같고 길이만 늘린 답변이 더 좋은 평가를 받는지 봅니다.

정답 답변 앞뒤에 사실 정보가 없는 인사말과 맺음말만 붙여 '긴 답변'을 만듭니다.
정확성 기준으로는 두 답변이 같아야 하므로, 이상적인 Judge라면 무승부나 같은 점수를 줘야 합니다.

실행: python -m ch04.verbosity_bias                    (정확성 기준)
      python -m ch04.verbosity_bias --criterion help    (03-4의 "도움이 되게" 기준)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ch03.ab_test import CRITERION as HELP_CRITERION
from ch03.judges import ACCURACY_CRITERION, judge_pairwise, judge_score

DATASET = json.loads(Path("ch03/qa_dataset.json").read_text(encoding="utf-8"))
SWAP = {"A": "B", "B": "A", "TIE": "TIE"}


def pad(text: str) -> str:
    """사실 정보 없이 길이만 늘립니다."""
    return ("좋은 질문입니다! 많은 분들이 궁금해하시는 내용이라 차근차근 설명해 드리겠습니다. "
            f"{text} "
            "이 설명이 이해하시는 데 도움이 되었기를 바랍니다. 더 궁금한 점이 있으면 언제든지 편하게 물어보세요.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--criterion", choices=["accuracy", "help"], default="accuracy")
    args = parser.parse_args()
    criterion = HELP_CRITERION if args.criterion == "help" else ACCURACY_CRITERION
    print(f"평가 기준: {criterion}\n")

    winners = Counter()
    score_diff = []
    for item in DATASET:
        q, ref = item["question"], item["reference"]
        short = next(a["text"] for a in item["answers"] if a["label"] == "정답")
        long = pad(short)
        # 두 순서로 모두 비교해 순서 편향을 상쇄합니다
        first = judge_pairwise(q, short, long, ref, criterion).get("winner")             # 짧은 답이 A
        second = SWAP.get(judge_pairwise(q, long, short, ref, criterion).get("winner"))  # 긴 답이 A → 되돌림
        for w in (first, second):
            winners[{"A": "짧은 답", "B": "긴 답", "TIE": "무승부"}.get(w, "오류")] += 1
        s_short = judge_score(q, short, ref).get("score")
        s_long = judge_score(q, long, ref).get("score")
        score_diff.append((s_short, s_long))
        print(f"  {item['id']:2d} Pairwise {first}/{second}  점수 짧은 답 {s_short} / 긴 답 {s_long}")

    n = sum(winners.values())
    print(f"\nPairwise {n}회(20문항 × 두 순서): 짧은 답 승 {winners['짧은 답']} / 긴 답 승 {winners['긴 답']} "
          f"/ 무승부 {winners['무승부']}")
    higher = sum(l > s for s, l in score_diff if s and l)
    lower = sum(l < s for s, l in score_diff if s and l)
    print(f"점수: 긴 답이 더 높음 {higher} / 더 낮음 {lower} / 같음 {len(score_diff) - higher - lower}")

    out = Path("results") / f"ch04_verbosity_bias_{args.criterion}.json"
    out.write_text(json.dumps({"winners": winners, "scores": score_diff}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
