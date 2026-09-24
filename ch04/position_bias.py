"""04-2 실습 1: 두 답변의 순서를 바꾸면 Pairwise 판정이 바뀌는지 봅니다.

03-5 에서는 정답을 항상 A 자리에 두었습니다. 이번에는 정답을 B 자리에 두고 다시 비교해,
두 순서의 판정이 같은 답변을 가리키는지(일관성) 셉니다.

실행: python -m ch04.position_bias      (먼저 python -m ch03.evaluate_all, python -m ch03.ab_test 실행)
      python -m ch04.position_bias --local   (04-4: 로컬 Judge)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ch03.ab_test import CRITERION
from ch03.judges import judge_pairwise
from ch04.local_judge import add_local_args, apply_local

DATASET = {d["id"]: d for d in json.loads(Path("ch03/qa_dataset.json").read_text(encoding="utf-8"))}
SWAP = {"A": "B", "B": "A", "TIE": "TIE"}


def consistency(pairs: list[tuple[str, str]]) -> str:
    """(원래 순서 판정, 뒤집은 순서 판정을 원래 기준으로 되돌린 값) 쌍에서 일치한 비율."""
    same = sum(a == b for a, b in pairs)
    return f"{same}/{len(pairs)}"


def main():
    parser = argparse.ArgumentParser()
    add_local_args(parser)
    args = parser.parse_args()
    judge_name = apply_local(args)
    print(f"Judge: {judge_name}\n")

    # 1) 03-5 의 정답 vs 부분·오답 비교를 순서만 바꿔 다시 판정
    original = json.loads((Path("results") / "ch03_evaluate_all.json").read_text(encoding="utf-8"))["pairs"]
    if args.local:
        # 03장 결과는 API Judge의 판정이므로, 로컬 Judge는 원래 순서 판정도 직접 다시 냅니다
        for p in original:
            item = DATASET[p["id"]]
            text = {a["label"]: a["text"] for a in item["answers"]}
            p["winner"] = judge_pairwise(item["question"], text["정답"], text[p["rival"]], item["reference"]).get("winner")
    results = []
    for p in original:
        item = DATASET[p["id"]]
        text = {a["label"]: a["text"] for a in item["answers"]}
        # 정답을 B 자리에 둡니다
        verdict = judge_pairwise(item["question"], text[p["rival"]], text["정답"], item["reference"])
        swapped = SWAP.get(verdict.get("winner"), "오류")  # 원래 순서 기준으로 되돌림
        results.append({"id": p["id"], "rival": p["rival"], "original": p["winner"], "swapped": swapped,
                        "reason": verdict.get("reason")})
        mark = "" if p["winner"] == swapped else "  ← 바뀜"
        print(f"  {p['id']:2d} 정답 vs {p['rival']}: 정답이 A일 때 {p['winner']:>3} / 정답이 B일 때 {swapped:>3}{mark}")

    for rival in ["오답", "부분"]:
        sub = [(r["original"], r["swapped"]) for r in results if r["rival"] == rival]
        wins = Counter(s for _, s in sub)
        print(f"정답 vs {rival}: 순서를 바꿔도 같은 판정 {consistency(sub)}, "
              f"정답이 B일 때 정답 승 {wins['A']} / 상대 승 {wins['B']} / 무승부 {wins['TIE']}")

    # 2) 03-4 A/B 테스트 답변을 순서만 바꿔 다시 판정 (도움이 되게 기준)
    ab = json.loads((Path("results") / "ch03_ab_test.json").read_text(encoding="utf-8"))
    ab_pairs = []
    for r in ab:
        ref = next(d["reference"] for d in DATASET.values() if d["question"] == r["question"])
        if args.local:  # 원래 순서(프롬프트 B 답변이 두 번째) 판정도 로컬 Judge로 다시 냅니다
            r["winner"] = judge_pairwise(r["question"], r["a"], r["b"], ref, criterion=CRITERION).get("winner")
        verdict = judge_pairwise(r["question"], r["b"], r["a"], ref, criterion=CRITERION)  # B 답변을 A 자리에
        swapped = SWAP.get(verdict.get("winner"), "오류")
        ab_pairs.append((r["winner"], swapped))
        print(f"  [{r['winner']:>3} / {swapped:>3}] {r['question']}")
    counts = Counter(s for _, s in ab_pairs)
    first = Counter(o for o, _ in ab_pairs)
    print(f"A/B 테스트: 순서를 바꿔도 같은 판정 {consistency(ab_pairs)}, "
          f"뒤집은 순서에서 프롬프트 B 승률 {(counts['B'] + 0.5 * counts['TIE']) / len(ab_pairs):.2f}")
    print(f"           원래 순서에서 프롬프트 B 승률 {(first['B'] + 0.5 * first['TIE']) / len(ab_pairs):.2f}")

    out = Path("results") / f"ch04_position_bias_{judge_name}.json"
    out.write_text(json.dumps({"pairs": results, "ab": ab_pairs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
