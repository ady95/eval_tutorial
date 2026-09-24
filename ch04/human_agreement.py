"""04-3 실습: 사람 라벨(ch04/human_labels.csv)과 Judge 판정이 얼마나 일치하는지 잽니다.

실행: python -m ch04.human_agreement                 (03장의 API Judge와 프롬프트)
      python -m ch04.human_agreement --prompt v2     (보정한 PASS/FAIL 프롬프트)
      python -m ch04.human_agreement --local         (04-4: 로컬 Judge, Ollama)
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from ch03 import judges
from ch04.agreement import accuracy, cohen_kappa, spearman
from ch04.calibrated import BINARY_SYSTEM_V2
from ch04.local_judge import add_local_args, apply_local

LABELS = list(csv.DictReader(Path("ch04/human_labels.csv").open(encoding="utf-8-sig")))
ANSWERS = {str(a["id"]): a for a in json.loads(Path("ch04/label_answers.json").read_text(encoding="utf-8"))}


def judge_binary_v2(question: str, answer: str, reference: str) -> dict:
    user = f"질문: {question}\n정답: {reference}\n답변: {answer}"
    return judges.call_judge(BINARY_SYSTEM_V2, user, {"reason": None, "result": ["PASS", "FAIL"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v1")
    add_local_args(parser)
    args = parser.parse_args()
    judge_name = apply_local(args)
    name = f"{judge_name}-{args.prompt}"
    binary_fn = judge_binary_v2 if args.prompt == "v2" else judges.judge_binary
    print(f"Judge: {name}\n")

    rows = []
    for lab in LABELS:
        q, ref, ans = lab["question"], lab["reference"], ANSWERS[lab["id"]]["answer"]
        start = time.time()
        b = binary_fn(q, ans, ref)
        s = judges.judge_score(q, ans, ref)
        rows.append({"id": int(lab["id"]), "human": lab["label"].strip().upper(), "memo": lab["memo"],
                     "judge": b.get("result"), "reason": b.get("reason"), "score": s.get("score"),
                     "score_reason": s.get("reason"), "seconds": round(time.time() - start, 1)})
        r = rows[-1]
        mark = "" if r["human"] == r["judge"] else "  ← 불일치"
        print(f"  {r['id']:2d} 사람 {r['human']}  Judge {r['judge'] or '오류'}  {r['score'] or '-'}점{mark}")

    human = [r["human"] for r in rows]
    judge = [r["judge"] for r in rows]
    matrix = Counter((h, j) for h, j in zip(human, judge))
    print(f"\n정확도(일치 비율): {accuracy(human, judge):.3f}  ({sum(h == j for h, j in zip(human, judge))}/{len(rows)})")
    print(f"Cohen's Kappa    : {cohen_kappa(human, judge):.3f}")
    valid = [r for r in rows if r["score"]]
    rho = spearman([1 if r["human"] == "PASS" else 0 for r in valid], [r["score"] for r in valid])
    print(f"Spearman(사람 PASS=1/FAIL=0 vs Judge 점수): {rho:.3f}")
    print(f"혼동 행렬: 사람PASS→JudgePASS {matrix['PASS', 'PASS']}, 사람PASS→JudgeFAIL {matrix['PASS', 'FAIL']}, "
          f"사람FAIL→JudgePASS {matrix['FAIL', 'PASS']}, 사람FAIL→JudgeFAIL {matrix['FAIL', 'FAIL']}")
    print(f"답변당 평균 소요(PASS/FAIL + 점수): {sum(r['seconds'] for r in rows) / len(rows):.1f}초")

    out = Path("results") / f"ch04_human_agreement_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
