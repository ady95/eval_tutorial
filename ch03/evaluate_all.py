"""03-5 실습: 20개 질문, 60개 답변을 세 가지 Judge로 채점하고 02장 지표와 비교합니다.

실행: python -m ch03.evaluate_all      (Judge 호출 약 160회)
"""
import json
from collections import Counter
from pathlib import Path

from ch02.metrics import cosine_similarity, f1
from ch03.judges import judge_binary, judge_pairwise, judge_score

DATASET = json.loads(Path("ch03/qa_dataset.json").read_text(encoding="utf-8"))
LABEL_TO_PASS = {"정답": "PASS", "부분": "FAIL", "오답": "FAIL"}  # 라벨별로 기대하는 판정


def grade_answers() -> list[dict]:
    rows = []
    for item in DATASET:
        for ans in item["answers"]:
            q, a, ref = item["question"], ans["text"], item["reference"]
            binary = judge_binary(q, a, ref)
            score = judge_score(q, a, ref)
            rows.append({
                "id": item["id"], "label": ans["label"], "error": ans["error"], "answer": a,
                "binary": binary.get("result"), "binary_reason": binary.get("reason"),
                "score": score.get("score"), "score_reason": score.get("reason"),
                "attempts": binary["attempts"] + score["attempts"],
                "f1": f1(a, ref, "char"), "cosine": cosine_similarity(a, ref),
            })
            print(f"  {item['id']:2d} {ans['label']}  {rows[-1]['binary'] or '오류':4}  "
                  f"{rows[-1]['score'] or '-'}점  {a}")
    return rows


def compare_pairs(rows: list[dict]) -> list[dict]:
    """같은 질문 안에서 정답 vs 오답, 정답 vs 부분을 Pairwise로 비교합니다."""
    pairs = []
    for item in DATASET:
        text = {ans["label"]: ans["text"] for ans in item["answers"]}
        for rival in ["오답", "부분"]:
            # 정답을 A 자리에 둡니다. 순서를 바꾸면 어떻게 되는지는 04-2에서 봅니다
            verdict = judge_pairwise(item["question"], text["정답"], text[rival], item["reference"])
            pairs.append({"id": item["id"], "rival": rival, "winner": verdict.get("winner"),
                          "reason": verdict.get("reason")})
    return pairs


def pair_wins(rows: list[dict], key: str, rival: str) -> str:
    """지표 key 로 '정답 > rival' 인 질문 수."""
    wins = 0
    for item in DATASET:
        by_label = {r["label"]: r[key] for r in rows if r["id"] == item["id"]}
        wins += (by_label["정답"] or 0) > (by_label[rival] or 0)
    return f"{wins}/{len(DATASET)}"


def main():
    print("[1] 답변별 채점 (PASS/FAIL, 1~5점)")
    rows = grade_answers()
    print("\n[2] Pairwise 비교")
    pairs = compare_pairs(rows)

    print("\n[3] 요약")
    correct = sum(r["binary"] == LABEL_TO_PASS[r["label"]] for r in rows)
    print(f"PASS/FAIL 판정이 라벨과 일치: {correct}/{len(rows)}")
    for label in ["정답", "부분", "오답"]:
        sub = [r for r in rows if r["label"] == label]
        passes = sum(r["binary"] == "PASS" for r in sub)
        scores = Counter(r["score"] for r in sub)
        mean = sum(r["score"] for r in sub if r["score"]) / len(sub)
        print(f"  {label}: PASS {passes}/{len(sub)}, 평균 {mean:.2f}점, 분포 {dict(sorted(scores.items()))}")

    print("\n정답이 상대보다 높게 평가된 질문 수 (20문항)")
    print("  지표            정답>오답  정답>부분")
    for name, key in [("F1(음절)", "f1"), ("코사인", "cosine"), ("Rubric 점수", "score")]:
        print(f"  {name:<12}  {pair_wins(rows, key, '오답'):>7}  {pair_wins(rows, key, '부분'):>7}")
    for rival in ["오답", "부분"]:
        c = Counter(p["winner"] for p in pairs if p["rival"] == rival)
        print(f"  Pairwise(정답 vs {rival}): 정답 승 {c['A']}, 상대 승 {c['B']}, 무승부 {c['TIE']}")
    retries = sum(r["attempts"] - 2 for r in rows)
    print(f"\n형식 오류로 다시 호출한 횟수: {retries}회")

    out = Path("results") / "ch03_evaluate_all.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "pairs": pairs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
