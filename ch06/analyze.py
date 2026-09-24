"""06-2~06-4 실습: RAGAS 점수를 문항별로 들여다보고, 05장의 직접 구현 결과와 비교합니다. (LLM 호출 없음)

실행: python -m ch06.analyze      (먼저 python -m ch06.ragas_eval, python -m ch05.rejudge 실행)
"""
import json
from pathlib import Path

from ch04.agreement import spearman

ragas = {r["id"]: r for r in json.loads((Path("results") / "ch06_ragas_c400_k3.json").read_text(encoding="utf-8"))}
ours = {r["id"]: r for r in json.loads((Path("results") / "ch05_rejudge.json").read_text(encoding="utf-8"))}

METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy",
           "factual_correctness", "noise_sensitivity"]


def column(name):
    return [ragas[i]["ragas"][name] for i in sorted(ragas)]


# 1) 지표별 분포: 평균만 보지 않고 낮은 점수가 몇 개인지 봅니다
print("[1] 지표별 분포")
print(f"  {'지표':<20} 평균   최소   1.0 개수  0.5 미만 문항")
for name in METRICS:
    values = [(i, ragas[i]["ragas"][name]) for i in sorted(ragas) if ragas[i]["ragas"][name] is not None]
    avg = sum(v for _, v in values) / len(values)
    low = [i for i, v in values if (v > 0.5 if name == "noise_sensitivity" else v < 0.5)]
    print(f"  {name:<20} {avg:.3f}  {min(v for _, v in values):.2f}  {sum(v == 1.0 for _, v in values):4d}      {low}")
print("  (noise_sensitivity 는 낮을수록 좋으므로 0.5 초과 문항을 표시)")

# 2) 05장 직접 구현과 비교
print("\n[2] 05장 직접 구현과 비교 (문항별 Spearman 순위 상관)")
pairs = [
    ("context_recall", "evidence_recall", "근거 문장 포함"),
    ("context_precision", "context_precision", "관련성 Judge Context Precision"),
    ("faithfulness", "faithfulness_v2", "Faithfulness v2"),
    ("factual_correctness", "correct_v2", "문서 참조 Correctness (PASS=1)"),
]
for rname, oname, label in pairs:
    xs, ys = [], []
    for i in sorted(ragas):
        x = ragas[i]["ragas"][rname]
        y = ours[i][oname]
        if oname == "correct_v2":
            y = 1.0 if y == "PASS" else 0.0
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    print(f"  RAGAS {rname:<20} vs {label:<32} ρ={spearman(xs, ys):.3f}  (n={len(xs)})")

# 3) 지표끼리의 상관
print("\n[3] RAGAS 지표끼리의 Spearman 상관")
print("  " + " " * 20 + "".join(f"{n[:9]:>10}" for n in METRICS))
for a in METRICS:
    row = []
    for b in METRICS:
        xs, ys = zip(*[(ragas[i]["ragas"][a], ragas[i]["ragas"][b]) for i in sorted(ragas)
                       if ragas[i]["ragas"][a] is not None and ragas[i]["ragas"][b] is not None])
        row.append(spearman(list(xs), list(ys)))
    print(f"  {a:<20}" + "".join(f"{v:10.2f}" for v in row))

# 4) Factual Correctness 가 낮지만 문서 참조 Correctness 는 PASS 인 문항
print("\n[4] factual_correctness < 0.5 인데 문서 참조 Correctness 는 PASS 인 문항")
for i in sorted(ragas):
    fc = ragas[i]["ragas"]["factual_correctness"]
    if fc is not None and fc < 0.5 and ours[i]["correct_v2"] == "PASS":
        print(f"  {i:2d} fc={fc:.2f}  답변 {len(ragas[i]['answer'])}자 / 정답 {len(ragas[i]['reference'])}자  {ragas[i]['question']}")

# 5) 낮은 점수로 실패를 찾을 수 있을까
REAL_FAILURES = {8, 12, 39, 42}  # 05-4에서 답변을 읽고 '실제 실패'로 분류한 문항
print(f"\n[5] 낮은 점수 기준으로 문항을 골랐을 때 실제 실패({sorted(REAL_FAILURES)})가 몇 개 걸리나")
flagged_any = set()
for name in METRICS:
    bad = {i for i in ragas if ragas[i]["ragas"][name] is not None
           and (ragas[i]["ragas"][name] > 0.5 if name == "noise_sensitivity" else ragas[i]["ragas"][name] < 0.5)}
    flagged_any |= bad
    hit = bad & REAL_FAILURES
    print(f"  {name:<20} 걸린 문항 {len(bad):2d}개, 그중 실제 실패 {len(hit)}개 {sorted(hit)}")
hit = flagged_any & REAL_FAILURES
print(f"  {'하나라도 걸림':<20} 걸린 문항 {len(flagged_any):2d}개, 그중 실제 실패 {len(hit)}개 {sorted(hit)},"
      f" 놓친 실패 {sorted(REAL_FAILURES - flagged_any)}")
