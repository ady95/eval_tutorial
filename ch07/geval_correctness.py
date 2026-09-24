"""07-2 실습: 42문항을 G-Eval 정답 지표 두 가지로 두 번씩 채점하고, 05-4 재채점 판정과 비교합니다.

실행: python -m ch07.geval_correctness      (먼저 python -m ch05.rejudge 실행)
"""
import json
import os
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS", "1800")  # 07-1 의 대기 한도 문제

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase

from ch04.agreement import accuracy, cohen_kappa
from ch07.geval_metrics import correctness_criteria, correctness_steps

rows = json.loads((Path("results") / "ch05_rejudge.json").read_text(encoding="utf-8"))
labels = {r["id"]: r["correct_v2"] for r in rows}  # 05-4 문서 참고 Judge 의 PASS/FAIL
ids = sorted(labels)
truth = [labels[i] for i in ids]

test_cases = [
    LLMTestCase(input=r["question"], actual_output=r["answer"], expected_output=r["reference"],
                retrieval_context=[c["text"] for c in r["retrieved"]], name=str(r["id"]))
    for r in rows
]


def run_once() -> dict:
    """두 지표로 한 번 채점하고 {지표 이름: {문항 번호: (점수, 평가 단계)}} 를 돌려줍니다."""
    result = evaluate(
        test_cases,
        [correctness_criteria(), correctness_steps()],
        async_config=AsyncConfig(max_concurrent=4),
        display_config=DisplayConfig(print_results=False, show_indicator=False, inspect_after_run=False),
        error_config=ErrorConfig(ignore_errors=True),
        cache_config=CacheConfig(write_cache=False),
    )
    scores = {}
    for tr in result.test_results:
        for md in tr.metrics_data or []:
            steps = md.verbose_logs.split("Rubric:")[0]  # 로그 앞부분에 이 문항에 쓴 평가 단계가 있습니다
            scores.setdefault(md.name, {})[int(tr.name)] = (md.score, steps)
    return scores


def verdicts(scores: dict, threshold: float) -> list[str]:
    return ["PASS" if scores[i][0] >= threshold else "FAIL" for i in ids]


runs = [run_once(), run_once()]
out = Path("results") / "ch07_geval_correctness.json"
out.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")

for name in runs[0]:
    print(f"[{name}]")
    for n, run in enumerate(runs, 1):
        scores = run[name]
        missing = [i for i in ids if i not in scores]
        if missing:
            print(f"  {n}회 결과가 없는 문항 {missing}")
            continue
        pred = verdicts(scores, 0.7)
        wrong = [i for i, p, t in zip(ids, pred, truth) if p != t]
        mean = sum(scores[i][0] for i in ids) / len(ids)
        print(f"  {n}회 평균 {mean:.2f}  PASS {pred.count('PASS')}개  05-4 와 일치 {accuracy(pred, truth):.2f}"
              f"  κ={cohen_kappa(pred, truth):.2f}  불일치 {wrong}")
        print(f"      서로 다른 평가 단계 {len({steps for _, steps in scores.values()})}종")
    first, second = verdicts(runs[0][name], 0.7), verdicts(runs[1][name], 0.7)
    print(f"  1회와 2회에서 PASS/FAIL 이 바뀐 문항 {[i for i, a, b in zip(ids, first, second) if a != b]}\n")

print("[Correctness (steps)] threshold 별 05-4 와의 일치 (1회 / 2회)")
for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
    cells = []
    for run in runs:
        pred = verdicts(run["Correctness (steps) [GEval]"], threshold)
        cells.append(f"일치 {accuracy(pred, truth):.2f} κ={cohen_kappa(pred, truth):.2f} FAIL {pred.count('FAIL')}개")
    print(f"  {threshold:.1f}  " + "  /  ".join(cells))

print(f"\n결과 저장: {out}")
