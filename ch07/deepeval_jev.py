"""07-4 실습: DeepEval 4.2.6 의 eval_mode 로 07-1 의 42문항을 LLM, hybrid(LLM 추출 + Jev 판정), system_one(Jev 만) 세 가지로 채점합니다.

별도 가상환경(.venv-jev)에서 실행합니다: pip install "deepeval==4.2.6" "typesafe-sdk==0.7.1"
실행: python -m ch07.deepeval_jev      (먼저 06장 python -m ch06.ragas_eval 실행)
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS", "1800")  # 07-1 의 대기 한도 문제

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from ch04.agreement import spearman
from ch07.judge_model import OpenAICompatibleJudge

rows = json.loads((Path("results") / "ch06_ragas_c400_k3.json").read_text(encoding="utf-8"))
judge = OpenAICompatibleJudge()  # llm·hybrid 모드에서 추출과 이유 작성을 맡습니다. system_one 은 쓰지 않습니다
JEV_MODEL = "jev-1.13.0"  # 04-5, 06-7 과 같은 Jev 버전으로 고정합니다 (지정하지 않으면 jev-latest)
METRICS = {"Answer Relevancy": AnswerRelevancyMetric, "Faithfulness": FaithfulnessMetric,
           "Contextual Precision": ContextualPrecisionMetric, "Contextual Recall": ContextualRecallMetric}
test_cases = [
    LLMTestCase(input=r["question"], actual_output=r["answer"], expected_output=r["reference"],
                retrieval_context=[c["text"] for c in r["retrieved"]], name=str(r["id"]))
    for r in rows
]
# 07-1 에서 DeepEval 4.2.5 로 채점한 결과(LLM 모드)를 기준으로 비교합니다
previous = {r["id"]: r["deepeval"] for r in json.loads((Path("results") / "ch07_deepeval_c400_k3.json")
                                                         .read_text(encoding="utf-8"))}

results = {}
for mode in ("llm", "hybrid", "system_one"):
    start = time.time()
    result = evaluate(
        test_cases,
        [cls(model=judge, threshold=0.7, eval_mode=mode, system_one_model=JEV_MODEL) for cls in METRICS.values()],
        async_config=AsyncConfig(max_concurrent=4),
        display_config=DisplayConfig(print_results=False, show_indicator=False, inspect_after_run=False),
        error_config=ErrorConfig(ignore_errors=True),
        cache_config=CacheConfig(write_cache=False),
    )
    scores = {int(tr.name): {md.name: md.score for md in tr.metrics_data or []} for tr in result.test_results}
    results[mode] = scores
    print(f"[{mode}] 채점 {len(scores)}문항 × 지표 {len(METRICS)}개, {time.time() - start:.0f}초")

out = Path("results") / "ch07_deepeval_jev.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def rho(a: list[float], b: list[float]) -> str:
    """42문항이 모두 같은 점수면 순위를 매길 수 없으므로 '-' 로 표시합니다."""
    return f"{spearman(a, b):.3f}" if len(set(a)) > 1 and len(set(b)) > 1 else "-"


print(f"\n  {'지표':<22}{'4.2.5 llm':>10}{'llm':>8}{'hybrid':>9}{'system_one':>12}   4.2.5 llm 과의 순위 상관(llm / hybrid / system_one)")
for name in METRICS:
    cells, rhos = [], []
    base = [previous[i].get(name) for i in sorted(previous)]
    cells.append(sum(v for v in base if v is not None) / len([v for v in base if v is not None]))
    for mode in results:
        pairs = [(results[mode][i].get(name), previous[i].get(name)) for i in sorted(results[mode])
                 if results[mode][i].get(name) is not None and previous[i].get(name) is not None]
        cells.append(sum(a for a, _ in pairs) / len(pairs))
        rhos.append(rho([a for a, _ in pairs], [b for _, b in pairs]))
    print(f"  {name:<22}" + "".join(f"{c:>9.3f}" for c in cells) + "    " + " / ".join(rhos))

print("\n모름 유형(39~42)의 Answer Relevancy, 계산 문항(35·37)의 Faithfulness")
for i in (39, 40, 41, 42, 35, 37):
    name = "Answer Relevancy" if i >= 39 else "Faithfulness"
    values = {mode: results[mode].get(i, {}).get(name) for mode in results}
    print(f"  {i}번 {name:<18}" + "  ".join(f"{mode} {'-' if v is None else f'{v:.2f}'}" for mode, v in values.items()))

print(f"\n결과 저장: {out}")
