"""09-2 실습: 저장된 실행 결과의 답변에 07-2 의 답변 정책 지표(Nurisol Policy)를 매겨 결과 파일에 더합니다.

정답과 비교하는 채점만으로는 "문서에 없는 내용을 덧붙이는" 퇴행을 놓칠 수 있어, 게이트에 정책 지표를 함께 씁니다.
실행: python -m ch09.policy_check baseline-1 baseline-2 baseline-3 friendly qwen
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS", "1800")  # 07-1 의 대기 한도 문제

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase

from ch07.geval_metrics import policy

for run in sys.argv[1:]:
    path = Path("results") / f"ch09_run_{run}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["correct"] != "ERROR"]
    result = evaluate(
        [LLMTestCase(input=r["question"], actual_output=r["answer"],
                     retrieval_context=[c["text"] for c in r["retrieved"]], name=str(r["id"])) for r in rows],
        [policy()],
        async_config=AsyncConfig(max_concurrent=4),
        display_config=DisplayConfig(print_results=False, show_indicator=False, inspect_after_run=False),
        error_config=ErrorConfig(ignore_errors=True),
        cache_config=CacheConfig(write_cache=False),
    )
    scores = {int(tr.name): tr.metrics_data[0] for tr in result.test_results if tr.metrics_data}
    for r in rows:
        md = scores.get(r["id"])
        r["policy"], r["policy_reason"] = (md.score, md.reason) if md else (None, None)
    violations = sorted(r["id"] for r in rows if r.get("policy") is not None and r["policy"] < 0.7)
    data["summary"]["policy_violations"] = violations
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{run:<12} 정책 점수 0.7 미만 {len(violations)}개 {violations}")
