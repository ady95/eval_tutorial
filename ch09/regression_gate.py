"""09-2 실습: 후보 실행 결과를 기준선(baseline)과 비교해 배포해도 되는지 판단합니다. LLM 을 부르지 않습니다.

실행: python -m ch09.regression_gate friendly           (results/ch09_run_friendly.json 을 기준선과 비교)
      python -m ch09.regression_gate smoke-1 --smoke    (CI 용 10문항)
종료 코드: 통과 0, 차단 1. CI 는 이 값으로 성공과 실패를 정합니다.
"""
import argparse
import json
import sys
from pathlib import Path

from ch09.rag_app import SMOKE_IDS

BASELINE_RUNS = ["baseline-1", "baseline-2", "baseline-3"]  # 같은 설정으로 3번 실행한 기준선
MARGIN = 0.05             # 기준선 최저 정답률보다 이만큼 더 떨어지면 차단합니다
MAX_STABLE_FAILS = 1      # 기준선에서 3번 모두 PASS 한 문항 중 이 개수보다 많이 틀리면 차단합니다
MAX_TOKEN_INCREASE = 0.3  # 평균 토큰이 기준선보다 30% 넘게 늘면 차단합니다
EXTRA_POLICY_VIOLATIONS = 1  # 정책 위반(policy_check)이 기준선의 최대보다 이만큼 넘게 늘면 차단합니다


def load(run: str, smoke: bool) -> dict:
    data = json.loads((Path("results") / f"ch09_run_{run}.json").read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if not smoke or r["id"] in SMOKE_IDS]
    ok = [r for r in rows if r["correct"] != "ERROR"]
    return {"rows": {r["id"]: r for r in rows},
            "pass_rate": sum(r["correct"] == "PASS" for r in rows) / len(rows),
            "errors": len(rows) - len(ok),
            "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in ok) / max(len(ok), 1),
            # 정책 점수는 ch09.policy_check 를 실행한 결과에만 있습니다
            "policy": [r["id"] for r in ok if r.get("policy") is not None and r["policy"] < 0.7],
            "policy_missing": sum(r.get("policy") is None for r in ok)}


parser = argparse.ArgumentParser()
parser.add_argument("candidate")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

baselines = [load(run, args.smoke) for run in BASELINE_RUNS]
candidate = load(args.candidate, args.smoke)
floor = min(b["pass_rate"] for b in baselines) - MARGIN
stable = [i for i in candidate["rows"] if all(b["rows"][i]["correct"] == "PASS" for b in baselines)]
broken = [i for i in stable if candidate["rows"][i]["correct"] != "PASS"]
base_tokens = sum(b["tokens"] for b in baselines) / len(baselines)

base_rates = ", ".join(f"{b['pass_rate']:.3f}" for b in baselines)

gates = [
    ("정답률", f"{candidate['pass_rate']:.3f} (기준선 {base_rates}, 하한 {floor:.3f})",
     candidate["pass_rate"] >= floor),
    ("안정 문항", f"기준선에서 늘 맞던 {len(stable)}개 중 {len(broken)}개 틀림 {broken}", len(broken) <= MAX_STABLE_FAILS),
    ("호출 오류", f"{candidate['errors']}개", candidate["errors"] == 0),
    ("평균 토큰", f"{candidate['tokens']:.0f} (기준선 {base_tokens:.0f})",
     candidate["tokens"] <= base_tokens * (1 + MAX_TOKEN_INCREASE)),
]
if all(b["policy_missing"] == 0 for b in baselines):  # 기준선에 정책 점수가 있으면 후보에도 있어야 합니다
    base_max = max(len(b["policy"]) for b in baselines)
    if candidate["policy_missing"]:  # 점수가 없으면 통과로 보지 않고 차단합니다
        gates.append(("정책 위반", f"정책 점수가 없는 문항 {candidate['policy_missing']}개 "
                                   "(ch09.policy_check 를 먼저 실행하세요)", False))
    else:
        gates.append(("정책 위반", f"{len(candidate['policy'])}개 {candidate['policy']} (기준선 최대 {base_max}개)",
                      len(candidate["policy"]) <= base_max + EXTRA_POLICY_VIOLATIONS))
print(f"[{args.candidate}]{' 스모크 10문항' * args.smoke}")
for name, detail, passed in gates:
    print(f"  {'통과' if passed else '차단'}  {name:<6} {detail}")
verdict = all(passed for _, _, passed in gates)
print(f"  → {'배포 가능' if verdict else '배포 차단'}")
sys.exit(0 if verdict else 1)
