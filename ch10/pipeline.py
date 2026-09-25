"""10-3 프로젝트 3: 자동 평가 파이프라인. 실행 → 평가 엔진(규칙 + Judge + 지표) → 기준선 비교 → 배포/거부 → 기록.

실행: python -m ch10.pipeline --candidate after --run-args "--top-k 5 --prompt v2"   (새로 실행하고 판단)
      python -m ch10.pipeline --candidate friendly                                  (09-2 의 실행 결과로 판단)
종료 코드: 배포 0, 거부 1
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("--candidate", required=True, help="후보 실행 이름 (results/ch09_run_<이름>.json)")
parser.add_argument("--run-args", help="주면 ch09.run_eval 과 ch09.policy_check 를 먼저 실행합니다")
parser.add_argument("--config", default="ch10/pipeline_config.json")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text(encoding="utf-8"))


# 1) 실행: 09-2 의 스크립트를 그대로 부릅니다. CI 에서 명령을 차례로 실행하는 것과 같습니다
if args.run_args:
    for command in (f"-m ch09.run_eval --run {args.candidate} {args.run_args}", f"-m ch09.policy_check {args.candidate}"):
        subprocess.run([sys.executable, *shlex.split(command)], check=True, stdout=subprocess.DEVNULL)


def pad(text: str, width: int) -> str:
    """한글처럼 두 칸을 차지하는 글자를 세어 표의 칸을 맞춥니다 (08-5 와 같은 방법)."""
    used = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - used, 0)


def load(run: str) -> list[dict]:
    return json.loads((Path("results") / f"ch09_run_{run}.json").read_text(encoding="utf-8"))["rows"]


# 2) 평가 엔진: 검사 하나는 "실행 결과(rows) → 숫자" 함수입니다. 규칙은 LLM 없이, Judge 는 저장된 판정을 씁니다
def sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def measure(check: dict, rows: list[dict]) -> float:
    ok = [r for r in rows if r["correct"] != "ERROR"]
    rule = check["rule"]
    if rule == "pass_rate":
        return sum(r["correct"] == "PASS" for r in rows) / len(rows)
    if rule == "count_below":
        return sum(r.get(check["field"]) is not None and r[check["field"]] < check["threshold"] for r in ok)
    if rule == "not_found_phrase":  # 규정에 답이 없는 질문에는 정해진 문구로 답해야 합니다(05장 프롬프트)
        return sum(r["type"] == "none" and check["phrase"] not in r["answer"] for r in ok)
    if rule == "latin_words":       # 한국어 답변에 영어 소문자 단어가 섞였는가 (KTX 같은 대문자 약어는 제외)
        return sum(bool(re.search(r"\b[a-z]{3,}\b", r["answer"])) for r in ok)
    if rule == "long_answer":       # "두세 문장으로" 지시를 어겼는가
        return sum(sentences(r["answer"]) > check["max_sentences"] for r in ok)
    if rule == "errors":
        return len(rows) - len(ok)
    if rule == "monthly_cost":
        price = check["price_per_1m"]
        per_query = mean((r["prompt_tokens"] * price["input"] + r["completion_tokens"] * price["output"]) / 1e6 for r in ok)
        return per_query * check["queries_per_month"]
    if rule == "latency_p95":
        latency = sorted(r["search_ms"] + r["rerank_ms"] + r["llm_ms"] for r in ok)
        return latency[max(0, round(len(latency) * 0.95) - 1)]
    raise ValueError(f"알 수 없는 검사: {rule}")


def judge(check: dict, value: float, base: list[float]) -> bool:
    """후보의 값과 기준선의 값으로 통과 여부를 정합니다."""
    if check["rule"] == "pass_rate":
        return value >= min(base) - check["max_drop"]
    if check["rule"] == "stable_fails":
        return value <= check["max"]
    if "extra" in check:  # 위반 개수: 기준선에서 나온 최대 개수보다 extra 개 넘게 늘면 안 됩니다
        return value <= max(base) + check["extra"]
    if "max" in check:
        return value <= check["max"]
    return value <= check["max_usd"]


baselines = [load(run) for run in config["baseline"]]
candidate = load(args.candidate)
stable = {r["id"] for r in candidate if all(next(b for b in run if b["id"] == r["id"])["correct"] == "PASS" for run in baselines)}

results = []
for check in config["checks"]:
    if check["rule"] == "stable_fails":
        broken = sorted(r["id"] for r in candidate if r["id"] in stable and r["correct"] != "PASS")
        value, base, detail = len(broken), [0, 0, 0], f"{broken}"
    else:
        value, base, detail = measure(check, candidate), [measure(check, run) for run in baselines], ""
    passed = judge(check, value, base)
    if check.get("field", "correct") != "correct":  # 저장된 Judge 점수가 빠진 문항이 있으면 통과로 보지 않습니다
        missing = sum(r.get(check["field"]) is None for r in candidate if r["correct"] != "ERROR")
        if missing:
            passed, detail = False, f"점수 없는 문항 {missing}개"
    results.append({"name": check["name"], "kind": check["kind"], "value": round(value, 3),
                    "baseline": [round(b, 3) for b in base], "passed": passed, "detail": detail})

# 3) 판단과 기록: 결과를 표로 보여 주고, 판단 이력을 파일에 쌓습니다
decision = all(r["passed"] for r in results)
print(f"[{args.candidate}]\n        {pad('검사', 20)}{pad('종류', 8)}{pad('후보', 10)}기준선 3회")
for r in results:
    print(f"  {'통과' if r['passed'] else '거부'}  {pad(r['name'], 20)}{pad(r['kind'], 8)}{pad(str(r['value']), 10)}"
          f"{r['baseline']} {r['detail']}")
print(f"  → {'배포' if decision else '거부'}")

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
with open(Path("results") / "ch10_pipeline_history.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "commit": commit or None,
                        "candidate": args.candidate, "decision": "deploy" if decision else "reject",
                        "failed": [r["name"] for r in results if not r["passed"]]}, ensure_ascii=False) + "\n")
sys.exit(0 if decision else 1)
