"""09-4 실습: 09-2 에서 실행한 설정들을 품질·비용·지연·안정성으로 비교하고 Pareto Frontier 를 찾습니다. LLM 을 부르지 않습니다.

실행: python -m ch09.cost_quality
"""
import json
import unicodedata
from pathlib import Path
from statistics import mean

# 가정 단가(USD, 100만 토큰당). 실제 가격이 아닙니다. 쓰는 모델의 가격표로 바꿔 넣으세요.
PRICE = {"input": 1.00, "output": 8.00}
QUERIES_PER_MONTH = 2_000 * 22  # 하루 2,000건, 한 달 22 영업일로 가정

CONFIGS = {  # 표에 쓸 이름: 실행 이름들 (같은 설정을 여러 번 돌렸으면 모두 적습니다)
    "기준(k3)": ["baseline-1", "baseline-2", "baseline-3"],
    "top_k 1": ["k1"],
    "top_k 5": ["k5"],
    "top_k 10": ["k10"],
    "Reranker": ["rr"],
    "로컬 qwen3.5:9b": ["qwen"],
}
LOCAL = {"로컬 qwen3.5:9b"}  # API 요금이 없는 설정 (서버 비용은 따로 듭니다)


def pad(text: str, width: int) -> str:
    """한글처럼 두 칸을 차지하는 글자를 세어 표의 칸을 맞춥니다 (08-5 와 같은 방법)."""
    used = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - used, 0)


def load(run: str) -> dict:
    return json.loads((Path("results") / f"ch09_run_{run}.json").read_text(encoding="utf-8"))


table = []
for name, runs in CONFIGS.items():
    data = [load(run) for run in runs]
    rows = [r for d in data for r in d["rows"] if r["correct"] != "ERROR"]
    latency = sorted(r["search_ms"] + r["rerank_ms"] + r["llm_ms"] for r in rows)
    cost = 0.0 if name in LOCAL else mean(
        (r["prompt_tokens"] * PRICE["input"] + r["completion_tokens"] * PRICE["output"]) / 1_000_000 for r in rows)
    rates = [d["summary"]["pass_rate"] for d in data]
    table.append({
        "name": name, "quality": mean(rates), "quality_range": f"{min(rates):.2f}~{max(rates):.2f}" if len(rates) > 1 else "",
        "input_tokens": mean(r["prompt_tokens"] for r in rows), "output_tokens": mean(r["completion_tokens"] for r in rows),
        "cost": cost, "p50": latency[len(latency) // 2], "p95": latency[max(0, round(len(latency) * 0.95) - 1)],
        "search": mean(r["search_ms"] + r["rerank_ms"] for r in rows), "llm": mean(r["llm_ms"] for r in rows),
        "errors": sum(d["summary"]["errors"] for d in data),
    })


def dominated(a: dict, others: list[dict], key: str) -> bool:
    """다른 설정 중 품질이 같거나 높으면서 key(비용이나 지연)가 같거나 낮고, 둘 중 하나는 더 나은 것이 있으면 True."""
    return any(b["quality"] >= a["quality"] and b[key] <= a[key] and (b["quality"] > a["quality"] or b[key] < a[key])
               for b in others if b is not a)


print(f"{pad('설정', 17)}정답률            입력 토큰  출력 토큰  질의당 비용  월 비용   지연 p50  p95     검색   LLM    오류")
for t in table:
    print(f"{pad(t['name'], 17)}{t['quality']:.3f} {t['quality_range']:<10} {t['input_tokens']:7.0f} {t['output_tokens']:8.0f}"
          f"    ${t['cost']:.5f}   ${t['cost'] * QUERIES_PER_MONTH:6.1f}  {t['p50'] / 1000:5.1f}초 {t['p95'] / 1000:5.1f}초"
          f"  {t['search'] / 1000:4.1f}초 {t['llm'] / 1000:4.1f}초  {t['errors']}")

print("\nPareto Frontier (다른 설정에 완전히 밀리지 않는 설정)")
for key, label in (("cost", "품질-비용"), ("p95", "품질-지연 p95")):
    print(f"  {label}: {', '.join(t['name'] for t in table if not dominated(t, table, key))}")
