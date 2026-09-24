"""09-3 실습: 09-2 의 RAG 실패와 08장 v0 Agent 의 실패를 Judge 로 유형 분류하고, 규칙 판정과 대조해 우선순위를 정합니다.

실행: python -m ch09.failure_taxonomy                 (먼저 09-2 의 run_eval 실행들과 08장 evaluate_tools 실행)
      python -m ch09.failure_taxonomy --preview 150   (문서·도구 결과를 150자까지만 보여 준 처음 버전)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ch03.judges import call_judge
from ch08.agent_metrics import tool_calls

CATEGORIES = {
    "Retrieval Error": "필요한 문서를 검색하지 못해 답하지 못했거나 틀렸다",
    "Hallucination": "검색된 문서나 도구 결과에 없는 내용을 지어내거나 추측했다",
    "Incomplete Answer": "근거는 있었지만 질문의 일부에만 답했거나 결론을 흐렸다",
    "Tool Error": "도구에 잘못된 인자를 넘겼거나, 도구 오류를 처리하지 못했다",
    "Planning Error": "필요한 단계를 건너뛰었거나 불필요한 단계를 넣었다(예: 계산기를 써야 하는데 암산)",
    "Format Error": "내용은 맞지만 요구한 형식(두세 문장, 정해진 문구 등)을 어겼다",
    "Evaluation Error": "실제로는 맞는 답인데 채점이 잘못됐다",
}
# 실패 한 건의 비용(가정): 직원에게 틀린 정보를 주는 실패를 가장 비싸게 봅니다
SEVERITY = {"Hallucination": 3, "Retrieval Error": 2, "Tool Error": 2, "Incomplete Answer": 1,
            "Planning Error": 1, "Format Error": 1, "Evaluation Error": 0}

CLASSIFY = "당신은 LLM 서비스의 실패 원인을 분류하는 평가자입니다. 아래 실패 사례를 가장 알맞은 유형 하나로 분류하세요.\n\n" + \
    "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items()) + \
    '\n\n여러 유형에 걸치면 가장 먼저 일어난 원인을 고르세요. JSON 으로만 답하세요: {"reason": "근거 한두 문장", "category": "유형 이름"}'

parser = argparse.ArgumentParser()
parser.add_argument("--preview", type=int, default=None, help="Judge 에게 보여 줄 문서·도구 결과의 글자 수 (기본: 전체)")
args = parser.parse_args()

RAG_RUNS = ["baseline-1", "baseline-2", "baseline-3", "friendly", "qwen", "k1", "k5", "k10", "rr"]


def rag_failures() -> list[dict]:
    cases = []
    for run in RAG_RUNS:
        rows = json.loads((Path("results") / f"ch09_run_{run}.json").read_text(encoding="utf-8"))["rows"]
        for r in rows:
            if r["correct"] != "FAIL":
                continue
            docs = "\n".join(f"- {c['doc_id']}: {c['text'][:args.preview]}" for c in r["retrieved"])
            text = (f"[RAG 질문] {r['question']}\n[정답] {r['reference']}\n[정답이 있는 문서] {r['docs'] or '없음(규정에 없는 질문)'}\n"
                    f"[검색된 문서]\n{docs}\n[답변] {r['answer']}\n[채점 Judge 의 FAIL 이유] {r['reason']}")
            # 규칙 판정: 정답 문서를 하나도 못 찾았으면 검색 실패가 확실합니다
            rule = "Retrieval Error" if r["doc_recall"] == 0 else None
            cases.append({"source": f"RAG {run}", "id": r["id"], "text": text, "rule": rule})
    return cases


def agent_failures() -> list[dict]:
    evals = {e["id"]: e for e in json.loads(Path("results/ch08_eval_v0.json").read_text(encoding="utf-8"))[0]}
    rows = json.loads(Path("results/ch08_traces_v0_run1.json").read_text(encoding="utf-8"))
    cases = []
    for r in rows:
        e = evals[r["id"]]
        if e["completed"] and e["tool_correct"] and e["arguments_ok"]:
            continue
        calls = "\n".join(f"- {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)}) → {c['output'][:args.preview]}"
                          for c in tool_calls(r["trace"]))
        text = (f"[Agent 요청] {r['task']}\n[오늘 날짜] 2026-10-05(월)\n[기대 도구] {[t['name'] for t in r['expected_tools']]}\n"
                f"[실제 도구 호출]\n{calls or '(없음)'}\n[최종 답변] {r['trace']['final_answer']}")
        # 규칙 판정: 08-3 의 인자 검사와 도구 선택 검사
        rule = "Tool Error" if not e["arguments_ok"] else "Planning Error" if e["missing"] else None
        cases.append({"source": "Agent v0", "id": r["id"], "text": text, "rule": rule})
    return cases


cases = rag_failures() + agent_failures()
for case in cases:
    result = call_judge(CLASSIFY, case["text"], {"reason": None, "category": list(CATEGORIES)})
    case["category"], case["reason"] = result.get("category", "분류 실패"), result.get("reason")

print(f"실패 {len(cases)}건 (RAG {sum(c['source'].startswith('RAG') for c in cases)}, Agent {sum(c['source'] == 'Agent v0' for c in cases)})\n")
print(f"{'출처':<14}" + "".join(f"{name.split()[0]:>11}" for name in CATEGORIES))
for source in dict.fromkeys(c["source"] for c in cases):
    counts = Counter(c["category"] for c in cases if c["source"] == source)
    print(f"{source:<14}" + "".join(f"{counts[name]:>11}" for name in CATEGORIES))

checked = [c for c in cases if c["rule"]]
agree = [c for c in checked if c["category"] == c["rule"]]
print(f"\n규칙으로 원인이 확실한 {len(checked)}건 중 Judge 분류가 일치한 것 {len(agree)}건")
for c in checked:
    if c["category"] != c["rule"]:
        print(f"  {c['source']} {c['id']}번: 규칙 {c['rule']} / Judge {c['category']} — {c['reason']}")

# 최종 분류: 규칙으로 원인이 확실하면 규칙을, 그렇지 않으면 Judge 의 분류를 씁니다
for c in cases:
    c["final"] = c["rule"] or c["category"]
total = Counter(c["final"] for c in cases)
print("\n최종 분류(규칙 우선)")
print("유형        건수  비용(가정)  건수×비용")
def weighted(item: tuple[str, int]) -> int:
    name, count = item
    return count * SEVERITY.get(name, 0)


for name, count in sorted(total.items(), key=weighted, reverse=True):
    print(f"  {name:<18} {count:3d}  {SEVERITY.get(name, 0):6d}  {count * SEVERITY.get(name, 0):8d}")

Path(f"results/ch09_failures{f'_preview{args.preview}' if args.preview else ''}.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
