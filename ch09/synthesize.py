"""09-1 실습: 규정 문서에서 LLM 으로 평가 문항을 합성하고, 직접 만든 평가셋과 무엇이 다른지 비교합니다.

실행: python -m ch09.synthesize          (먼저 python -m ch09.run_eval --run baseline-1 실행)
"""
import json
import random
import re
from pathlib import Path

import numpy as np

from ch03.judges import call_judge
from ch05.rag import Retriever, load_docs, load_eval_set
from ch05.rag_judges import judge_correct_with_docs
from ch09.rag_app import RagApp, RagConfig
from common.llm import embed

GENERATE = """당신은 사내 규정 챗봇의 평가 문항을 만드는 사람입니다.
아래 규정 문단만 보고 직원이 물을 만한 질문 하나와 정답을 만드세요. 정답은 문단에 있는 내용으로 한두 문장으로 씁니다.
JSON 으로만 답하세요: {"question": "질문", "answer": "정답"}"""

DISTRACTORS = {"21-domestic-travel-2025", "22-remote-work-2024", "23-affiliate-welfare"}  # 폐지·계열사 문서는 뺍니다


def words(text: str) -> list[str]:
    return [w for w in re.sub(r"[?!.,·~()\"'‘’“”]", " ", text).split() if len(w) >= 2]


def overlap(question: str, source: str) -> float:
    """질문의 어절 중 원문에 그대로 나오는 비율. 높을수록 문서의 표현을 베낀 질문입니다."""
    ws = words(question)
    return sum(w in source for w in ws) / len(ws) if ws else 0.0


# 1) 문서 청크에서 무작위로 20개를 골라 문항을 만듭니다
random.seed(0)
chunks = [c for c in Retriever(400).chunks if c.doc_id not in DISTRACTORS and "## " in c.text]
synthetic = []
for chunk in random.sample(chunks, 20):
    # 03장의 call_judge 는 JSON 형식 검증과 재시도가 들어 있어 문항 생성에도 그대로 씁니다
    made = call_judge(GENERATE, chunk.text, {"question": None, "answer": None})
    if "error" not in made:
        synthetic.append({"question": made["question"], "reference": made["answer"], "doc_id": chunk.doc_id,
                          "source": chunk.text})
print(f"합성 문항 {len(synthetic)}개")
for s in synthetic[:5]:
    print(f"  [{s['doc_id']}] {s['question']}")

# 2) 같은 RAG 로 답하고 05-4 의 Judge 로 채점합니다
app = RagApp(RagConfig())
for s in synthetic:
    out = app.answer(s["question"])
    s["top1"] = out["retrieved"][0]["doc_id"]
    s["correct"] = judge_correct_with_docs(s["question"], out["answer"], s["reference"],
                                           [c["text"] for c in out["retrieved"]]).get("result")

# 3) 직접 만든 평가셋과 비교합니다
docs = load_docs()
eval_set = load_eval_set()
baseline = {r["id"]: r for r in json.loads((Path("results") / "ch09_run_baseline-1.json").read_text(encoding="utf-8"))["rows"]}
answerable = [q for q in eval_set if q["docs"]]
print(f"\n{'':<16}{'문항':>4}  {'어절 겹침':>8}  {'1위 문서 적중':>12}  {'정답률':>6}")
rows = [
    ("합성 문항", len(synthetic), np.mean([overlap(s["question"], s["source"]) for s in synthetic]),
     np.mean([s["top1"] == s["doc_id"] for s in synthetic]), np.mean([s["correct"] == "PASS" for s in synthetic])),
    ("직접 만든 평가셋", len(answerable),
     np.mean([overlap(q["question"], " ".join(docs[d] for d in q["docs"])) for q in answerable]),
     np.mean([baseline[q["id"]]["retrieved"][0]["doc_id"] in q["docs"] for q in answerable]),
     np.mean([baseline[q["id"]]["correct"] == "PASS" for q in answerable])),
]
for name, n, ov, hit, acc in rows:
    print(f"{name:<14}{n:>6}  {ov:>8.2f}  {hit:>12.2f}  {acc:>8.2f}")
print("  (직접 만든 평가셋은 답이 규정에 있는 문항만, 결과는 09-2 기준선 1회차)")

# 4) 오염 확인: 합성 문항과 평가셋 문항이 사실상 같은 질문인지 임베딩 유사도로 봅니다
a = np.array(embed([s["question"] for s in synthetic]))
b = np.array(embed([q["question"] for q in eval_set]))
sim = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T
print("\n평가셋과 가장 비슷한 합성 문항 3개")
for i in np.argsort(-sim.max(axis=1))[:3]:
    j = int(sim[i].argmax())
    print(f"  {sim[i, j]:.3f}  합성: {synthetic[i]['question']}\n         평가셋 {eval_set[j]['id']}번: {eval_set[j]['question']}")

Path("results/ch09_synthetic.json").write_text(json.dumps(synthetic, ensure_ascii=False, indent=2), encoding="utf-8")
