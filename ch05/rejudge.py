"""05-4 실습: 저장된 RAG 답변을 개선한 Judge로 다시 채점하고, 처음 채점과 비교합니다. (답변은 다시 만들지 않습니다)

실행: python -m ch05.rejudge      (먼저 python -m ch05.evaluate_rag 실행)
"""
import json
from pathlib import Path

from ch05.rag_judges import faithfulness_v2, judge_correct_with_docs

path = Path("results") / "ch05_evaluate_rag_c400_k3.json"
rows = json.loads(path.read_text(encoding="utf-8"))

print(" id 유형     Faith v1→v2   정답 v1→v2")
for r in rows:
    contexts = [c["text"] for c in r["retrieved"]]
    f = faithfulness_v2(r["question"], r["answer"], contexts)
    c = judge_correct_with_docs(r["question"], r["answer"], r["reference"], contexts)
    r["faithfulness_v2"], r["verdicts_v2"] = f["score"], f["verdicts"]
    r["correct_v2"], r["correct_v2_reason"] = c.get("result"), c.get("reason")
    f1 = "  - " if r["faithfulness"] is None else f"{r['faithfulness']:.2f}"
    f2 = "  - " if f["score"] is None else f"{f['score']:.2f}"
    mark = "  ←" if (f1 != f2 or r["correct"] != r["correct_v2"]) else ""
    print(f"{r['id']:3d} {r['type']:<7}  {f1} → {f2}   {r['correct']} → {r['correct_v2']}{mark}")


def avg(key):
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals)


print(f"\nFaithfulness 평균: v1 {avg('faithfulness'):.2f} → v2 {avg('faithfulness_v2'):.2f}")
print(f"정답률: v1 {sum(r['correct'] == 'PASS' for r in rows) / len(rows):.2f}"
      f" → 문서 참조 {sum(r['correct_v2'] == 'PASS' for r in rows) / len(rows):.2f}")
out = Path("results") / "ch05_rejudge.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"결과 저장: {out}")
