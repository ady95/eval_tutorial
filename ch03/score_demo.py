"""03-3 실습: Rubric 유무와 정답 제공 여부에 따라 점수가 어떻게 달라지는지 봅니다.

실행: python -m ch03.score_demo
"""
import json
from pathlib import Path

from ch03.judges import judge_score

DATASET = json.loads(Path("ch03/qa_dataset.json").read_text(encoding="utf-8"))
PICK = [3, 11, 15]  # 목성, RAG, 과적합

SETTINGS = {
    "Rubric 없음": {"rubric": False, "use_reference": True},
    "Rubric 있음": {"rubric": True, "use_reference": True},
    "Rubric·정답 없음": {"rubric": True, "use_reference": False},
}

records = []
print("  라벨  " + "  ".join(f"{name:>8}" for name in SETTINGS) + "  답변")
for item in (d for d in DATASET if d["id"] in PICK):
    print(f"질문: {item['question']}")
    for ans in item["answers"]:
        scores = []
        for opt in SETTINGS.values():
            reference = item["reference"] if opt["use_reference"] else None
            result = judge_score(item["question"], ans["text"], reference, rubric=opt["rubric"])
            scores.append(result.get("score", "오류"))
            records.append({"id": item["id"], "label": ans["label"], "setting": list(SETTINGS)[len(scores) - 1],
                            "score": result.get("score"), "reason": result.get("reason")})
        print(f"  {ans['label']}  " + "  ".join(f"{s:>8}" for s in scores) + f"  {ans['text']}")

out = Path("results") / "ch03_score_demo.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"결과 저장: {out}")
