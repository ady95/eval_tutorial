"""04-3 실습: 보정한 프롬프트(v2)가 보정에 쓰지 않은 데이터에서도 잘 동작하는지 확인합니다.

03-5 의 답변 60개(미리 정해 둔 라벨)를 v2 로 판정해, v1 결과(03-5)와 비교합니다.

실행: python -m ch04.calibration_check      (먼저 python -m ch03.evaluate_all 실행)
"""
import json
from pathlib import Path

from ch03.evaluate_all import LABEL_TO_PASS
from ch04.human_agreement import judge_binary_v2

DATASET = {d["id"]: d for d in json.loads(Path("ch03/qa_dataset.json").read_text(encoding="utf-8"))}
v1_rows = json.loads((Path("results") / "ch03_evaluate_all.json").read_text(encoding="utf-8"))["rows"]

v1_correct = v2_correct = 0
changed = []
for r in v1_rows:
    item = DATASET[r["id"]]
    v2 = judge_binary_v2(item["question"], r["answer"], item["reference"]).get("result")
    expected = LABEL_TO_PASS[r["label"]]
    v1_correct += r["binary"] == expected
    v2_correct += v2 == expected
    if v2 != r["binary"]:
        changed.append((r["id"], r["label"], r["binary"], v2, r["answer"]))

print(f"미리 정해 둔 라벨과 일치: v1 {v1_correct}/{len(v1_rows)} → v2 {v2_correct}/{len(v1_rows)}")
print(f"판정이 바뀐 답변 {len(changed)}개")
for qid, label, b1, b2, answer in changed:
    print(f"  {qid:2d} {label}  v1 {b1} → v2 {b2}  {answer}")
