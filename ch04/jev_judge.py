"""04-5 실습: LLM 대신 Jev(TypeSafe System One)로 PASS/FAIL Judge 를 만들고, 04-3 의 사람 라벨과 비교합니다.

별도 가상환경(.venv-jev)에서 실행합니다: pip install "typesafe-sdk==0.7.1" python-dotenv
실행: python -m ch04.jev_judge
"""
import csv
import json
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

from ch04.agreement import accuracy, cohen_kappa, spearman

load_dotenv(".env")  # TYPESAFE_API_KEY 를 읽습니다

JEV_MODEL = "jev-1.13.0"  # 결과를 다시 만들 수 있도록 모델 버전을 고정합니다
LABELS = list(csv.DictReader(Path("ch04/human_labels.csv").open(encoding="utf-8-sig")))
ANSWERS = {str(a["id"]): a for a in json.loads(Path("ch04/label_answers.json").read_text(encoding="utf-8"))}

# 03-2 의 PASS/FAIL 기준(v1)과 04-3 에서 보정한 기준(v2)을 Jev 의 질문 형식으로 옮겼습니다
INSTRUCTIONS = {
    "v1": "답변이 정답과 같은 사실을 말하는지 판정하세요. 표현, 말투, 길이의 차이는 무시합니다. "
          "정답에 없는 부연 설명이라도 틀렸다면 FAIL입니다. 질문이 요구한 핵심 내용이 빠졌으면 FAIL입니다.",
    "v2": "답변이 정답과 같은 사실을 말하는지 판정하세요. 표현, 말투, 길이, 띄어쓰기의 차이는 무시합니다. "
          "이름이나 용어의 오탈자는 무엇을 가리키는지 분명하면 틀린 것으로 보지 않습니다. "
          "정답에 '약'이 붙은 숫자나 대략적인 값을 묻는 질문이면 근사값을 허용합니다. "
          "부연 설명을 포함한 모든 주장을 하나씩 확인해, 사실과 다른 내용이 하나라도 있으면 FAIL입니다. "
          "핵심 내용이 빠졌거나, 질문의 대상이나 범위를 좁히거나 바꿨거나, 답변 안에서 모순되면 FAIL입니다.",
}
CRITERIA = {"PASS": "정답과 같은 사실을 말하고 틀린 내용이 없다", "FAIL": "틀린 내용이 있거나 핵심이 빠졌다"}


def judge_jev(client: TypeSafeClient, question: str, answer: str, reference: str, version: str) -> dict:
    start = time.perf_counter()
    response = client.system_one(
        state={"question": question, "reference": reference, "answer": answer},
        questions={"verdict": Choice(instructions=INSTRUCTIONS[version], criteria=CRITERIA)},
        model=JEV_MODEL,
    )
    verdict = response.choices["verdict"]
    return {"result": verdict.choice, "confidence": verdict.confidence, "p_pass": verdict.probabilities["PASS"],
            "ms": round((time.perf_counter() - start) * 1000), "tokens": response.usage.input_tokens}


human = [lab["label"].strip().upper() for lab in LABELS]
runs = {}
with TypeSafeClient() as client:
    for version in ("v1", "v2"):
        runs[version] = [[judge_jev(client, lab["question"], ANSWERS[lab["id"]]["answer"], lab["reference"], version)
                          for lab in LABELS] for _ in range(3)]  # 같은 판정을 세 번 반복합니다

print(f"Jev 모델: {JEV_MODEL} · 사람 라벨 {len(LABELS)}개 (04-3)\n")
print("기준  회차  정확도  Kappa  Spearman(P(PASS))  사람과 불일치")
for version, repeats in runs.items():
    for n, rows in enumerate(repeats, 1):
        judge = [r["result"] for r in rows]
        wrong = [int(lab["id"]) for lab, h, j in zip(LABELS, human, judge) if h != j]
        rho = spearman([1 if h == "PASS" else 0 for h in human], [r["p_pass"] for r in rows])
        print(f"  {version}   {n}회   {accuracy(human, judge):.3f}  {cohen_kappa(human, judge):.3f}"
              f"       {rho:.3f}        {wrong}")
    flips = [int(lab["id"]) for i, lab in enumerate(LABELS) if len({rows[i]["result"] for rows in repeats}) > 1]
    print(f"        세 번 사이에 판정이 바뀐 답변: {flips}\n")

# 사람과 어긋난 판정의 confidence 는 맞은 판정보다 낮은가 (v2 1회차)
rows = runs["v2"][0]
right = [r["confidence"] for r, h in zip(rows, human) if r["result"] == h]
wrong = [r["confidence"] for r, h in zip(rows, human) if r["result"] != h]
print(f"confidence 평균: 사람과 일치 {sum(right) / len(right):.3f} ({len(right)}개) / "
      f"불일치 {sum(wrong) / len(wrong) if wrong else float('nan'):.3f} ({len(wrong)}개)")
low = sorted((r["confidence"], int(lab["id"]), r["result"], h) for r, lab, h in zip(rows, LABELS, human))[:5]
print("confidence 가 가장 낮은 5개 (id, Jev, 사람):",
      ", ".join(f"{i}번 {c:.2f} {j}/{h}" for c, i, j, h in low))

calls = [r for repeats in runs.values() for rows in repeats for r in rows]
ms = sorted(r["ms"] for r in calls)
print(f"\n호출 {len(calls)}번 · 지연 중앙값 {ms[len(ms) // 2]}ms · 최대 {ms[-1]}ms · "
      f"입력 토큰 평균 {sum(r['tokens'] for r in calls) / len(calls):.0f}")

Path("results/ch04_jev_judge.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
print("결과 저장: results/ch04_jev_judge.json")
