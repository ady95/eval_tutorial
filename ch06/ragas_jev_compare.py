"""06-7 실습: 06장의 RAG 결과 42문항을 RAGAS-JEV 로 채점하고, RAGAS 0.4.3 점수와 비교합니다.

별도 가상환경(.venv-jev)에서 실행합니다: pip install "ragas-jev==0.2.0"
실행: python -m ch06.ragas_jev_compare prepare     (ragas-jev 입력 JSONL 만들기)
      ragas-jev evaluate -i results/ch06_jev_input.jsonl -o results/ch06_jev_run1.jsonl
      python -m ch06.ragas_jev_compare compare run1   (06-2·06-3 의 RAGAS 결과와 비교)
"""
import json
import sys
from pathlib import Path
from statistics import mean

from ch04.agreement import spearman

RAGAS = Path("results") / "ch06_ragas_c400_k3.json"  # 06-2 에서 RAGAS 점수를 붙여 저장한 결과
INPUT = Path("results") / "ch06_jev_input.jsonl"
# RAGAS-JEV 지표 → 06장 RAGAS 지표
PAIRS = {"context_precision": "context_precision", "context_recall": "context_recall",
         "faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy"}


def prepare() -> None:
    rows = json.loads(RAGAS.read_text(encoding="utf-8"))
    with INPUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"sample_id": str(r["id"]), "question": r["question"], "answer": r["answer"],
                                "contexts": [c["text"] for c in r["retrieved"]], "reference": r["reference"],
                                "metadata": {"type": r["type"]}}, ensure_ascii=False) + "\n")
    print(f"{len(rows)}문항 → {INPUT}")


def load_run(run: str) -> list[dict]:
    lines = (Path("results") / f"ch06_jev_{run}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def jev_scores(rows: list[dict]) -> dict:
    """RAGAS-JEV 결과에서 문항별 점수를 꺼냅니다. 점수는 retrieval / generation 아래에 있습니다.
    기본 점수 말고도 context_precision_ap(순위 가중 AP) 같은 변형 점수가 함께 들어 있습니다."""
    return {int(r["sample_id"]): {**(r.get("retrieval") or {}), **(r.get("generation") or {})} for r in rows}


def check_run(rows: list[dict]) -> None:
    """종료 코드가 0이어도 확인할 것: 오류, 비어 있는 지표, 재검토 실패, 사람 검토 표시."""
    status = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    errors = [int(r["sample_id"]) for r in rows if r.get("error")]
    failures = sum((r.get("escalation") or {}).get("routing_failures", 0) for r in rows)
    review = [f"{r['sample_id']}번 {u['metric']}" for r in rows for u in r["units"]
              if (u.get("decision") or {}).get("needs_human_review")]
    print(f"점검: 상태 {status}, 오류 문항 {errors}, 재검토 실패 {failures}건, 사람 검토 표시 {review}")


def rule_ap(row: dict) -> float:
    """정답 문서(docs)로 계산한 검색 AP. 정답 문서에서 나온 청크가 위에 있을수록 높습니다 (05-2)."""
    hits, total = 0, 0.0
    for rank, chunk in enumerate(row["retrieved"], 1):
        if chunk["doc_id"] in row["docs"]:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def compare(run: str) -> None:
    ragas = {r["id"]: r for r in json.loads(RAGAS.read_text(encoding="utf-8"))}
    rows = load_run(run)
    jev = jev_scores(rows)
    print(f"RAGAS-JEV {run} · {len(jev)}문항")
    check_run(rows)
    # 점수가 빈(None) 문항은 빼고 평균을 내므로, 지표마다 문항 수 n 이 다를 수 있습니다
    print(f"\n  {'지표':<18} n   RAGAS 평균  RAGAS-JEV 평균  순위 상관 ρ  0.5 이상 차이 나는 문항")
    for name, ragas_name in PAIRS.items():
        pairs = [(i, jev[i][name], ragas[i]["ragas"][ragas_name]) for i in sorted(jev)
                 if jev[i][name] is not None and ragas[i]["ragas"][ragas_name] is not None]
        far = [i for i, a, b in pairs if abs(a - b) >= 0.5]
        print(f"  {name:<18} {len(pairs):<3} {mean(b for _, _, b in pairs):.3f}       {mean(a for _, a, _ in pairs):.3f}"
              f"           {spearman([a for _, a, _ in pairs], [b for _, _, b in pairs]):.3f}      {far}")

    # 어느 쪽이 맞는지는 서로 비교해서는 알 수 없으므로, 규칙으로 계산한 기준과 대조합니다 (답이 규정에 있는 문항만)
    ids = [i for i in sorted(jev) if ragas[i]["docs"]]
    truth = {"검색 AP(정답 문서 기준)": [rule_ap(ragas[i]) for i in ids],
             "근거 포함률(05-2)": [ragas[i]["evidence_recall"] for i in ids]}
    candidates = [("검색 AP(정답 문서 기준)", "RAGAS context_precision", [ragas[i]["ragas"]["context_precision"] for i in ids]),
                  ("검색 AP(정답 문서 기준)", "RAGAS-JEV context_precision", [jev[i]["context_precision"] for i in ids]),
                  ("검색 AP(정답 문서 기준)", "RAGAS-JEV context_precision_ap", [jev[i]["context_precision_ap"] for i in ids]),
                  ("근거 포함률(05-2)", "RAGAS context_recall", [ragas[i]["ragas"]["context_recall"] for i in ids]),
                  ("근거 포함률(05-2)", "RAGAS-JEV context_recall", [jev[i]["context_recall"] for i in ids])]
    print(f"\n규칙 기준과의 순위 상관 ({len(ids)}문항, 기준 평균: "
          + ", ".join(f"{k} {mean(v):.3f}" for k, v in truth.items()) + ")")
    for basis, label, values in candidates:
        print(f"  {label:<32} 평균 {mean(values):.3f}  ρ={spearman(values, truth[basis]):.3f}  ({basis})")


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        compare(sys.argv[2])
