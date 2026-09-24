"""05장 평가용 RAG: 문서 청크 나누기 → 임베딩 검색 → 답변 생성.

05~10장에서 공통으로 씁니다. 설정(청크 크기, top_k)을 바꿔 가며 평가하는 것이 목적이라
구조를 최대한 단순하게 두었습니다.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.llm import MODEL, client, embed

DOCS_DIR = Path("data/docs")


@dataclass
class Chunk:
    doc_id: str      # 파일 이름 (예: 06-domestic-travel)
    title: str       # 문서 제목 (예: 국내 출장비 규정)
    index: int       # 문서 안에서 몇 번째 청크인지
    text: str        # 문서 제목을 앞에 붙인 청크 본문


def load_docs() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(DOCS_DIR.glob("*.md"))}


def split_doc(doc_id: str, text: str, chunk_size: int) -> list[Chunk]:
    """문단을 모아 chunk_size 글자를 넘지 않게 묶습니다. 문단 하나가 더 길면 그대로 한 청크로 둡니다."""
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip()
    # 모든 문서에 똑같이 들어 있는 안내문(> 로 시작하는 인용 블록)은 검색을 방해하므로 뺍니다
    paragraphs = [p.strip() for p in "\n".join(lines[1:]).split("\n\n") if p.strip() and not p.startswith(">")]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > chunk_size:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return [Chunk(doc_id, title, i, f"[{title}]\n{c}") for i, c in enumerate(chunks)]


class Retriever:
    def __init__(self, chunk_size: int = 400):
        self.chunks = [c for doc_id, text in load_docs().items() for c in split_doc(doc_id, text, chunk_size)]
        vectors = []
        for i in range(0, len(self.chunks), 32):  # 32개씩 나눠 임베딩합니다
            vectors += embed([c.text for c in self.chunks[i:i + 32]])
        m = np.array(vectors)
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        q = np.array(embed([query])[0])
        scores = self.matrix @ (q / np.linalg.norm(q))
        best = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in best]


SYSTEM = """당신은 누리솔 사내 규정 안내 도우미입니다.
아래 [참고 문서]에 있는 내용만 근거로 질문에 답하세요.
참고 문서에서 답을 찾을 수 없으면 추측하지 말고 "규정에서 찾을 수 없습니다"라고 답하세요.
답변은 두세 문장으로 간결하게 쓰세요."""


def generate(question: str, contexts: list[str]) -> str:
    docs = "\n\n---\n\n".join(contexts)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"[참고 문서]\n{docs}\n\n[질문]\n{question}"}],
    )
    return response.choices[0].message.content.strip()


def run_rag(questions: list[dict], chunk_size: int = 400, top_k: int = 3) -> list[dict]:
    """질문마다 검색과 생성을 실행해 평가에 필요한 정보를 모두 기록합니다."""
    retriever = Retriever(chunk_size)
    results = []
    for q in questions:
        hits = retriever.search(q["question"], top_k)
        contexts = [c.text for c, _ in hits]
        results.append({
            **q,
            "retrieved": [{"doc_id": c.doc_id, "index": c.index, "score": round(s, 4), "text": c.text}
                          for c, s in hits],
            "answer": generate(q["question"], contexts),
        })
    return results


def load_eval_set() -> list[dict]:
    return json.loads(Path("data/eval_set.json").read_text(encoding="utf-8"))
