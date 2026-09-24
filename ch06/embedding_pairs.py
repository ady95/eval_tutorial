"""06-6 실습: 임베딩 모델 세 개로 02-2의 문장 쌍 유사도를 비교합니다.

실행: python -m ch06.embedding_pairs --base-url http://localhost:11434/v1
"""
import argparse

import numpy as np
from openai import OpenAI

MODELS = ["bge-m3", "qwen3-embedding:0.6b", "nomic-embed-text"]
PAIRS = [
    ("같은 뜻", "대한민국의 수도는 서울입니다.", "한국의 수도는 서울특별시입니다."),
    ("같은 뜻", "환불은 구매 후 7일 이내에 가능합니다.", "물건을 산 날부터 일주일 안에는 돈을 돌려받을 수 있어요."),
    ("부정문", "환불은 구매 후 7일 이내에 가능합니다.", "환불은 구매 후 7일 이내에 불가능합니다."),
    ("숫자", "환불은 구매 후 7일 이내에 가능합니다.", "환불은 구매 후 30일 이내에 가능합니다."),
    ("무관", "환불은 구매 후 7일 이내에 가능합니다.", "오늘은 날씨가 맑습니다."),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="ollama")

    print("구분     " + "".join(f"{m:>22}" for m in MODELS))
    table = {}
    for model in MODELS:
        texts = [t for _, a, b in PAIRS for t in (a, b)]
        v = np.array([d.embedding for d in client.embeddings.create(model=model, input=texts).data])
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        table[model] = [float(v[2 * i] @ v[2 * i + 1]) for i in range(len(PAIRS))]
    for i, (kind, a, b) in enumerate(PAIRS):
        print(f"{kind:<6} " + "".join(f"{table[m][i]:22.3f}" for m in MODELS))
    gap = {m: min(table[m][0], table[m][1]) - table[m][4] for m in MODELS}
    print("\n'같은 뜻' 중 낮은 쪽 - '무관' (클수록 의미를 잘 구별): " + ", ".join(f"{m} {g:.3f}" for m, g in gap.items()))


if __name__ == "__main__":
    main()
