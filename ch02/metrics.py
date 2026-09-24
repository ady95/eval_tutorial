"""02장 전통적인 지표: Exact Match, F1, BLEU, ROUGE-L, 임베딩 코사인 유사도."""
import re
from collections import Counter

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer

from common.llm import embed


def normalize(text: str) -> str:
    """문장 부호를 지우고 공백을 하나로 합칩니다."""
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """정규화한 두 문자열이 완전히 같으면 1, 아니면 0."""
    return float(normalize(prediction) == normalize(reference))


def tokenize(text: str, unit: str = "word") -> list[str]:
    """word: 띄어쓰기(어절) 단위, char: 공백을 뺀 음절(글자) 단위."""
    text = normalize(text)
    return text.split() if unit == "word" else list(text.replace(" ", ""))


def f1(prediction: str, reference: str, unit: str = "word") -> float:
    """겹치는 토큰으로 Precision과 Recall을 구해 조화평균을 냅니다."""
    pred, ref = tokenize(prediction, unit), tokenize(reference, unit)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def bleu(prediction: str, reference: str) -> float:
    """문장 단위 BLEU (0~1로 환산)."""
    return sacrebleu.sentence_bleu(prediction, [reference]).score / 100


class KoreanTokenizer:
    """rouge-score 기본 토크나이저는 한글을 지워 버리므로 어절 단위로 직접 자릅니다."""

    def tokenize(self, text: str) -> list[str]:
        return tokenize(text, "word")


_rouge = rouge_scorer.RougeScorer(["rougeL"], tokenizer=KoreanTokenizer())


def rouge_l(prediction: str, reference: str) -> float:
    """가장 긴 공통 부분 수열(LCS) 기반 ROUGE-L F1."""
    return _rouge.score(reference, prediction)["rougeL"].fmeasure


def cosine_similarity(a: str, b: str) -> float:
    """두 문장의 임베딩 코사인 유사도."""
    va, vb = (np.array(v) for v in embed([a, b]))
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
