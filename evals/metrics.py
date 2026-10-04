"""Reference-based metrics from Module 1: Exact Match, token F1, BLEU and ROUGE.

Each function follows the formula in its original paper. The tests in
tests/test_metrics.py check them against hand calculations and against
sacrebleu and rouge-score.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


# --- Exact Match and token F1 (Rajpurkar et al., 2016) ---------------------


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> int:
    """Return 1 if the normalised prediction equals any normalised reference, else 0."""
    if not references:
        raise ValueError("at least one reference is required")
    pred = normalize_answer(prediction)
    return max(int(pred == normalize_answer(ref)) for ref in references)


def _f1_single(prediction: str, reference: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    ref_toks = normalize_answer(reference).split()
    if not pred_toks or not ref_toks:
        # Both empty counts as a match; one empty is a miss.
        return float(pred_toks == ref_toks)
    common = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall = common / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, references: list[str]) -> float:
    """SQuAD-style token F1, taking the best score over all references."""
    if not references:
        raise ValueError("at least one reference is required")
    return max(_f1_single(prediction, ref) for ref in references)


# --- BLEU (Papineni et al., 2002) -------------------------------------------


def ngrams(tokens: list[str], n: int) -> Counter:
    """Count the n-grams in a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(predictions: list[str], references: list[str], max_n: int = 4) -> dict:
    """Corpus-level BLEU with one reference per prediction and whitespace tokens.

    Returns the score on a 0-100 scale (like sacrebleu), the n-gram
    precisions (also 0-100), the brevity penalty and both corpus lengths.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    matches, totals = [0] * max_n, [0] * max_n
    pred_len = ref_len = 0
    for pred, ref in zip(predictions, references):
        p_toks, r_toks = pred.split(), ref.split()
        pred_len += len(p_toks)
        ref_len += len(r_toks)
        for n in range(1, max_n + 1):
            # The & of two Counters keeps the smaller count: this is clipping.
            matches[n - 1] += sum((ngrams(p_toks, n) & ngrams(r_toks, n)).values())
            totals[n - 1] += max(len(p_toks) - n + 1, 0)

    precisions = [m / t if t else 0.0 for m, t in zip(matches, totals)]
    result = {
        "precisions": [100 * p for p in precisions],
        "pred_len": pred_len,
        "ref_len": ref_len,
    }
    if pred_len == 0 or min(precisions) == 0:
        return {**result, "bleu": 0.0, "bp": None}

    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    return {**result, "bleu": 100 * bp * geo_mean, "bp": bp}


# --- ROUGE (Lin, 2004) ------------------------------------------------------


def rouge_tokenize(text: str) -> list[str]:
    """Tokenise like Google's rouge-score: lowercase, non-alphanumerics become spaces."""
    return _NON_ALNUM.sub(" ", text.lower()).split()


def _prf(overlap: int, pred_total: int, ref_total: int) -> dict:
    p = overlap / pred_total if pred_total else 0.0
    r = overlap / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "fmeasure": f}


def rouge_n(prediction: str, reference: str, n: int) -> dict:
    """ROUGE-N precision, recall and F-measure."""
    p_ng = ngrams(rouge_tokenize(prediction), n)
    r_ng = ngrams(rouge_tokenize(reference), n)
    return _prf(sum((p_ng & r_ng).values()), sum(p_ng.values()), sum(r_ng.values()))


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence, by dynamic programming.

    Uses two rows instead of a full table, so memory grows with len(b) only.
    """
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> dict:
    """ROUGE-L precision, recall and F-measure based on the LCS."""
    p, r = rouge_tokenize(prediction), rouge_tokenize(reference)
    return _prf(lcs_length(p, r), len(p), len(r))
