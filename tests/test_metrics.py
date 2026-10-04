"""Tests for evals.metrics: hand-calculated cases plus agreement with the standard libraries."""

import math

import pytest

from evals.metrics import (
    corpus_bleu,
    exact_match,
    lcs_length,
    normalize_answer,
    rouge_l,
    rouge_n,
    token_f1,
)

GEN_DATA = [
    ("the cat sat on the mat", "the cat is sitting on the mat"),
    (
        "retrieval augmented generation combines a retriever with a language model",
        "retrieval augmented generation combines a document retriever with a generative language model",
    ),
    (
        "the model was evaluated on three public benchmarks and achieved strong results",
        "the model achieved strong results on three public benchmarks",
    ),
]


# --- Exact Match and token F1 ------------------------------------------------


def test_normalize_answer_strips_case_punctuation_and_articles():
    assert normalize_answer("The Eiffel Tower in Paris.") == "eiffel tower in paris"


@pytest.mark.parametrize(
    "prediction, references, expected",
    [
        ("Paris", ["Paris"], 1),
        ("paris.", ["Paris"], 1),
        ("William Shakespeare", ["Shakespeare", "William Shakespeare"], 1),
        ("The Eiffel Tower in Paris.", ["Eiffel Tower"], 0),
        ("Saturn", ["Jupiter"], 0),
    ],
)
def test_exact_match(prediction, references, expected):
    assert exact_match(prediction, references) == expected


def test_token_f1_hand_calculation():
    # common = 2, precision = 2/4, recall = 2/2, F1 = 2/3
    assert token_f1("The Eiffel Tower in Paris.", ["Eiffel Tower"]) == pytest.approx(2 / 3)


def test_token_f1_missing_word():
    # precision = 1, recall = 2/3, F1 = 0.8 (Module 1, exercise 1)
    assert token_f1("barack obama", ["president barack obama"]) == pytest.approx(0.8)


def test_token_f1_takes_best_reference():
    assert token_f1("Shakespeare", ["William Shakespeare", "Shakespeare"]) == 1.0


def test_token_f1_no_overlap_is_zero():
    assert token_f1("Saturn", ["Jupiter"]) == 0.0


def test_metrics_require_a_reference():
    with pytest.raises(ValueError):
        exact_match("Paris", [])
    with pytest.raises(ValueError):
        token_f1("Paris", [])


# --- BLEU ---------------------------------------------------------------------


def test_bleu_matches_hand_calculation():
    res = corpus_bleu(*zip(*GEN_DATA))
    # Matches / totals per n-gram size: 24/28, 16/25, 8/22, 3/19
    expected_precisions = [24 / 28, 16 / 25, 8 / 22, 3 / 19]
    assert res["precisions"] == pytest.approx([100 * p for p in expected_precisions])
    assert res["bp"] == 1.0  # c = r = 28, so exp(1 - 28/28) = 1
    geo = math.exp(sum(math.log(p) for p in expected_precisions) / 4)
    assert res["bleu"] == pytest.approx(100 * geo)


def test_bleu_brevity_penalty():
    res = corpus_bleu(["the cat"], ["the cat sat on the mat"], max_n=2)
    assert res["bp"] == pytest.approx(math.exp(-2))
    assert res["bleu"] == pytest.approx(100 * math.exp(-2))


def test_bleu_clipping():
    res = corpus_bleu(["the the the the"], ["the cat"], max_n=1)
    assert res["precisions"][0] == pytest.approx(25.0)


def test_bleu_zero_when_an_ngram_order_has_no_match():
    assert corpus_bleu(["a b c d"], ["w x y z"])["bleu"] == 0.0


def test_bleu_length_mismatch_raises():
    with pytest.raises(ValueError):
        corpus_bleu(["a"], ["a", "b"])


def test_bleu_agrees_with_sacrebleu():
    sacrebleu = pytest.importorskip("sacrebleu")
    preds, refs = map(list, zip(*GEN_DATA))
    lib = sacrebleu.corpus_bleu(preds, [refs], tokenize="none")
    assert corpus_bleu(preds, refs)["bleu"] == pytest.approx(lib.score, abs=1e-4)


# --- ROUGE --------------------------------------------------------------------


def test_lcs_hand_calculation():
    # "the cat on the mat" is the longest common subsequence
    assert lcs_length("the cat sat on the mat".split(), "the cat is sitting on the mat".split()) == 5


def test_lcs_edge_cases():
    assert lcs_length([], ["a"]) == 0
    assert lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3
    assert lcs_length(["a", "b", "c"], ["c", "b", "a"]) == 1


def test_rouge_l_hand_calculation():
    res = rouge_l("the cat sat on the mat", "the cat is sitting on the mat")
    assert res["precision"] == pytest.approx(5 / 6)
    assert res["recall"] == pytest.approx(5 / 7)


@pytest.mark.parametrize("pred, ref", GEN_DATA)
def test_rouge_agrees_with_rouge_score(pred, ref):
    rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    lib = scorer.score(ref, pred)  # rouge-score takes (target, prediction)
    ours = {"rouge1": rouge_n(pred, ref, 1), "rouge2": rouge_n(pred, ref, 2), "rougeL": rouge_l(pred, ref)}
    for key in ("rouge1", "rouge2", "rougeL"):
        for field in ("precision", "recall", "fmeasure"):
            assert ours[key][field] == pytest.approx(getattr(lib[key], field), abs=1e-4)
