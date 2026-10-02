# Module 1: Answer-Key Grading

How do you score a model's answer when you already know the correct one?

The simplest family of evaluation metrics compares the model's output with a reference answer and measures how much they overlap. This module covers the four most common ones: Exact Match, token F1, BLEU and ROUGE.

Think of grading a test with an answer key:

- A multiple-choice question is either right or wrong. That is Exact Match.
- For a short written answer, you might give partial credit for the words that match the key. That is F1.
- For a translation, many wordings are acceptable, so you check how many phrases the answer shares with a reference translation. That is BLEU.
- For a summary, you check how much of the reference summary's content was covered. That is ROUGE.

All four rest on the same assumption: matching words means a good answer. The notebook ends with examples where that assumption fails.

## Exact Match

After light normalisation, is the prediction identical to the reference? The normalisation follows the official SQuAD evaluation script: lowercase, remove punctuation, remove the articles *a*, *an* and *the*, and collapse whitespace.

```
EM = 1 if normalize(prediction) == normalize(reference) else 0
```

When several references are acceptable, the best score is used.

## Token F1

Treat both answers as bags of words and measure the overlap.

```
common    = number of shared tokens (counting repeats)
precision = common / tokens in prediction
recall    = common / tokens in reference
F1        = 2 * precision * recall / (precision + recall)
```

Precision penalises extra words that are not in the reference. Recall penalises leaving out words that are. F1 is their harmonic mean, so it is only high when both are.

## BLEU

BLEU (Papineni et al., 2002) was designed for machine translation. It counts matching n-grams of length 1 to 4 and penalises outputs that are too short.

```
p_n  = clipped n-gram matches / n-grams in prediction,  n = 1..4
BP   = 1                      if c > r
       exp(1 - r / c)         otherwise
BLEU = BP * exp( (1/4) * sum(log p_n) )
```

Here `c` is the total prediction length and `r` the total reference length.

- **Clipping:** each n-gram is credited at most as many times as it appears in the reference. Without it, "the the the the" would get perfect unigram precision against "the cat".
- **Brevity penalty:** precision alone would reward very short outputs. The brevity penalty corrects for that.
- **Corpus level:** BLEU is defined at corpus level, meaning counts are summed over all sentences before the score is computed. Sentence-level BLEU is noisy and needs smoothing.

## ROUGE

ROUGE (Lin, 2004) was designed for summarisation and is built around recall: how much of the reference does the output cover?

- **ROUGE-1 and ROUGE-2** measure unigram and bigram overlap.
- **ROUGE-L** uses the longest common subsequence (LCS): the longest sequence of words that appears in both texts in the same order, not necessarily next to each other.

```
P = LCS / tokens in prediction
R = LCS / tokens in reference
F = 2PR / (P + R)
```

## The notebook

1. Implements all four metrics using only `collections`, `math` and `re`.
2. Checks BLEU against [`sacrebleu`](https://github.com/mjpost/sacrebleu) and ROUGE against Google's [`rouge-score`](https://pypi.org/project/rouge-score/). Both agree to four decimal places.
3. Shows three cases where overlap metrics give the wrong verdict: a correct paraphrase, a negated sentence, and a wrong fact in otherwise identical wording.
4. Ends with a few exercises.

## Takeaway

These metrics measure similarity of wording, not correctness. They are cheap, deterministic and useful for short factual answers or as regression checks. For open-ended outputs they need to be combined with human or model-based judgement, which is covered in Module 3.

## References

- Rajpurkar et al., 2016. [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250). Source of the Exact Match and F1 definitions.
- Papineni et al., 2002. [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- Post, 2018. [A Call for Clarity in Reporting BLEU Scores](https://arxiv.org/abs/1804.08771). The paper behind `sacrebleu`.
- Lin, 2004. [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
