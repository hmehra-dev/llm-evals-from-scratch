# LLM Evals from Scratch

Notes and code from my attempt to understand LLM evaluation properly, starting from the basic metrics and working up to LLM judges, RAG and agents.

Each module implements the metrics in plain Python first, then checks the results against the standard library for that metric. I also include examples where the metric gives a misleading score, because knowing when a number is wrong matters as much as knowing how to compute it.

A comparison I keep coming back to: evaluating an LLM is a lot like grading a student. You check answers against a key, ask whether the grade can be trusted, sometimes need a teacher to judge open-ended answers, and eventually watch the student do real work. The modules follow roughly that order.

## Contents

| # | Module | Topics | Status |
|---|---|---|---|
| 1 | [Answer-key grading](./01-answer-key-grading) | Exact Match, token F1, BLEU, ROUGE | Done |
| 2 | Can I trust the score? | Confidence intervals, bootstrap, significance testing | In progress |
| 3 | LLM-as-a-judge | Rubric and pairwise judges, judge bias, agreement with human labels | Planned |
| 4 | Evaluating RAG | recall@k, MRR, faithfulness, answer relevance | Planned |
| 5 | Evaluating agents | Task success, tool-call accuracy, step-level evaluation | Planned |
| 6 | Case study | Applying modules 1 to 5 to a research-paper QA system I built | Planned |

Each module folder contains a `README.md` with the theory and a `notebook.ipynb` with the implementation. Everything runs on CPU with small hand-made datasets.

## Setup

```bash
git clone https://github.com/hmehra-dev/llm-evals-from-scratch.git
cd llm-evals-from-scratch
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Tested with Python 3.13.

## Approach

- The implementations are written from the formulas in the original papers. Libraries such as `sacrebleu` and `rouge-score` are only used to check that the results match.
- Each module links to the papers and documentation it is based on.
- LLM-as-a-judge and agent evaluation are changing quickly, so those modules note when the material was last reviewed.

## References

- Clémentine Fourrier et al., [The LLM Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook), Hugging Face
- Chip Huyen, *AI Engineering* (O'Reilly, 2025), chapters 3 and 4; companion repo [chiphuyen/aie-book](https://github.com/chiphuyen/aie-book)
- Shankar et al., [Who Validates the Validators?](https://arxiv.org/abs/2404.12272) (2024)
- Evaluation frameworks: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai), [ragas](https://github.com/vibrantlabsai/ragas), [deepeval](https://github.com/confident-ai/deepeval)

## Feedback

If you find a mistake or have a suggestion, please open an issue.

## License

[MIT](./LICENSE)
