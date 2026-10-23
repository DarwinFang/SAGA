# SAGA

Code for **Semantic Alignment for Unsupervised Text Answer Aggregation in LLM Crowdsourcing**.

SAGA estimates query-dependent source alignments from pairwise answer agreement,
computes semantic-center weights, and selects an existing response.

## Setup

```bash
python -m pip install -e '.[test,plots]'
python -m pytest
```

## Experiments

Run from the repository root with the evaluation bundles under `data/`:

```bash
saga-experiments main
saga-experiments ablations --workers 4
saga-experiments source-selection
saga-experiments reversal
saga-experiments cases
saga-experiments source-composition
saga-experiments supplementary
saga-experiments figures
```

| Command | Output |
| --- | --- |
| `main` | Individual sources, SMS, RASA, BAU, AQER, and SAGA |
| `ablations` | SAGA-HA, SAGA-NC, SAGA w/o Query, and 20 SAGA-SQ permutations |
| `source-selection` | Source selection counts and category-wise EM/count Pearson correlations |
| `reversal` | Refitted diagnostic batches of 42 Mintaka and 26 PopQA instances |
| `cases` | Four illustrative inputs, candidates, weights, and selected sources |
| `source-composition` | All four three-source combinations on each dataset |
| `supplementary` | HRRASA, Smoothie, USC, and UAS comparisons |
| `figures` | Weight violins, category heatmaps, and query-region projections |

`--datasets`, `--data`, `--output`, and `--config` select the datasets and paths.
Cached fits are reused when the inputs, configuration, and implementation match.
References enter evaluation and diagnostic-batch construction, not model fitting
or answer selection.

## Configuration

| Dataset | Instances | Metric |
| --- | ---: | --- |
| Mintaka | 480, 60 per category | EM |
| PopQA | 480, 30 per relation | EM |
| CrowdWSA | 450 | Mean sentence GLEU |
| GSM8K | 500 | Final-answer accuracy |

The four sources are Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct,
deepseek-llm-7b-chat, and Granite-3.3-8B-Instruct.
Frozen E5-large-v2 encodes queries and responses with `query: ` prefixes,
attention-mask mean pooling, a 512-token encoding limit, and L2 normalization.
For GSM8K, response embeddings encode the generated solutions, including
reasoning. Evaluation uses the separately extracted final numerical answers.

Generation uses temperature zero and seed 42. Output limits are 64 tokens for
Mintaka and PopQA, 96 for CrowdWSA, and 512 for GSM8K. A GSM8K response that
reaches the length limit without an explicit final answer is generated once
more with a 1,024-token limit. The last boxed answer takes precedence over an
explicit final-answer line, followed by the last output line.

Three input-level validation folds select SAGA regularization by held-out
pairwise agreement loss. The query transform and predictor are then refitted
on the full unlabeled batch. AQER uses uniform initialization, an RMSE stopping
tolerance of `1e-6`, and a maximum of 10,000 iterations.

The reversal analysis uses four query regions and equal numbers of opposing
source advantages, with at least 12 instances per direction. The largest
eligible batch is selected. All four candidate sources remain available.
The category heatmaps divide one selection count equally among sources with
exactly identical selected-answer text, then correlate these counts with source
EM within each category.

Smoothie uses Local estimation from one nearest other query. Residual estimates
are floored at `1e-6` before inverse normalization. USC uses Qwen as the selector
and averages the scores from four cyclic candidate orders. Invalid IDs score
zero. UAS uses the author's main-entry voting rule and cross-model mean
length-normalized log likelihood. UAS and USC require additional LLM inference.

## Data and inference

Each `data/<dataset>/` directory contains:

- `inputs.npz`: IDs, source names, query/answer embeddings, response texts,
  availability masks, source indices, and fitting-group IDs.
- `instances.jsonl`: `id`, `query`, task `prompt`, and category `label`.
- `references.jsonl` and `scores.npz`: references and aligned candidate scores.
- `candidates.jsonl`: response records and generation metadata.
- `final_answers.npz` for GSM8K: `ids`, `sources`, and extracted `texts`.
- `llm_scores.npz` for supplementary comparisons: `ids`, `sources`,
  `log_likelihood` with shape `(instances, scoring models, candidates)`, and
  `usc_choices` with shape `(instances, four cyclic orders)`.

Install `.[embedding]` for encoding and UAS probability scoring, and vLLM for
response generation and USC. Select a GPU with `CUDA_VISIBLE_DEVICES`.

```bash
python tools/rescore.py
python tools/encode.py --dataset GSM8K --output outputs/GSM8K_inputs.npz
python tools/generate.py --dataset GSM8K --source 0 --output outputs/qwen_gsm8k.jsonl
python tools/llm_inference.py uas --dataset GSM8K --model-index 0
python tools/llm_inference.py usc --dataset GSM8K
python tools/llm_inference.py pack --dataset GSM8K
```

Run UAS scoring for source indices 0 through 3 before `pack`. Use `--model-path`
to select a local checkpoint. The supplied task prompts and candidate texts are
used directly. Configuration is in `configs/paper.json`, and attribution is in
[THIRD_PARTY.md](THIRD_PARTY.md).
