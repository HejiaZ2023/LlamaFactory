# Beta-Weighted DPO Handoff

This note describes the beta-weighted DPO change set added to this repo, how to prepare data for it, and what to validate on the target GPU server.

## What Changed

The standard DPO path was extended so pairwise ranking samples may optionally include:

- `score_chosen`
- `score_rejected`

For each sample, the training pipeline computes:

```text
score_diff = max(score_chosen - score_rejected, 0)
beta_star = beta * score_diff
loss = -log sigmoid(beta_star * logratio_diff)
```

where `logratio_diff` is the usual DPO term:

```text
logratio_diff =
  (policy_chosen_logp - reference_chosen_logp)
  - (policy_rejected_logp - reference_rejected_logp)
```

If the score fields are absent, the implementation falls back to standard sigmoid DPO with constant `beta`.

## Files Changed

Core code:

- [`src/llamafactory/data/parser.py`](/home/dbmw/sandbox/LlamaFactory/src/llamafactory/data/parser.py)
- [`src/llamafactory/data/converter.py`](/home/dbmw/sandbox/LlamaFactory/src/llamafactory/data/converter.py)
- [`src/llamafactory/data/processor/pairwise.py`](/home/dbmw/sandbox/LlamaFactory/src/llamafactory/data/processor/pairwise.py)
- [`src/llamafactory/data/collator.py`](/home/dbmw/sandbox/LlamaFactory/src/llamafactory/data/collator.py)
- [`src/llamafactory/train/dpo/trainer.py`](/home/dbmw/sandbox/LlamaFactory/src/llamafactory/train/dpo/trainer.py)
- [`src/llamafactory/train/rm/trainer.py`](/home/dbmw/sandbox/LlamaFactory/src/llamafactory/train/rm/trainer.py)

Tests:

- [`tests/data/test_converter.py`](/home/dbmw/sandbox/LlamaFactory/tests/data/test_converter.py)
- [`tests/data/test_collator.py`](/home/dbmw/sandbox/LlamaFactory/tests/data/test_collator.py)
- [`tests/train/test_dpo_trainer.py`](/home/dbmw/sandbox/LlamaFactory/tests/train/test_dpo_trainer.py)

Example config:

- [`examples/train_full/llm4cov_distill_dpo.yaml`](/home/dbmw/sandbox/LlamaFactory/examples/train_full/llm4cov_distill_dpo.yaml)

## Dataset Format

The DPO dataset should still be a standard ranking dataset with prompt plus chosen/rejected responses. The only addition is the optional numeric score pair.

Example ShareGPT-style sample:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Question text"
    }
  ],
  "chosen": {
    "from": "gpt",
    "value": "Preferred answer"
  },
  "rejected": {
    "from": "gpt",
    "value": "Less preferred answer"
  },
  "score_chosen": 4.0,
  "score_rejected": 1.5
}
```

## dataset_info.json Entry

Add a dataset entry in [`data/dataset_info.json`](/home/dbmw/sandbox/LlamaFactory/data/dataset_info.json) like:

```json
{
  "llm4cov_distill_dpo": {
    "hf_hub_url": "your-org/your-dpo-dataset",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected",
      "score_chosen": "score_chosen",
      "score_rejected": "score_rejected"
    }
  }
}
```

If your data is local instead of on Hugging Face, use `file_name` instead of `hf_hub_url`.

## Training Config

A full-parameter DPO example was added at:

- [`examples/train_full/llm4cov_distill_dpo.yaml`](/home/dbmw/sandbox/LlamaFactory/examples/train_full/llm4cov_distill_dpo.yaml)

Expected launch pattern:

```bash
llamafactory-cli train examples/train_full/llm4cov_distill_dpo.yaml --output_dir xxx
```

Notes:

- `stage` must be `dpo`
- `pref_loss` should be `sigmoid`
- `dataset` should point to the DPO dataset entry
- `model_name_or_path` can be the base model or your SFT checkpoint, depending on your recipe

## Important Implementation Detail

The current interpretation of the user request is:

```text
f(score_diff) = score_diff
```

So:

```text
beta_star = beta * score_diff
```

This means:

- if `score_diff > 1`, the effective beta is larger than the base beta
- if `0 < score_diff < 1`, the effective beta is smaller
- if `score_diff = 0`, the pair contributes `-log(sigmoid(0))`, which is a constant `log(2)`

If you want a different scaling function such as `log1p(score_diff)`, clipping, normalization, or `1 + score_diff`, that should be changed in [`trainer.py`](/home/dbmw/sandbox/LlamaFactory/src/llamafactory/train/dpo/trainer.py).

## Fallback Behavior

If `score_chosen` and `score_rejected` are missing:

- the converter emits `NaN`
- preprocessing keeps `score_diff` as `NaN`
- the DPO trainer falls back to constant `beta`

This keeps old pairwise datasets and old tokenized datasets usable.

## Reward Model Compatibility

The same pairwise collator is used by reward-model training. Because of that, RM training would have received the extra score tensors unless explicitly filtered. That filtering was added in:

- [`src/llamafactory/train/rm/trainer.py`](/home/dbmw/sandbox/LlamaFactory/src/llamafactory/train/rm/trainer.py)

## What To Verify On The GPU Server

1. Environment imports

- `trl` is installed
- `pytest` is installed if you want to run tests
- `PYTHONPATH=src` works if you run repo-local tests directly

2. Dataset parsing

- one batch contains `score_chosen`, `score_rejected`, and `score_diff`
- values are numeric and not accidentally strings or nulls

3. Training behavior

- training starts without `unexpected keyword argument` errors
- metrics include `scores/chosen`, `scores/rejected`, and `scores/diff` during DPO
- loss is finite

4. Sanity check the score scale

- inspect a few examples and compute rough `score_diff` statistics
- verify the resulting `beta_star` scale is reasonable for your score range

## Recommended First Debug Commands

Basic syntax/import smoke test:

```bash
python -m py_compile \
  src/llamafactory/data/parser.py \
  src/llamafactory/data/converter.py \
  src/llamafactory/data/processor/pairwise.py \
  src/llamafactory/data/collator.py \
  src/llamafactory/train/dpo/trainer.py \
  src/llamafactory/train/rm/trainer.py
```

If the target server has `pytest` and `trl`:

```bash
PYTHONPATH=src python -m pytest tests/data/test_converter.py -q
PYTHONPATH=src python -m pytest tests/data/test_collator.py -q
PYTHONPATH=src python -m pytest tests/train/test_dpo_trainer.py -q
```

Single-run training debug:

```bash
llamafactory-cli train examples/train_full/llm4cov_distill_dpo.yaml \
  --max_samples 8 \
  --overwrite_cache true \
  --output_dir /tmp/llm4cov_dpo_debug
```

## Known Open Question

The requested formula specified `beta * f(score_diff)` but did not define `f`. This implementation uses the simplest choice:

```text
f(x) = x
```

If results are unstable or too weak, the first knob to revisit is the score transform.
