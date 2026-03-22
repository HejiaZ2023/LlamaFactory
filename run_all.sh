#!/usr/bin/env bash

set -e  # stop if any run fails

for yaml in examples/train_full/llm4cov_distill_*.yaml; do
    echo "========================================"
    echo "Running training with config: $yaml"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=4,5,6,7 \
        llamafactory-cli train "$yaml"
done
