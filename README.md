# Knowledge Vector of Logical Reasoning in Large Language Models

This repository contains the code framework for the paper
**Knowledge Vector of Logical Reasoning in Large Language Models**.

## Code Layout

```text
kv_code/
  configs/
    config.yaml
  scripts/
    extract_activations.py
    train_naive_vectors.py
    build_sae_subspace.py
    refine_vectors.py
    run_steering.py
  src/kv_reasoning/
    activations.py
    data.py
    eval.py
    models.py
    prompts.py
    sae.py
    steering.py
    utils.py
    vectors.py
  dataset/
  outputs/
  checkpoints/
```

## Data

The paper uses the following datasets.

| Reasoning type | Dataset |
| --- | --- |
| Deductive | JustLogic |
| Inductive | DEER |
| Abductive | ART |

Dataset placeholder files are provided under `dataset/`:

```text
dataset/
  deductive.json
  inductive.json
  abductive.json
```

## Setup

```bash
cd kv_code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export HF_TOKEN=your_huggingface_token
```

Set `model.model_id` in `configs/config.yaml` before running experiments.

For inductive METEOR evaluation, NLTK may require:

```bash
python -m nltk.downloader wordnet omw-1.4
```

## Pipeline

### 1. Extract contrastive activations

Deductive example:

```bash
PYTHONPATH=src python scripts/extract_activations.py \
  --task deductive \
  --data-path dataset/deductive.json \
  --output outputs/deductive_activations.pt
```

Abductive example using Hugging Face ART:

```bash
PYTHONPATH=src python scripts/extract_activations.py \
  --task abductive \
  --dataset-name allenai/art \
  --split validation \
  --output outputs/abductive_activations.pt
```

### 2. Train naive reasoning vectors

```bash
PYTHONPATH=src python scripts/train_naive_vectors.py \
  --activation-file outputs/deductive_activations.pt \
  --output outputs/deductive_naive_vector.pt
```

Run the same command for inductive and abductive activations.

### 3. Optional: build SAE subspaces

If you have saved SAE latent activations and decoder weights, use:

```bash
PYTHONPATH=src python scripts/build_sae_subspace.py \
  --positive-latents outputs/deductive_positive_sae_latents.pt \
  --negative-latents outputs/deductive_negative_sae_latents.pt \
  --decoder-weight checkpoints/sae_decoder.pt \
  --output outputs/deductive_subspace.pt
```

### 4. Refine complementary vectors

```bash
PYTHONPATH=src python scripts/refine_vectors.py \
  --deductive-activations outputs/deductive_activations.pt \
  --inductive-activations outputs/inductive_activations.pt \
  --abductive-activations outputs/abductive_activations.pt \
  --deductive-vector outputs/deductive_naive_vector.pt \
  --inductive-vector outputs/inductive_naive_vector.pt \
  --abductive-vector outputs/abductive_naive_vector.pt \
  --deductive-subspace outputs/deductive_subspace.pt \
  --inductive-subspace outputs/inductive_subspace.pt \
  --abductive-subspace outputs/abductive_subspace.pt \
  --output-dir outputs/refined
```

The `--*-subspace` arguments are optional. Without them, the script applies the probe loss plus complementary loss only.

### 5. Run steering evaluation

```bash
PYTHONPATH=src python scripts/run_steering.py \
  --task deductive \
  --data-path dataset/deductive.json \
  --vector outputs/refined/deductive_refined_vector.pt \
  --coeff 1.0
```

For ART:

```bash
PYTHONPATH=src python scripts/run_steering.py \
  --task abductive \
  --dataset-name allenai/art \
  --split validation \
  --vector outputs/refined/abductive_refined_vector.pt \
  --coeff 1.0
```
