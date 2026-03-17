# ActiveCQ: Active Estimation of Causal Quantities

This repository contains the official implementation of **ActiveCQ**, a framework for actively acquiring data to efficiently estimate causal quantities such as CATE, ATE, ATT, and distributional shift effects.

## Installation

```bash
pip install -r requirements.txt
```

**Note:** When running the code with `warm_size` set to 500, you may encounter a dtype mismatch bug in `gpytorch`. To fix it, add the following lines to `lib/python3.x/site-packages/linear_operator/operators/dense_linear_operator.py` (line 65, in `_matmul`):

```python
self.tensor = self.tensor.to(torch.float64)
rhs = rhs.to(torch.float64)
```

## Data Preparation

Create a `data/` directory in the project root, and prepare the datasets as follows:

| Dataset | Description | Location |
|---------|-------------|----------|
| **Simulation** | Synthetic data (generated automatically) | No download needed |
| **IHDP** | Infant Health and Development Program | `data/ihdp/` |

## Quick Start

Run a simple example with simulation data (no external data needed):

```bash
bash run_example.sh
```

This runs active learning with `random` and `VR_CME_B` acquisition functions on a synthetic CATE task, evaluates with AMSE, and plots convergence curves.

## Usage

The entry point is `src/application/main.py`, which uses chained CLI commands:

### 1. Active Learning

```bash
python src/application/main.py \
    active-learning \
        --job-dir experiments/my_exp \
        --num-trials 5 \
        --acq-size 20 \
        --warm-start-size 50 \
        --max-acquisitions 35 \
        --acquisition-function acqe \
        --adaptive-strategy VR \
        --cde-estimator CME \
        --batch-aware B \
    simulation \
        --num-examples 500 \
        --task_type cate \
        --treatment_type discrete \
    imp \
        --learning-rate 0.05 \
        --gp_epochs 500
```

**Acquisition functions:** `random`, `coresets`, `var_rank`, `var_reduction_rank`, `acqe`

**Adaptive strategies (for `acqe`):** `IG` (Information Gain), `VR` (Variance Reduction)

**CDE estimators (for `acqe`):** `CME` (Conditional Mean Embedding), `MDN` (Mixture Density Network), `LSCDE` (Least-Squares CDE)

**Batch modes:** `B` (batch-aware), `G` (greedy)

**Datasets:** `simulation`, `ihdp`

### 2. Evaluation

```bash
python src/application/main.py \
    evaluate \
        --experiment-dir experiments/my_exp/active_learning/VR_CME_B \
        --output-dir experiments/my_exp/results \
    amse
```

### 3. Plotting

```bash
python src/application/main.py \
    evaluate \
        --experiment-dir experiments/my_exp/results \
    plot-convergence-in-out \
        --prefix my_exp \
        -m random \
        -m VR_CME_B
```

## Project Structure

```
ActiveCQ/
  src/
    application/
      main.py                    # CLI entry point
      workflows/
        active_learning.py       # Active learning loop
        training.py              # Model training
        tuning.py                # Hyperparameter tuning
        evaluation.py            # AMSE evaluation and plotting
        utils.py                 # Shared utilities
    library/
      acquisitions.py            # Acquisition functions (random, coresets, ACQE, etc.)
      plotting.py                # Visualization utilities
      models/
        imp.py                   # IMP models (CATE, ATE, ATT, DS)
        deep_kernel.py           # Deep Kernel GP
        tarnet.py                # TARNet
        neural_network.py        # Neural network model
      modules/
        CME.py                   # Conditional Mean Embedding
        gaussian_process.py      # GP modules
        cdest/                   # Conditional density estimation (MDN, LSCDE, KMN)
      datasets/
        simulation.py            # Synthetic data generation
        ihdp.py                  # Semi-synthetic dataset loader
        active_learning.py       # Active learning dataset wrapper
  requirements.txt
  run_example.sh                 # Quick start example
```

## Reference

If you find this code useful, please cite:

```bibtex
@inproceedings{
gao2026activecq,
title={Active{CQ}: Active Estimation of Causal Quantities},
author={Erdun Gao and Dino Sejdinovic},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=CWpQsAubxy}
}
```
