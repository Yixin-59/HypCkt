# HypCkt

**HypCkt: A Hyperbolic Variational Framework for
Analog Circuit Topology Generation**.

HypCkt models subgraph-level circuit DAGs in the Poincare ball. The encoder
performs topology-aware asynchronous message passing, the variational latent
distribution is a wrapped normal, and the autoregressive decoder remains
Euclidean.

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yaml
conda activate hypckt
```

Alternatively, install the Python dependencies directly:

```bash
pip install -r requirements.txt
```

## Dataset

Download OCB datasets from the
[Open Circuit Benchmark repository](https://github.com/zehao-dong/CktGNN)

## Training

```bash
python scripts/train.py --config configs/hypckt.yaml
```

To resume from a saved epoch:

```bash
python scripts/train.py --config configs/hypckt.yaml --resume-epoch 100
```

## Evaluation

```bash
python scripts/eval.py \
  --config configs/hypckt.yaml \
  --checkpoint results/hypckt/model_epoch_300.pt
```

The evaluation reports reconstruction accuracy, valid-DAG rate, valid-circuit
rate, and novelty.
