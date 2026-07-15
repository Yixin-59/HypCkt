# HypCkt

Official implementation of **HypCkt: A Hyperbolic Variational Framework for
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

Download Ckt-Bench-101 from the
[Open Circuit Benchmark repository](https://github.com/zehao-dong/CktGNN)
and place the data at:

```text
OCB/CktBench101/ckt_bench_101.pkl
```

The path can be changed through `data.root_dir` in `configs/hypckt.yaml`.

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

## Acknowledgements

The circuit representation and Euclidean autoregressive decoder are adapted
from [CktGNN](https://github.com/zehao-dong/CktGNN). Hyperbolic operations use
[geoopt](https://github.com/geoopt/geoopt).
