"""Evaluate HypCkt reconstruction, validity, and novelty."""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hypckt import HypCkt
from scripts.train import load_config, load_ocb_data, set_seed
from utils import (
    is_same_DAG,
    is_valid_Circuit,
    is_valid_DAG,
    load_module_state,
    ratio_same_DAG,
)


@torch.no_grad()
def evaluate_reconstruction(
    model,
    test_graphs,
    batch_size=128,
    latent_samples=10,
    decode_trials=10,
):
    model.eval()
    reconstruction_loss = 0.0
    exact_matches = 0
    total_decodes = 0
    graph_batch = []

    for index, graph in enumerate(tqdm(test_graphs, desc="reconstruction")):
        graph_batch.append(graph)
        if len(graph_batch) < batch_size and index != len(test_graphs) - 1:
            continue

        collated = model._collate_fn(graph_batch)
        mean, log_variance = model.encode(collated)
        _, reconstruction, _, _, _, _ = model.loss(
            mean, log_variance, collated
        )
        reconstruction_loss += float(reconstruction.item())

        for _ in range(latent_samples):
            latent = model.reparameterize(mean, log_variance)
            for _ in range(decode_trials):
                decoded = model.decode(latent)
                exact_matches += sum(
                    is_same_DAG(target, prediction)
                    for target, prediction in zip(graph_batch, decoded)
                )
                total_decodes += len(decoded)
        graph_batch = []

    return {
        "reconstruction_loss": reconstruction_loss / len(test_graphs),
        "reconstruction_accuracy": exact_matches / total_decodes,
    }


@torch.no_grad()
def evaluate_generation(model, train_graphs, n_samples=1000, batch_size=128):
    model.eval()
    valid_dags = 0
    valid_circuits = 0
    generated_valid_dags = []
    generated = 0

    progress = tqdm(total=n_samples, desc="generation")
    while generated < n_samples:
        current_batch = min(batch_size, n_samples - generated)
        graph_batch = model.generate_sample(current_batch)
        for graph in graph_batch:
            if is_valid_DAG(graph, subgraph_level=True):
                valid_dags += 1
                generated_valid_dags.append(graph)
            if is_valid_Circuit(graph, subgraph_level=True):
                valid_circuits += 1
        generated += current_batch
        progress.update(current_batch)
    progress.close()

    novelty = (
        1.0 - ratio_same_DAG(train_graphs, generated_valid_dags)
        if generated_valid_dags
        else 0.0
    )
    return {
        "valid_dag_rate": valid_dags / n_samples,
        "valid_circuit_rate": valid_circuits / n_samples,
        "novelty": novelty,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hypckt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-samples", type=int, default=10)
    parser.add_argument("--decode-trials", type=int, default=10)
    parser.add_argument("--generation-samples", type=int, default=1000)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["training"].get("seed", 1)))
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device or default_device)

    train_graphs, test_graphs = load_ocb_data(config)
    model = HypCkt.from_config(config).to(device)
    load_module_state(model, args.checkpoint, device)

    results = evaluate_reconstruction(
        model,
        test_graphs,
        batch_size=args.batch_size,
        latent_samples=args.latent_samples,
        decode_trials=args.decode_trials,
    )
    results.update(
        evaluate_generation(
            model,
            train_graphs,
            n_samples=args.generation_samples,
            batch_size=args.batch_size,
        )
    )

    for name, value in results.items():
        print(f"{name}: {value:.6f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
