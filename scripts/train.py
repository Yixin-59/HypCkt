"""Train HypCkt on Ckt-Bench-101."""

import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hypckt import HypCkt
from utils import load_module_state


class LinearBetaScheduler:
    def __init__(self, start=0.0, end=1.0, anneal_epochs=20):
        if anneal_epochs <= 0:
            raise ValueError("anneal_epochs must be positive")
        self.start = float(start)
        self.end = float(end)
        self.anneal_epochs = int(anneal_epochs)

    def __call__(self, epoch):
        progress = min(max(epoch, 0) / self.anneal_epochs, 1.0)
        return self.start + progress * (self.end - self.start)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_ocb_data(config, smoke_test=False):
    data_path = Path(config["data"]["root_dir"]) / "ckt_bench_101.pkl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Ckt-Bench-101 was not found at {data_path}. "
            "Update data.root_dir in the configuration file."
        )
    with data_path.open("rb") as handle:
        datasets = pickle.load(handle)

    train_graphs = [item[0] for item in datasets[0]]
    test_graphs = [item[0] for item in datasets[1]]
    if smoke_test:
        train_graphs = train_graphs[:100]
        test_graphs = test_graphs[:20]
    return train_graphs, test_graphs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))


def build_scheduler(optimizer, config):
    scheduler_config = config["training"].get("scheduler", {})
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config.get("factor", 0.1)),
        patience=int(scheduler_config.get("patience", 30)),
    )


def save_checkpoint(model, optimizer, scheduler, epoch, save_dir):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / f"model_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        save_path / f"training_state_epoch_{epoch}.pt",
    )


def load_checkpoint(model, optimizer, scheduler, epoch, save_dir, device):
    save_path = Path(save_dir)
    load_module_state(model, save_path / f"model_epoch_{epoch}.pt", device)
    training_state = torch.load(
        save_path / f"training_state_epoch_{epoch}.pt", map_location=device
    )
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])


def assert_finite(model, loss, context):
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at {context}")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient in {name} at {context}")


def train_one_epoch(model, optimizer, graphs, beta, batch_size, epoch, grad_clip):
    model.train()
    random.shuffle(graphs)
    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "type": 0.0,
        "position": 0.0,
        "feature": 0.0,
    }

    graph_batch = []
    progress = tqdm(graphs, desc=f"epoch {epoch}")
    for index, graph in enumerate(progress):
        graph_batch.append(graph)
        if len(graph_batch) < batch_size and index != len(graphs) - 1:
            continue

        optimizer.zero_grad()
        collated = model._collate_fn(graph_batch)
        mean, log_variance = model.encode(collated)
        loss_values = model.loss(mean, log_variance, collated, beta=beta)
        loss, reconstruction, kl, type_loss, position_loss, feature_loss = loss_values
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        assert_finite(model, loss, f"epoch={epoch}, batch={index // batch_size}")
        optimizer.step()

        totals["loss"] += float(loss.item())
        totals["reconstruction"] += float(reconstruction.item())
        totals["kl"] += float(kl.item())
        totals["type"] += float(type_loss)
        totals["position"] += float(position_loss)
        totals["feature"] += float(feature_loss)
        progress.set_postfix(
            loss=f"{loss.item() / len(graph_batch):.3f}",
            curvature=f"{model.manifold._get_c().item():.3f}",
        )
        graph_batch = []

    return totals


def append_metrics(path, epoch, metrics, n_graphs, beta, curvature, elapsed):
    is_new = not Path(path).exists()
    with open(path, "a", encoding="utf-8") as handle:
        if is_new:
            handle.write(
                "epoch,loss,reconstruction,kl,type,position,feature,beta,curvature,time_s\n"
            )
        values = [
            epoch,
            metrics["loss"] / n_graphs,
            metrics["reconstruction"] / n_graphs,
            metrics["kl"] / n_graphs,
            metrics["type"] / n_graphs,
            metrics["position"] / n_graphs,
            metrics["feature"] / n_graphs,
            beta,
            curvature,
            elapsed,
        ]
        handle.write(",".join(str(value) for value in values) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hypckt.yaml")
    parser.add_argument("--resume-epoch", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    if args.smoke_test:
        training["batch_size"] = 16
        training["epochs"] = 2

    set_seed(int(training.get("seed", 1)))
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device or default_device)
    train_graphs, _ = load_ocb_data(config, smoke_test=args.smoke_test)

    model = HypCkt.from_config(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    save_dir = config.get("save_dir", "results/hypckt")

    start_epoch = 1
    if args.resume_epoch is not None:
        load_checkpoint(
            model, optimizer, scheduler, args.resume_epoch, save_dir, device
        )
        start_epoch = args.resume_epoch + 1

    beta_config = training.get("beta_anneal", {})
    if beta_config.get("enabled", False):
        beta_schedule = LinearBetaScheduler(
            start=beta_config.get("start", 0.0),
            end=beta_config.get("end", 1.0),
            anneal_epochs=beta_config.get("anneal_epochs", 20),
        )
    else:
        constant_beta = float(training.get("beta_default", 0.005))
        beta_schedule = lambda _epoch: constant_beta

    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    grad_clip = float(training.get("grad_clip", 5.0))
    save_interval = int(training.get("save_interval", 50))
    metrics_path = Path(save_dir) / "training_metrics.csv"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        beta = beta_schedule(epoch)
        start_time = time.time()
        metrics = train_one_epoch(
            model,
            optimizer,
            train_graphs,
            beta,
            batch_size,
            epoch,
            grad_clip,
        )
        average_loss = metrics["loss"] / len(train_graphs)
        scheduler.step(average_loss)
        curvature = model.manifold._get_c().item()
        append_metrics(
            metrics_path,
            epoch,
            metrics,
            len(train_graphs),
            beta,
            curvature,
            time.time() - start_time,
        )
        print(
            f"epoch={epoch} loss={average_loss:.4f} "
            f"beta={beta:.4f} curvature={curvature:.4f}"
        )
        if epoch % save_interval == 0 or epoch == epochs:
            save_checkpoint(model, optimizer, scheduler, epoch, save_dir)


if __name__ == "__main__":
    main()
