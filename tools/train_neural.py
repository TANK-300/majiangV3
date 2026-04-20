#!/usr/bin/env python3
"""B-2a: train a small multi-head MLP on B-1 self-play samples, export ONNX.

This is the first step of Plan B's neural-network path. It deliberately stays
small and reproducible so we can exercise the *pipeline*:

    samples.jsonl  -->  train_neural.py  -->  model.pt  +  model.onnx

The MLP takes the per-step `model_features` dict (same keys the V3 search
engine already computes in C++) and outputs three heads:

    * agari_logit     -> sigmoid -> predicted P(I win)
    * houjuu_logit    -> sigmoid -> predicted P(I deal in)
    * tsumo_num_value -> regression on `steps_to_end`

The C++ inference side (B-2b) will load the exported ONNX with onnxruntime
and drop the predictions into `LinhaiSearchEngineV3`'s EV estimator in place
of the V3LinearModel. For now, the Python-side policy (`NeuralPolicy` in
tools/selfplay_eval.py) consumes the same ONNX to give us NN-vs-heuristic
win-rate numbers *without* needing the C++ rebuild.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class NeuralModelBundle:
    feature_names: List[str]
    feature_means: List[float]
    feature_stds: List[float]
    model_pt_path: Optional[Path]
    model_onnx_path: Optional[Path]
    metrics: Dict[str, float]

    def to_metadata(self) -> Dict:
        return {
            "model_type": "mlp_multihead",
            "heads": ["agari", "houjuu", "tsumo_num"],
            "feature_names": self.feature_names,
            "feature_means": {name: float(mean) for name, mean in zip(self.feature_names, self.feature_means)},
            "feature_stds": {name: float(std) for name, std in zip(self.feature_names, self.feature_stds)},
            "model_pt": str(self.model_pt_path) if self.model_pt_path else None,
            "model_onnx": str(self.model_onnx_path) if self.model_onnx_path else None,
            "metrics": self.metrics,
        }


def _load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stable_feature_names(rows: Iterable[Dict]) -> List[str]:
    names = set()
    for row in rows:
        features = row.get("model_features") or {}
        if isinstance(features, dict):
            names.update(features.keys())
    return sorted(names)


def _vectorize(
    rows: Sequence[Dict],
    feature_names: Sequence[str],
) -> Tuple[List[List[float]], Dict[str, float], Dict[str, float]]:
    X = []
    for row in rows:
        features = row.get("model_features") or {}
        if not isinstance(features, dict):
            features = {}
        X.append([float(features.get(name, 0.0)) for name in feature_names])

    n = max(len(X), 1)
    means = [sum(row[i] for row in X) / n for i in range(len(feature_names))]
    variances = [
        sum((row[i] - means[i]) ** 2 for row in X) / n for i in range(len(feature_names))
    ]
    stds = [max(1e-6, var ** 0.5) for var in variances]

    mean_map = {name: means[i] for i, name in enumerate(feature_names)}
    std_map = {name: stds[i] for i, name in enumerate(feature_names)}

    normalized = [
        [(row[i] - means[i]) / stds[i] for i in range(len(feature_names))]
        for row in X
    ]
    return normalized, mean_map, std_map


def _labels(rows: Sequence[Dict]) -> Tuple[List[float], List[float], List[float]]:
    y_agari: List[float] = []
    y_houjuu: List[float] = []
    y_tsumo: List[float] = []
    for row in rows:
        labels = row.get("task_labels", {})
        source_meta = row.get("source_meta", {})
        can_win = 1.0 if bool(labels.get("can_win_label", False)) else 0.0
        houjuu = float(source_meta.get("houjuu_label", 0))
        tsumo = float(source_meta.get("tsumo_num_label", 0.0))
        y_agari.append(can_win)
        y_houjuu.append(houjuu)
        y_tsumo.append(tsumo)
    return y_agari, y_houjuu, y_tsumo


def _build_module(n_features: int, hidden: int):
    import torch
    from torch import nn

    class LinhaiMLP(nn.Module):
        def __init__(self, n: int, h: int) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
            )
            self.head_agari = nn.Linear(h, 1)
            self.head_houjuu = nn.Linear(h, 1)
            self.head_tsumo = nn.Linear(h, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.trunk(x)
            out = torch.cat([self.head_agari(z), self.head_houjuu(z), self.head_tsumo(z)], dim=1)
            return out

    return LinhaiMLP(n_features, hidden)


def train_neural_bundle(
    dataset_path: Path,
    output_dir: Path,
    *,
    hidden: int = 64,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 0,
    export_onnx: bool = True,
) -> NeuralModelBundle:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)

    rows = _load_rows(dataset_path)
    if not rows:
        raise ValueError(f"No training rows in {dataset_path}")

    feature_names = _stable_feature_names(rows)
    if not feature_names:
        raise ValueError("Dataset rows have no `model_features` dicts; nothing to train on.")

    X_norm, mean_map, std_map = _vectorize(rows, feature_names)
    y_agari, y_houjuu, y_tsumo = _labels(rows)

    X = torch.tensor(X_norm, dtype=torch.float32)
    y_agari_t = torch.tensor(y_agari, dtype=torch.float32).unsqueeze(1)
    y_houjuu_t = torch.tensor(y_houjuu, dtype=torch.float32).unsqueeze(1)
    y_tsumo_t = torch.tensor(y_tsumo, dtype=torch.float32).unsqueeze(1)

    n = X.shape[0]
    perm = torch.randperm(n)
    val_count = max(1, int(n * val_split)) if n >= 5 else 0
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]

    train_dataset = TensorDataset(
        X[train_idx],
        y_agari_t[train_idx],
        y_houjuu_t[train_idx],
        y_tsumo_t[train_idx],
    )
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, max(1, len(train_dataset))), shuffle=True)

    model = _build_module(len(feature_names), hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    model.train()
    metrics: Dict[str, float] = {"epochs": float(epochs), "train_n": float(len(train_idx)), "val_n": float(len(val_idx))}
    for epoch in range(epochs):
        total = 0.0
        batches = 0
        for xb, ya, yh, yt in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = bce(out[:, 0:1], ya) + bce(out[:, 1:2], yh) + 0.02 * mse(out[:, 2:3], yt)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        metrics[f"train_loss_epoch_{epoch}"] = total / max(1, batches)

    model.eval()
    with torch.no_grad():
        out_all = model(X)
        agari_pred = torch.sigmoid(out_all[:, 0:1])
        houjuu_pred = torch.sigmoid(out_all[:, 1:2])
        tsumo_pred = out_all[:, 2:3]

    def _acc(pred: "torch.Tensor", label: "torch.Tensor") -> float:
        return float(((pred >= 0.5).float() == label).float().mean().item())

    metrics["train_agari_acc"] = _acc(agari_pred[train_idx], y_agari_t[train_idx])
    metrics["train_houjuu_acc"] = _acc(houjuu_pred[train_idx], y_houjuu_t[train_idx])
    metrics["train_tsumo_mae"] = float(torch.abs(tsumo_pred[train_idx] - y_tsumo_t[train_idx]).mean().item())
    if val_count > 0:
        metrics["val_agari_acc"] = _acc(agari_pred[val_idx], y_agari_t[val_idx])
        metrics["val_houjuu_acc"] = _acc(houjuu_pred[val_idx], y_houjuu_t[val_idx])
        metrics["val_tsumo_mae"] = float(torch.abs(tsumo_pred[val_idx] - y_tsumo_t[val_idx]).mean().item())

    output_dir.mkdir(parents=True, exist_ok=True)
    model_pt_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_pt_path)

    model_onnx_path: Optional[Path] = None
    if export_onnx:
        model_onnx_path = output_dir / "model.onnx"
        dummy = torch.zeros((1, len(feature_names)), dtype=torch.float32)
        torch.onnx.export(
            model,
            (dummy,),
            str(model_onnx_path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

    bundle = NeuralModelBundle(
        feature_names=feature_names,
        feature_means=[mean_map[name] for name in feature_names],
        feature_stds=[std_map[name] for name in feature_names],
        model_pt_path=model_pt_path,
        model_onnx_path=model_onnx_path,
        metrics=metrics,
    )
    (output_dir / "model_meta.json").write_text(
        json.dumps(bundle.to_metadata(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multi-head MLP on self-play samples (B-2a).")
    parser.add_argument("--dataset", required=True, help="JSONL file from tools/selfplay_sample.py")
    parser.add_argument("--output-dir", required=True, help="Directory to write model.pt/model.onnx/model_meta.json")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export (e.g. for unit tests)")
    args = parser.parse_args()

    bundle = train_neural_bundle(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        export_onnx=not args.no_onnx,
    )
    print(json.dumps(bundle.to_metadata(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
