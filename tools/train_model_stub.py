#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TASK_LABEL_KEYS = {
    "agari_prob": "task_labels.can_win_label",
    "tenpai_prob": "source_meta.tenpai_label",
    "houjuu_prob": "source_meta.houjuu_label",
    "betaori": "source_meta.betaori_label",
    "tsumo_num": "source_meta.tsumo_num_label",
    "ryukyoku_prob": "source_meta.ryukyoku_label",
}


def summarize_dataset(path: Path) -> Dict:
    total_records = 0
    records_with_task_labels = 0
    label_key_counts: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_records += 1
            task_labels = row.get("task_labels", {})
            if isinstance(task_labels, dict) and task_labels:
                records_with_task_labels += 1
                for key, value in task_labels.items():
                    if value is None:
                        continue
                    label_key_counts[key] = label_key_counts.get(key, 0) + 1

    return {
        "dataset_path": str(path),
        "total_records": total_records,
        "records_with_task_labels": records_with_task_labels,
        "label_key_counts": label_key_counts,
    }


def get_nested_value(row: Dict, dotted_key: str) -> Optional[object]:
    current: object = row
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def infer_model_features(row: Dict) -> Dict[str, float]:
    features = row.get("model_features")
    if isinstance(features, dict) and features:
        return {str(key): float(value) for key, value in features.items() if isinstance(value, (int, float, bool))}

    current_player = row.get("current_player", {})
    hand_counts = current_player.get("hand_counts", {})
    if not isinstance(hand_counts, dict):
        hand_counts = {}
    remaining_counts = row.get("remaining_counts", {})
    if not isinstance(remaining_counts, dict):
        remaining_counts = {}

    features = {
        "wall_remaining": float(row.get("wall_remaining", 0)),
        "white_tiles_in_hand": float(row.get("white_tiles_in_hand", hand_counts.get("white", 0))),
        "tree_active": 1.0 if row.get("tree_active", False) else 0.0,
        "grab_charge_active": 1.0 if row.get("grab_charge_active", False) else 0.0,
        "grab_charge_hits": float(row.get("grab_charge_hits", 0)),
        "grab_charge_limit": float(row.get("grab_charge_limit", 0)),
        "contract_target_count": float(row.get("contract_target_count", 0)),
        "contract_counter": float(row.get("contract_counter", 0)),
        "opponent_meld_count": float(row.get("opponent_meld_count", 0)),
        "opponent_discard_count": float(row.get("opponent_discard_count", 0)),
        "hand_size": float(sum(int(value) for value in hand_counts.values() if isinstance(value, (int, float)))),
        "distinct_tile_count": float(sum(1 for value in hand_counts.values() if isinstance(value, (int, float)) and value > 0)),
        "remaining_total": float(sum(int(value) for value in remaining_counts.values() if isinstance(value, (int, float)))),
        "can_win": 1.0 if row.get("can_win", False) else 0.0,
        "available_action_count": float(len(row.get("available_actions", []))),
    }
    return features


def normalize_label(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_training_rows(path: Path, label_key: str) -> Tuple[List[Dict[str, float]], List[float], int]:
    features_rows: List[Dict[str, float]] = []
    labels: List[float] = []
    skipped = 0

    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label_value = normalize_label(get_nested_value(row, label_key))
            if label_value is None:
                skipped += 1
                continue
            features_rows.append(infer_model_features(row))
            labels.append(label_value)

    return features_rows, labels, skipped


def collect_feature_names(rows: Sequence[Dict[str, float]]) -> List[str]:
    names = set()
    for row in rows:
        names.update(row.keys())
    return sorted(names)


def compute_scaler(rows: Sequence[Dict[str, float]], feature_names: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    count = float(len(rows))
    for name in feature_names:
        mean = sum(row.get(name, 0.0) for row in rows) / count
        variance = sum((row.get(name, 0.0) - mean) ** 2 for row in rows) / count
        means[name] = mean
        stds[name] = math.sqrt(variance) or 1.0
    return means, stds


def vectorize_rows(
    rows: Sequence[Dict[str, float]],
    feature_names: Sequence[str],
    means: Dict[str, float],
    stds: Dict[str, float],
) -> List[List[float]]:
    vectors: List[List[float]] = []
    for row in rows:
        vectors.append([(row.get(name, 0.0) - means[name]) / stds[name] for name in feature_names])
    return vectors


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def fit_logistic_regression(
    x_rows: Sequence[Sequence[float]],
    labels: Sequence[float],
    learning_rate: float,
    epochs: int,
    l2: float,
) -> Dict:
    feature_count = len(x_rows[0]) if x_rows else 0
    weights = [0.0] * feature_count
    positive_rate = min(max(sum(labels) / len(labels), 1e-4), 1.0 - 1e-4)
    intercept = math.log(positive_rate / (1.0 - positive_rate))

    sample_count = float(len(labels))
    for _ in range(epochs):
        grad_intercept = 0.0
        grad_weights = [0.0] * feature_count
        for features, label in zip(x_rows, labels):
            linear = intercept + sum(weight * value for weight, value in zip(weights, features))
            prediction = sigmoid(linear)
            error = prediction - label
            grad_intercept += error
            for idx, value in enumerate(features):
                grad_weights[idx] += error * value

        intercept -= learning_rate * (grad_intercept / sample_count)
        for idx in range(feature_count):
            grad = grad_weights[idx] / sample_count + l2 * weights[idx]
            weights[idx] -= learning_rate * grad

    predictions = [sigmoid(intercept + sum(weight * value for weight, value in zip(weights, features))) for features in x_rows]
    eps = 1e-6
    log_loss = -sum(
        label * math.log(max(min(prediction, 1.0 - eps), eps))
        + (1.0 - label) * math.log(max(min(1.0 - prediction, 1.0 - eps), eps))
        for prediction, label in zip(predictions, labels)
    ) / sample_count
    accuracy = sum((prediction >= 0.5) == (label >= 0.5) for prediction, label in zip(predictions, labels)) / sample_count

    return {
        "model_type": "logistic_regression",
        "intercept": intercept,
        "weights": weights,
        "metrics": {
            "log_loss": log_loss,
            "accuracy": accuracy,
            "positive_rate": sum(labels) / len(labels),
        },
    }


def fit_linear_regression(
    x_rows: Sequence[Sequence[float]],
    labels: Sequence[float],
    learning_rate: float,
    epochs: int,
    l2: float,
) -> Dict:
    feature_count = len(x_rows[0]) if x_rows else 0
    weights = [0.0] * feature_count
    intercept = sum(labels) / len(labels)
    sample_count = float(len(labels))

    for _ in range(epochs):
        grad_intercept = 0.0
        grad_weights = [0.0] * feature_count
        for features, label in zip(x_rows, labels):
            prediction = intercept + sum(weight * value for weight, value in zip(weights, features))
            error = prediction - label
            grad_intercept += error
            for idx, value in enumerate(features):
                grad_weights[idx] += error * value

        intercept -= learning_rate * (2.0 * grad_intercept / sample_count)
        for idx in range(feature_count):
            grad = 2.0 * grad_weights[idx] / sample_count + l2 * weights[idx]
            weights[idx] -= learning_rate * grad

    predictions = [intercept + sum(weight * value for weight, value in zip(weights, features)) for features in x_rows]
    mse = sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels)) / sample_count
    mae = sum(abs(prediction - label) for prediction, label in zip(predictions, labels)) / sample_count

    return {
        "model_type": "linear_regression",
        "intercept": intercept,
        "weights": weights,
        "metrics": {
            "mse": mse,
            "mae": mae,
            "target_mean": sum(labels) / len(labels),
        },
    }


def _try_train_with_sklearn(
    x_rows: Sequence[Sequence[float]],
    labels: Sequence[float],
    is_binary: bool,
    l2: float,
    validation_split: float,
) -> Optional[Dict]:
    """Train with scikit-learn if available, returning a payload compatible
    with the stub structure. Falls back to None so callers use the hand-rolled
    implementation for environments without sklearn.

    A-2a: the fitter here is the *same family* of model (logistic / ridge
    linear regression) that the C++ V3LinearModel loader expects, so the
    model.json emitted remains binary-compatible with the existing search
    engine -- but we get proper L2, convergence, and train/val metrics instead
    of the toy hand-rolled SGD.
    """
    try:
        from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
    except Exception:
        return None

    x_array: List[List[float]] = [list(row) for row in x_rows]
    y_array: List[float] = [float(v) for v in labels]

    if validation_split > 0.0 and len(y_array) >= 4:
        x_train, x_val, y_train, y_val = train_test_split(
            x_array,
            y_array,
            test_size=validation_split,
            random_state=0,
            stratify=y_array if is_binary and len(set(y_array)) == 2 else None,
        )
    else:
        x_train, x_val = x_array, []
        y_train, y_val = y_array, []

    metrics: Dict[str, float] = {}

    if is_binary:
        # A-2a: sklearn LogisticRegression requires >= 2 classes in y_train.
        # Tiny unit-test datasets or severely skewed real datasets can violate
        # that, so we degrade to the hand-rolled estimator in that case
        # (it silently clamps to the majority class prior, which is fine for
        # a smoke-level "model exists" guarantee).
        if len(set(y_train)) < 2:
            return None
        model = LogisticRegression(
            C=(1.0 / l2) if l2 > 0 else 1.0,
            max_iter=1000,
            solver="lbfgs",
        )
        model.fit(x_train, y_train)
        intercept = float(model.intercept_[0])
        weights = [float(w) for w in model.coef_[0]]
        model_type = "logistic_regression"
        metrics["train_accuracy"] = float(model.score(x_train, y_train))
        if x_val:
            metrics["val_accuracy"] = float(model.score(x_val, y_val))
    else:
        model = Ridge(alpha=max(l2, 1e-6))
        model.fit(x_train, y_train)
        intercept = float(model.intercept_)
        weights = [float(w) for w in model.coef_]
        model_type = "linear_regression"
        metrics["train_r2"] = float(model.score(x_train, y_train))
        if x_val:
            metrics["val_r2"] = float(model.score(x_val, y_val))

    return {
        "model_type": model_type,
        "intercept": intercept,
        "weights": weights,
        "metrics": metrics,
        "_fitter": "sklearn",
    }


def train_model(
    rows: Sequence[Dict[str, float]],
    labels: Sequence[float],
    learning_rate: float,
    epochs: int,
    l2: float,
    validation_split: float = 0.0,
) -> Dict:
    feature_names = collect_feature_names(rows)
    means, stds = compute_scaler(rows, feature_names)
    x_rows = vectorize_rows(rows, feature_names, means, stds)
    is_binary = all(label in {0.0, 1.0} for label in labels)

    sk_fit = _try_train_with_sklearn(x_rows, labels, is_binary, l2, validation_split)
    if sk_fit is not None:
        fitted = sk_fit
    elif is_binary:
        fitted = fit_logistic_regression(x_rows, labels, learning_rate, epochs, l2)
        fitted["_fitter"] = "handrolled"
    else:
        fitted = fit_linear_regression(x_rows, labels, learning_rate, epochs, l2)
        fitted["_fitter"] = "handrolled"

    fitter = fitted.pop("_fitter", "handrolled")

    return {
        **fitted,
        "feature_names": feature_names,
        "feature_means": means,
        "feature_stds": stds,
        "training_examples": len(labels),
        "fitter": fitter,
    }


def write_placeholder_metadata(output_dir: Path, task: str, dataset_path: Optional[Path]) -> Dict:
    metadata = {
        "task": task,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "placeholder",
    }
    if dataset_path:
        metadata["dataset"] = summarize_dataset(dataset_path)
    (output_dir / "model_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal versioned v3 model bundle.")
    parser.add_argument("--task", required=True, choices=["agari_prob", "tenpai_prob", "houjuu_prob", "betaori", "tsumo_num", "ryukyoku_prob"])
    parser.add_argument("--version-dir", required=True)
    parser.add_argument("--dataset", help="Optional canonical JSONL dataset used for training")
    parser.add_argument("--label-key", help="Optional dotted label key path, e.g. task_labels.can_win_label")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help="Fraction of rows held out for validation metrics (sklearn path only).",
    )
    args = parser.parse_args()

    output_dir = Path(args.version_dir) / "v3" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset) if args.dataset else None
    label_key = args.label_key or TASK_LABEL_KEYS.get(args.task)

    if dataset_path is None or not label_key:
        metadata = write_placeholder_metadata(output_dir, args.task, dataset_path)
        print(json.dumps({"created": str(output_dir / "model_meta.json"), "status": metadata["status"]}, ensure_ascii=False))
        return

    rows, labels, skipped = load_training_rows(dataset_path, label_key)
    if not rows:
        metadata = {
            "task": args.task,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "no_labeled_rows",
            "label_key": label_key,
            "dataset": summarize_dataset(dataset_path),
            "skipped_records": skipped,
        }
        (output_dir / "model_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"created": str(output_dir / "model_meta.json"), "status": metadata["status"]}, ensure_ascii=False))
        return

    trained_model = train_model(
        rows,
        labels,
        args.learning_rate,
        args.epochs,
        args.l2,
        validation_split=args.validation_split,
    )
    model_payload = {
        "task": args.task,
        "label_key": label_key,
        **trained_model,
    }
    model_path = output_dir / "model.json"
    model_path.write_text(json.dumps(model_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metadata = {
        "task": args.task,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained",
        "label_key": label_key,
        "dataset": summarize_dataset(dataset_path),
        "trained_examples": len(labels),
        "skipped_records": skipped,
        "model_file": str(model_path),
        "model_type": trained_model["model_type"],
        "metrics": trained_model["metrics"],
    }
    (output_dir / "model_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "created": str(output_dir / "model_meta.json"),
                "model": str(model_path),
                "status": metadata["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
