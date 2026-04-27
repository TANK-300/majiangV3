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


def _flatten_lightgbm_tree(root: Dict) -> List[Dict]:
    """Flatten LightGBM's nested ``tree_structure`` into a list of nodes.

    Each node dict has a uniform shape::

        {"feat": int, "thr": float, "left": int, "right": int, "leaf_value": float}

    Leaf nodes use ``feat=-1`` and ``left=right=-1``; internal nodes reference
    their children by index into the same list. This matches exactly what the
    C++ ``V3GBDTModel`` loader expects in PR-2, so the on-disk ``model.json``
    stays portable.

    NOTE: LightGBM uses ``decision_type`` to indicate the split predicate
    (``"<="`` vs ``"<"`` vs ``"=="``). For numeric features we train here we
    always get ``"<="``; we assert this at parse time so we never silently
    produce a miscalibrated tree.
    """
    nodes: List[Dict] = []

    def visit(node: Dict) -> int:
        idx = len(nodes)
        nodes.append({})

        if "split_feature" not in node:
            leaf_value = node.get("leaf_value")
            if leaf_value is None:
                raise ValueError(f"malformed LightGBM leaf node: {node!r}")
            nodes[idx] = {
                "feat": -1,
                "thr": 0.0,
                "left": -1,
                "right": -1,
                "leaf_value": float(leaf_value),
            }
            return idx

        decision_type = node.get("decision_type", "<=")
        if decision_type not in ("<=",):
            raise ValueError(
                f"unsupported LightGBM decision_type {decision_type!r}; "
                "A-2b C++ loader only supports '<=' splits for now"
            )

        left_idx = visit(node["left_child"])
        right_idx = visit(node["right_child"])
        nodes[idx] = {
            "feat": int(node["split_feature"]),
            "thr": float(node["threshold"]),
            "left": left_idx,
            "right": right_idx,
            "leaf_value": 0.0,
        }
        return idx

    visit(root)
    return nodes


def _predict_flat_tree(nodes: Sequence[Dict], x: Sequence[float]) -> float:
    idx = 0
    while nodes[idx]["feat"] >= 0:
        node = nodes[idx]
        idx = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
    return float(nodes[idx]["leaf_value"])


def _try_train_with_lightgbm(
    x_rows: Sequence[Sequence[float]],
    labels: Sequence[float],
    is_binary: bool,
    validation_split: float,
    n_estimators: int,
    num_leaves: int,
    max_depth: int,
    gbdt_learning_rate: float,
    min_training_rows: int,
) -> Optional[Dict]:
    """A-2b: train a LightGBM GBDT and serialize it into the PR-2 schema.

    We prefer GBDT over the linear models because the linear family cannot
    express non-linear rules like "drop a lone honor tile first" (see the
    ``north`` regression case in ``docs/EVAL.md``). Falls back to ``None`` so
    callers can degrade to sklearn linear / handrolled SGD when:

    * LightGBM isn't installed (test containers, minimal CI),
    * there aren't enough rows to avoid overfitting to noise,
    * or the binary target has only one class.

    The returned dict intentionally omits ``feature_means`` / ``feature_stds``
    because GBDT doesn't need scaling. That also makes the emitted JSON
    **incompatible with the old linear C++ loader**, which will politely fail
    to load and fall back to heuristic -- i.e. safe degradation before PR-2
    ships the GBDT-aware C++ loader.
    """
    try:
        import lightgbm as lgb  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    n_rows = len(labels)
    if n_rows < max(min_training_rows, 1):
        return None
    if is_binary and len(set(labels)) < 2:
        return None

    # LightGBM >= 4 rejects raw list-of-list inputs (TypeError: "Data list
    # can only be of ndarray or Sequence"). Convert once up-front.
    x_array = np.asarray([list(row) for row in x_rows], dtype=np.float64)
    y_array = np.asarray([float(v) for v in labels], dtype=np.float64)

    x_val = None
    y_val = None
    if validation_split > 0.0 and n_rows >= 8:
        try:
            from sklearn.model_selection import train_test_split  # type: ignore
        except Exception:
            x_train, y_train = x_array, y_array
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                x_array,
                y_array,
                test_size=validation_split,
                random_state=0,
                stratify=y_array if is_binary and len(set(y_array.tolist())) == 2 else None,
            )
    else:
        x_train, y_train = x_array, y_array

    objective = "binary" if is_binary else "regression"
    params = {
        "objective": objective,
        "learning_rate": gbdt_learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        # tiny fixture datasets would otherwise violate min_data_in_leaf=20 default
        "min_data_in_leaf": max(1, len(y_train) // 20),
        "feature_pre_filter": False,
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
    }
    if is_binary:
        params["metric"] = "binary_logloss"
    else:
        params["metric"] = "rmse"

    train_ds = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    valid_sets = [train_ds]
    valid_names = ["train"]
    if x_val is not None and len(x_val) > 0:
        val_ds = lgb.Dataset(x_val, label=y_val, reference=train_ds, free_raw_data=False)
        valid_sets.append(val_ds)
        valid_names.append("val")

    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    dumped = booster.dump_model()
    trees_raw = dumped.get("tree_info", [])
    trees_flat: List[List[Dict]] = [_flatten_lightgbm_tree(t["tree_structure"]) for t in trees_raw]

    # A-2b: derive the implicit init_score (prior) by comparing LightGBM's own
    # raw prediction on a known row to the sum of leaf values computed via our
    # flat-tree format. If they disagree, the C++ inference would drift from
    # LightGBM's -- the delta captures the "init" constant LightGBM adds for
    # binary objectives (log(p/(1-p))).
    probe_source = x_train if len(x_train) > 0 else x_array
    probe_row = np.asarray(probe_source[0], dtype=np.float64)
    lgb_raw = float(booster.predict(probe_row.reshape(1, -1), raw_score=True)[0])
    my_raw = sum(_predict_flat_tree(tree, probe_row.tolist()) for tree in trees_flat)
    init_score = lgb_raw - my_raw

    # Sanity: the discrepancy must be a *constant* across samples. Probe a
    # few extra rows; if it drifts, our flat-tree semantics diverges from
    # LightGBM and we refuse to ship (caller falls back to linear).
    probe_extra_count = min(3, max(0, len(probe_source) - 1))
    for i in range(1, 1 + probe_extra_count):
        row = np.asarray(probe_source[i], dtype=np.float64)
        lgb_raw_i = float(booster.predict(row.reshape(1, -1), raw_score=True)[0])
        my_raw_i = sum(_predict_flat_tree(tree, row.tolist()) for tree in trees_flat)
        if abs((lgb_raw_i - my_raw_i) - init_score) > 1e-4:
            return None

    metrics: Dict[str, float] = {}
    has_val = x_val is not None and len(x_val) > 0
    if is_binary:
        preds_train = booster.predict(x_train)
        metrics["train_accuracy"] = float(
            sum((p >= 0.5) == (y >= 0.5) for p, y in zip(preds_train, y_train)) / max(len(y_train), 1)
        )
        train_logloss = -sum(
            (y * math.log(max(min(p, 1.0 - 1e-6), 1e-6))
             + (1.0 - y) * math.log(max(min(1.0 - p, 1.0 - 1e-6), 1e-6)))
            for p, y in zip(preds_train, y_train)
        ) / max(len(y_train), 1)
        metrics["train_logloss"] = float(train_logloss)
        if has_val:
            preds_val = booster.predict(x_val)
            metrics["val_accuracy"] = float(
                sum((p >= 0.5) == (y >= 0.5) for p, y in zip(preds_val, y_val)) / max(len(y_val), 1)
            )
    else:
        preds_train = booster.predict(x_train)
        metrics["train_rmse"] = float(
            math.sqrt(sum((p - y) ** 2 for p, y in zip(preds_train, y_train)) / max(len(y_train), 1))
        )
        metrics["train_mae"] = float(
            sum(abs(p - y) for p, y in zip(preds_train, y_train)) / max(len(y_train), 1)
        )
        if has_val:
            preds_val = booster.predict(x_val)
            metrics["val_rmse"] = float(
                math.sqrt(sum((p - y) ** 2 for p, y in zip(preds_val, y_val)) / max(len(y_val), 1))
            )

    return {
        "model_type": "lightgbm_gbdt",
        "objective": objective,
        "init_score": init_score,
        # `trees[i].nodes` is the flat list produced by _flatten_lightgbm_tree;
        # wrapping it in a dict gives us room to add per-tree metadata later
        # (e.g. shrinkage) without another schema break.
        "trees": [{"nodes": tree} for tree in trees_flat],
        "n_trees": len(trees_flat),
        "metrics": metrics,
        "_fitter": "lightgbm",
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
    model_kind: str = "auto",
    n_estimators: int = 200,
    num_leaves: int = 31,
    max_depth: int = -1,
    gbdt_learning_rate: float = 0.05,
    gbdt_min_rows: int = 200,
) -> Dict:
    """Train a single-head model and serialize it for the C++ V3 engine.

    The ``model_kind`` selector controls the algorithm family:

    * ``"auto"`` (default): try LightGBM GBDT first (A-2b), then sklearn
      linear / ridge (A-2a), then the handrolled SGD (stub).
    * ``"gbdt"``: force LightGBM GBDT; return a failure-shaped record if
      LightGBM is unavailable instead of silently downgrading.
    * ``"linear"``: explicitly skip GBDT and use the linear family. Useful for
      pinned old deployments that haven't shipped the PR-2 C++ loader yet.
    """
    feature_names = collect_feature_names(rows)
    means, stds = compute_scaler(rows, feature_names)
    x_rows = vectorize_rows(rows, feature_names, means, stds)
    # GBDT uses the raw unscaled features (tree splits are scale-invariant);
    # this also lets the C++ side use the same name->value lookup table and
    # skip the means/stds step entirely at inference time.
    x_rows_raw = [[row.get(name, 0.0) for name in feature_names] for row in rows]
    is_binary = all(label in {0.0, 1.0} for label in labels)

    try_gbdt = model_kind in ("auto", "gbdt")
    try_linear_sklearn = model_kind in ("auto", "linear")
    allow_handrolled = model_kind in ("auto", "linear")

    gbdt_fit = None
    if try_gbdt:
        gbdt_fit = _try_train_with_lightgbm(
            x_rows_raw,
            labels,
            is_binary,
            validation_split=validation_split,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            gbdt_learning_rate=gbdt_learning_rate,
            min_training_rows=gbdt_min_rows,
        )

    if gbdt_fit is not None:
        fitter = gbdt_fit.pop("_fitter", "lightgbm")
        # A-2b schema: we intentionally *do not* persist feature_means/stds
        # because tree splits are scale-invariant; a stale linear C++ loader
        # will fail to parse this and safely fall back to heuristic.
        return {
            **gbdt_fit,
            "feature_names": feature_names,
            "training_examples": len(labels),
            "fitter": fitter,
        }

    if model_kind == "gbdt":
        # Caller explicitly asked for GBDT but we couldn't produce one
        # (LightGBM not installed, too few rows, or degenerate labels). Emit a
        # diagnostic shape instead of silently downgrading -- the release
        # script then refuses to promote this bundle.
        return {
            "model_type": "gbdt_unavailable",
            "feature_names": feature_names,
            "feature_means": means,
            "feature_stds": stds,
            "training_examples": len(labels),
            "fitter": "none",
            "metrics": {},
            "reason": "lightgbm_not_available_or_insufficient_data",
        }

    if try_linear_sklearn:
        sk_fit = _try_train_with_sklearn(x_rows, labels, is_binary, l2, validation_split)
    else:
        sk_fit = None

    if sk_fit is not None:
        fitted = sk_fit
    elif allow_handrolled and is_binary:
        fitted = fit_logistic_regression(x_rows, labels, learning_rate, epochs, l2)
        fitted["_fitter"] = "handrolled"
    elif allow_handrolled:
        fitted = fit_linear_regression(x_rows, labels, learning_rate, epochs, l2)
        fitted["_fitter"] = "handrolled"
    else:
        return {
            "model_type": "unavailable",
            "feature_names": feature_names,
            "feature_means": means,
            "feature_stds": stds,
            "training_examples": len(labels),
            "fitter": "none",
            "metrics": {},
            "reason": "no_fitter_available_for_model_kind",
        }

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
    # A-2b: GBDT controls
    parser.add_argument(
        "--model-kind",
        choices=["auto", "gbdt", "linear"],
        default="auto",
        help=(
            "Which algorithm family to use. 'auto' tries LightGBM GBDT first "
            "(A-2b) and falls back to sklearn linear / handrolled SGD; "
            "'gbdt' requires LightGBM; 'linear' skips GBDT entirely."
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of boosting rounds for GBDT.")
    parser.add_argument("--num-leaves", type=int, default=31, help="Max leaves per GBDT tree.")
    parser.add_argument("--max-depth", type=int, default=-1, help="Max depth per GBDT tree (-1 = unlimited).")
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.05, help="Learning rate for GBDT.")
    parser.add_argument(
        "--gbdt-min-rows",
        type=int,
        default=200,
        help="Minimum labeled rows before GBDT is attempted; otherwise fall back to linear.",
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
        model_kind=args.model_kind,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        gbdt_learning_rate=args.gbdt_learning_rate,
        gbdt_min_rows=args.gbdt_min_rows,
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
