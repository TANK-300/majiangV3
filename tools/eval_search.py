#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional


PROBABILITY_TARGETS = {
    "agari_prob": "task_labels.can_win_label",
    "tenpai_prob": "source_meta.tenpai_label",
    "houjuu_prob": "source_meta.houjuu_label",
    "betaori_prob": "source_meta.betaori_label",
    "ryukyoku_prob": "source_meta.ryukyoku_label",
}

REGRESSION_TARGETS = {
    "tsumo_num": "source_meta.tsumo_num_label",
}


def load_jsonl(path: Path) -> list[Dict]:
    rows: list[Dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_tile(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def normalize_action(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def normalize_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None


def normalize_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def clamp_probability(value: float) -> float:
    return min(max(value, 1e-6), 1.0 - 1e-6)


def get_nested_value(row: Dict, dotted_key: str) -> Optional[object]:
    current: object = row
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def build_label_index(label_rows: Iterable[Dict]) -> Dict[str, Dict]:
    return {str(row["record_id"]): row for row in label_rows if row.get("record_id")}


def extract_prediction_fields(row: Dict) -> Dict[str, object]:
    return {
        "record_id": str(row.get("record_id")) if row.get("record_id") is not None else None,
        "action": normalize_action(row.get("predicted_action", row.get("action"))),
        "tile": normalize_tile(row.get("predicted_tile", row.get("tile"))),
        "can_win": row.get("predicted_can_win", row.get("can_win")),
        "agari_prob": row.get("agari_prob"),
        "tenpai_prob": row.get("tenpai_prob"),
        "houjuu_prob": row.get("houjuu_prob"),
        "betaori_prob": row.get("betaori_prob"),
        "tsumo_num": row.get("tsumo_num"),
        "ryukyoku_prob": row.get("ryukyoku_prob"),
        "chosen_by": row.get("chosen_by"),
        "truncated": row.get("truncated"),
        "search_nodes": row.get("search_nodes"),
    }


def evaluate_probability_task(
    matched_predictions: Iterable[Dict[str, object]],
    label_index: Dict[str, Dict],
    prediction_key: str,
    label_key: str,
) -> Dict[str, Optional[float]]:
    total = 0
    log_loss_sum = 0.0
    brier_sum = 0.0

    for prediction in matched_predictions:
        record_id = prediction["record_id"]
        if not isinstance(record_id, str):
            continue
        label_value = normalize_bool(get_nested_value(label_index[record_id], label_key))
        predicted_value = normalize_float(prediction.get(prediction_key))
        if label_value is None or predicted_value is None:
            continue
        total += 1
        label_num = 1.0 if label_value else 0.0
        probability = clamp_probability(predicted_value)
        log_loss_sum += -(label_num * math.log(probability) + (1.0 - label_num) * math.log(1.0 - probability))
        brier_sum += (probability - label_num) ** 2

    return {
        f"{prediction_key}_count": total,
        f"{prediction_key}_log_loss": log_loss_sum / total if total else None,
        f"{prediction_key}_brier": brier_sum / total if total else None,
    }


def evaluate_regression_task(
    matched_predictions: Iterable[Dict[str, object]],
    label_index: Dict[str, Dict],
    prediction_key: str,
    label_key: str,
) -> Dict[str, Optional[float]]:
    total = 0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0

    for prediction in matched_predictions:
        record_id = prediction["record_id"]
        if not isinstance(record_id, str):
            continue
        label_value = normalize_float(get_nested_value(label_index[record_id], label_key))
        predicted_value = normalize_float(prediction.get(prediction_key))
        if label_value is None or predicted_value is None:
            continue
        total += 1
        error = predicted_value - label_value
        absolute_error_sum += abs(error)
        squared_error_sum += error * error

    return {
        f"{prediction_key}_count": total,
        f"{prediction_key}_mae": absolute_error_sum / total if total else None,
        f"{prediction_key}_rmse": math.sqrt(squared_error_sum / total) if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search predictions against labels.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", help="Optional canonical JSONL containing task_labels/source_meta")
    args = parser.parse_args()

    prediction_rows = load_jsonl(Path(args.predictions))
    total = len(prediction_rows)
    if not args.labels:
        print(json.dumps({"records": total, "status": "ok"}, ensure_ascii=False))
        return

    label_rows = load_jsonl(Path(args.labels))
    label_index = build_label_index(label_rows)
    matched_predictions: list[Dict[str, object]] = []

    matched_records = 0
    discard_total = 0
    discard_exact = 0
    response_total = 0
    response_exact = 0
    response_discard_total = 0
    response_discard_exact = 0
    can_win_total = 0
    can_win_exact = 0
    fallback_total = 0
    truncated_total = 0
    search_nodes_total = 0.0
    search_nodes_count = 0

    for prediction_row in prediction_rows:
        fields = extract_prediction_fields(prediction_row)
        record_id = fields["record_id"]
        if not isinstance(record_id, str) or record_id not in label_index:
            continue

        matched_records += 1
        matched_predictions.append(fields)
        labels = label_index[record_id].get("task_labels", {})
        if not isinstance(labels, dict):
            labels = {}

        if labels.get("best_discard_tile"):
            discard_total += 1
            if fields["tile"] == normalize_tile(labels.get("best_discard_tile")):
                discard_exact += 1

        if labels.get("best_response_action"):
            response_total += 1
            if fields["action"] == normalize_action(labels.get("best_response_action")):
                response_exact += 1

        if labels.get("best_response_discard_tile"):
            response_discard_total += 1
            if fields["tile"] == normalize_tile(labels.get("best_response_discard_tile")):
                response_discard_exact += 1

        if "can_win_label" in labels:
            can_win_total += 1
            predicted_can_win = normalize_bool(fields["can_win"])
            if predicted_can_win is not None and predicted_can_win == bool(labels.get("can_win_label")):
                can_win_exact += 1

        chosen_by = str(fields.get("chosen_by") or "").strip().lower()
        if chosen_by.startswith("fallback"):
            fallback_total += 1
        truncated = normalize_bool(fields.get("truncated"))
        if truncated:
            truncated_total += 1
        search_nodes = normalize_float(fields.get("search_nodes"))
        if search_nodes is not None:
            search_nodes_total += search_nodes
            search_nodes_count += 1

    metrics = {
        "records": total,
        "matched_records": matched_records,
        "discard_top1": discard_exact / discard_total if discard_total else None,
        "response_action_top1": response_exact / response_total if response_total else None,
        "response_discard_top1": response_discard_exact / response_discard_total if response_discard_total else None,
        "can_win_accuracy": can_win_exact / can_win_total if can_win_total else None,
        "fallback_rate": fallback_total / matched_records if matched_records else None,
        "truncate_rate": truncated_total / matched_records if matched_records else None,
        "avg_search_nodes": search_nodes_total / search_nodes_count if search_nodes_count else None,
        "status": "ok",
    }

    for prediction_key, label_key in PROBABILITY_TARGETS.items():
        metrics.update(evaluate_probability_task(matched_predictions, label_index, prediction_key, label_key))
    for prediction_key, label_key in REGRESSION_TARGETS.items():
        metrics.update(evaluate_regression_task(matched_predictions, label_index, prediction_key, label_key))

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
