from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_extract_canonical_states_outputs_normalized_record(tmp_path: Path) -> None:
    tool = Path("/Users/wf/Documents/wb/linhai-majiang-v3/tools/extract_canonical_states.py")
    source = tmp_path / "states.jsonl"
    target = tmp_path / "canonical.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "case-1",
                        "round_wind": "east",
                        "dealer_wind": "east",
                        "active_wind": "east",
                        "wall_remaining": 42,
                        "tree_active": True,
                        "grab_charge_active": True,
                        "target_hai": "3w",
                        "from_player": 1,
                        "available_actions": ["pass", "peng", "chi"],
                        "players": {
                            "east": {
                                "hand": ["1w", "2w", "3w", "3w", "white", "east"],
                                "melds": [{"type": "peng", "tiles": ["5w", "5w", "5w"]}],
                                "discards": ["9t"],
                                "tree_revealed": True,
                                "grab_charge_hits": 2,
                                "grab_charge_limit": 6,
                                "contract_targets": ["south"],
                                "contract_counter": 1,
                            },
                            "south": {
                                "hand": [],
                                "melds": [{"type": "chi", "tiles": ["1t", "2t", "3t"]}],
                                "discards": ["1w", "east"],
                            },
                        },
                        "label": {"best_action": "peng"},
                    },
                    ensure_ascii=False,
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.check_call([sys.executable, str(tool), "--input", str(source), "--output", str(target)])
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["record_id"] == "case-1"
    assert row["context_version"] == "v3"
    assert row["current_player"]["hand_counts"]["3w"] == 2
    assert row["current_player"]["hand_counts"]["white"] == 1
    assert row["white_tiles_in_hand"] == 1
    assert row["tree_active"] is True
    assert row["grab_charge_active"] is True
    assert row["contract_target_count"] == 1
    assert row["opponent_meld_count"] == 1
    assert row["opponent_discard_count"] == 2
    assert row["remaining_counts"]["white"] == 3
    assert row["target_hai"] == "3w"
    assert row["available_actions"] == ["pass", "peng", "chi"]
    assert row["label"]["best_action"] == "peng"
    assert row["task_labels"]["best_action"] == "peng"
    assert row["task_labels"]["best_response_action"] == "peng"
    assert row["model_features"]["white_tiles_in_hand"] == 1.0
    assert row["model_features"]["can_peng"] == 1.0
    assert row["model_features"]["opponent_meld_count"] == 1.0
    assert row["model_features"]["has_target_hai"] == 1.0
    assert row["model_features"]["target_in_hand_count"] == 2.0
    assert row["model_features"]["target_rank"] == 3.0
    assert row["model_features"]["target_is_honor"] == 0.0


def test_train_model_stub_creates_metadata(tmp_path: Path) -> None:
    tool = Path("/Users/wf/Documents/wb/linhai-majiang-v3/tools/train_model_stub.py")
    target = tmp_path / "params"
    subprocess.check_call([sys.executable, str(tool), "--task", "agari_prob", "--version-dir", str(target)])
    meta = target / "v3" / "agari_prob" / "model_meta.json"
    assert meta.exists()
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["task"] == "agari_prob"


def test_train_model_stub_summarizes_dataset(tmp_path: Path) -> None:
    tool = Path("/Users/wf/Documents/wb/linhai-majiang-v3/tools/train_model_stub.py")
    target = tmp_path / "params"
    dataset = tmp_path / "canonical.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "case-1",
                        "model_features": {
                            "wall_remaining": 40.0,
                            "white_tiles_in_hand": 1.0,
                            "can_win": 0.0,
                        },
                        "task_labels": {
                            "best_discard_tile": "east",
                            "can_win_label": False,
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "record_id": "case-2",
                        "model_features": {
                            "wall_remaining": 12.0,
                            "white_tiles_in_hand": 2.0,
                            "can_win": 1.0,
                        },
                        "task_labels": {},
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.check_call(
        [
            sys.executable,
            str(tool),
            "--task",
            "agari_prob",
            "--version-dir",
            str(target),
            "--dataset",
            str(dataset),
        ]
    )
    meta = target / "v3" / "agari_prob" / "model_meta.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["dataset"]["total_records"] == 2
    assert data["dataset"]["records_with_task_labels"] == 1
    assert data["dataset"]["label_key_counts"]["best_discard_tile"] == 1
    assert data["dataset"]["label_key_counts"]["can_win_label"] == 1
    assert data["status"] == "trained"
    assert data["trained_examples"] == 1
    model = json.loads((target / "v3" / "agari_prob" / "model.json").read_text(encoding="utf-8"))
    assert model["model_type"] == "logistic_regression"
    assert model["label_key"] == "task_labels.can_win_label"
    assert "white_tiles_in_hand" in model["feature_names"]


def test_train_model_stub_supports_explicit_label_key(tmp_path: Path) -> None:
    tool = Path("/Users/wf/Documents/wb/linhai-majiang-v3/tools/train_model_stub.py")
    target = tmp_path / "params"
    dataset = tmp_path / "canonical.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "case-1",
                        "model_features": {
                            "wall_remaining": 50.0,
                            "white_tiles_in_hand": 0.0,
                            "opponent_discard_count": 1.0,
                        },
                        "source_meta": {
                            "houjuu_label": 0,
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "record_id": "case-2",
                        "model_features": {
                            "wall_remaining": 8.0,
                            "white_tiles_in_hand": 2.0,
                            "opponent_discard_count": 10.0,
                        },
                        "source_meta": {
                            "houjuu_label": 1,
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.check_call(
        [
            sys.executable,
            str(tool),
            "--task",
            "houjuu_prob",
            "--version-dir",
            str(target),
            "--dataset",
            str(dataset),
            "--label-key",
            "source_meta.houjuu_label",
            "--epochs",
            "50",
        ]
    )
    meta = json.loads((target / "v3" / "houjuu_prob" / "model_meta.json").read_text(encoding="utf-8"))
    model = json.loads((target / "v3" / "houjuu_prob" / "model.json").read_text(encoding="utf-8"))
    assert meta["status"] == "trained"
    assert meta["label_key"] == "source_meta.houjuu_label"
    assert meta["trained_examples"] == 2
    assert model["model_type"] == "logistic_regression"
    assert model["training_examples"] == 2


def test_eval_search_compares_predictions_against_labels(tmp_path: Path) -> None:
    tool = Path("/Users/wf/Documents/wb/linhai-majiang-v3/tools/eval_search.py")
    labels = tmp_path / "labels.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    labels.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "discard-1",
                        "task_labels": {
                            "best_discard_tile": "east",
                            "can_win_label": True,
                        },
                        "source_meta": {
                            "tenpai_label": 1,
                            "houjuu_label": 0,
                            "betaori_label": 1,
                            "tsumo_num_label": 2.0,
                            "ryukyoku_label": 0,
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "record_id": "response-1",
                        "task_labels": {
                            "best_response_action": "peng",
                            "best_response_discard_tile": "1w",
                            "can_win_label": False,
                        },
                        "source_meta": {
                            "tenpai_label": 0,
                            "houjuu_label": 1,
                            "betaori_label": 0,
                            "tsumo_num_label": 5.0,
                            "ryukyoku_label": 1,
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "discard-1",
                        "action": "discard",
                        "tile": "east",
                        "predicted_can_win": True,
                        "agari_prob": 0.9,
                        "tenpai_prob": 0.8,
                        "houjuu_prob": 0.1,
                        "betaori_prob": 0.7,
                        "tsumo_num": 2.5,
                        "ryukyoku_prob": 0.2,
                        "chosen_by": "search",
                        "truncated": False,
                        "search_nodes": 12,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "record_id": "response-1",
                        "predicted_action": "peng",
                        "predicted_tile": "1w",
                        "predicted_can_win": False,
                        "agari_prob": 0.2,
                        "tenpai_prob": 0.3,
                        "houjuu_prob": 0.8,
                        "betaori_prob": 0.4,
                        "tsumo_num": 4.5,
                        "ryukyoku_prob": 0.7,
                        "chosen_by": "fallback_v2",
                        "truncated": True,
                        "search_nodes": 4,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = subprocess.check_output(
        [sys.executable, str(tool), "--predictions", str(predictions), "--labels", str(labels)],
        text=True,
    )
    metrics = json.loads(output)
    assert metrics["matched_records"] == 2
    assert metrics["discard_top1"] == 1.0
    assert metrics["response_action_top1"] == 1.0
    assert metrics["response_discard_top1"] == 1.0
    assert metrics["can_win_accuracy"] == 1.0
    assert metrics["fallback_rate"] == 0.5
    assert metrics["truncate_rate"] == 0.5
    assert metrics["avg_search_nodes"] == 8.0
    assert metrics["agari_prob_count"] == 2
    assert metrics["agari_prob_log_loss"] is not None
    assert metrics["agari_prob_brier"] is not None
    assert metrics["tenpai_prob_count"] == 2
    assert metrics["houjuu_prob_count"] == 2
    assert metrics["betaori_prob_count"] == 2
    assert metrics["ryukyoku_prob_count"] == 2
    assert metrics["tsumo_num_count"] == 2
    assert metrics["tsumo_num_mae"] == 0.5
    assert metrics["tsumo_num_rmse"] == 0.5
