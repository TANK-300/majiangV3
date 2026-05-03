#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.tiles import Honor, Suit, all_tile_codes, normalize_code, require_tile


ALL_TILE_CODES = sorted(all_tile_codes())
WIND_ORDER = ["east", "south", "west", "north"]

# A-2a: per-tile features. We emit one entry per tile in a STABLE ordering so
# the feature names match exactly between the Python trainer and the C++
# search-engine inference path. The names here map 1:1 to the keys produced
# by `LinhaiSearchEngineV3::build_state_features()` in
# engine/share/linhai_search_v3.cpp.
PER_TILE_FEATURE_ORDER: List[str] = (
    [f"{rank}w" for rank in range(1, 10)]
    + [f"{rank}t" for rank in range(1, 10)]
    + ["east", "south", "west", "north", "white", "green", "red"]
)


def _feature_key_for_tile(prefix: str, tile_code: str) -> str:
    return f"{prefix}_{tile_code}"


def normalize_tiles(tiles: Optional[Iterable[str]]) -> List[str]:
    return [normalize_code(tile) for tile in (tiles or [])]


def normalize_wind(value: Optional[str], default: str = "east") -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    return normalized if normalized in WIND_ORDER else default


def count_tiles(tiles: Iterable[str]) -> Dict[str, int]:
    counts = {code: 0 for code in ALL_TILE_CODES}
    for tile in tiles:
        counts[normalize_code(tile)] += 1
    return counts


def normalize_melds(melds: Optional[Iterable[Dict]]) -> List[Dict]:
    normalized: List[Dict] = []
    for meld in melds or []:
        if not isinstance(meld, dict):
            continue
        normalized.append(
            {
                "type": str(meld.get("type", "unknown")),
                "tiles": normalize_tiles(meld.get("tiles", [])),
            }
        )
    return normalized


def build_visible_counts(players: Dict[str, Dict]) -> Dict[str, int]:
    visible = {code: 0 for code in ALL_TILE_CODES}
    for player in players.values():
        for tile in player["hand"]:
            visible[tile] += 1
        for tile in player["discards"]:
            visible[tile] += 1
        for meld in player["melds"]:
            for tile in meld["tiles"]:
                visible[tile] += 1
    return visible


def build_remaining_counts(visible_counts: Dict[str, int]) -> Dict[str, int]:
    return {tile: max(0, 4 - visible_counts[tile]) for tile in ALL_TILE_CODES}


def parse_player(raw: Dict, wind: str) -> Dict:
    hand = normalize_tiles(raw.get("hand", []))
    return {
        "wind": wind,
        "hand": hand,
        "hand_counts": count_tiles(hand),
        "melds": normalize_melds(raw.get("melds", [])),
        "discards": normalize_tiles(raw.get("discards", [])),
        "tree_revealed": bool(raw.get("tree_revealed", False)),
        "grab_charge_hits": int(raw.get("grab_charge_hits", 0)),
        "grab_charge_limit": int(raw.get("grab_charge_limit", 0)),
        "contract_targets": [normalize_wind(value, default="east") for value in raw.get("contract_targets", [])],
        "contract_counter": int(raw.get("contract_counter", 0)),
        "passed_hu_this_round": bool(raw.get("passed_hu_this_round", False) or raw.get("pass_hu_this_round", False)),
    }


def build_player_snapshots(players: Dict[str, Dict]) -> List[Dict]:
    snapshots: List[Dict] = []
    for wind in WIND_ORDER:
        player = players[wind]
        snapshots.append(
            {
                "wind": wind,
                "discard_count": len(player["discards"]),
                "meld_count": len(player["melds"]),
                "hand_count": len(player["hand"]),
            }
        )
    return snapshots


def build_task_labels(payload: Dict, available_actions: List[str]) -> Dict:
    raw_label = payload.get("label")
    if not isinstance(raw_label, dict):
        return {}

    task_labels: Dict[str, object] = {}
    best_action = raw_label.get("best_action")
    if isinstance(best_action, str) and best_action.strip():
        task_labels["best_action"] = best_action.strip().lower()

    best_tile = raw_label.get("best_tile")
    if isinstance(best_tile, str) and best_tile.strip():
        task_labels["best_tile"] = normalize_code(best_tile)

    best_discard_tile = raw_label.get("best_discard_tile")
    if isinstance(best_discard_tile, str) and best_discard_tile.strip():
        task_labels["best_discard_tile"] = normalize_code(best_discard_tile)
    elif task_labels.get("best_action") == "discard" and "best_tile" in task_labels:
        task_labels["best_discard_tile"] = task_labels["best_tile"]

    best_response_action = raw_label.get("best_response_action")
    if isinstance(best_response_action, str) and best_response_action.strip():
        task_labels["best_response_action"] = best_response_action.strip().lower()
    elif task_labels.get("best_action") in set(available_actions) | {"pass", "peng", "chi", "gang", "hu", "guo"}:
        task_labels["best_response_action"] = task_labels["best_action"]

    best_response_discard_tile = raw_label.get("best_response_discard_tile")
    if isinstance(best_response_discard_tile, str) and best_response_discard_tile.strip():
        task_labels["best_response_discard_tile"] = normalize_code(best_response_discard_tile)

    can_win_label = raw_label.get("can_win")
    if can_win_label is not None:
        task_labels["can_win_label"] = bool(can_win_label)

    return task_labels


def build_model_features(
    active_player: Dict,
    remaining_counts: Dict[str, int],
    available_actions: List[str],
    can_win: bool,
    wall_remaining: int,
    from_player: int,
    target_hai: Optional[str],
    tree_active: bool,
    grab_charge_active: bool,
    contract_target_count: int,
    opponent_meld_count: int,
    opponent_discard_count: int,
    opponent_discards: Optional[List[str]] = None,
) -> Dict[str, float]:
    hand_counts = active_player["hand_counts"]
    hand_size = len(active_player["hand"])
    meld_count = len(active_player["melds"])
    discard_count = len(active_player["discards"])

    pair_count = 0
    triplet_count = 0
    quad_count = 0
    distinct_tile_count = 0
    suited_count = 0
    honor_count = 0
    terminal_count = 0
    simple_count = 0
    wind_count = 0
    dragon_count = 0

    for code, count in hand_counts.items():
        if count <= 0:
            continue
        distinct_tile_count += 1
        if count >= 2:
            pair_count += 1
        if count >= 3:
            triplet_count += 1
        if count >= 4:
            quad_count += 1

        tile = require_tile(code)
        if tile.suit is Suit.HONOR:
            honor_count += count
            if tile.honor in {Honor.EAST, Honor.SOUTH, Honor.WEST, Honor.NORTH}:
                wind_count += count
            else:
                dragon_count += count
        else:
            suited_count += count
            if tile.rank in {1, 9}:
                terminal_count += count
            else:
                simple_count += count

    remaining_total = sum(remaining_counts.values())
    target_code = normalize_code(target_hai) if target_hai else None
    target_in_hand_count = hand_counts.get(target_code, 0) if target_code else 0
    target_rank = 0
    target_is_honor = 0.0
    target_is_terminal = 0.0
    target_is_white = 0.0
    if target_code:
        target_tile = require_tile(target_code)
        if target_tile.suit is Suit.HONOR:
            target_is_honor = 1.0
            target_is_white = 1.0 if target_tile.honor is Honor.WHITE else 0.0
        else:
            target_rank = target_tile.rank or 0
            target_is_terminal = 1.0 if target_rank in {1, 9} else 0.0

    action_set = set(available_actions)
    opponent_discard_counts: Dict[str, int] = {code: 0 for code in ALL_TILE_CODES}
    for tile in opponent_discards or []:
        opponent_discard_counts[tile] = opponent_discard_counts.get(tile, 0) + 1

    # A-2a: expose per-tile hand / remaining / opponent-discard counts so the
    # downstream linear/GBDT model can actually see WHICH tile we're holding or
    # throwing, not just aggregate counts.
    #
    # A-2c-3: add per-tile safety flag (1 if tile appears in opponent river ->
    # genbutsu / "現物", i.e. safe to discard). This is redundant with
    # opp_disc_t_* > 0 in theory, but a binary flag is much easier for a GBDT
    # split to pick up as a single rule than a count threshold.
    per_tile_features: Dict[str, float] = {}
    safe_in_hand_count = 0.0
    raw_in_hand_count = 0.0
    opp_max_tile_disc = 0
    for tile_code in PER_TILE_FEATURE_ORDER:
        hand_c = float(hand_counts.get(tile_code, 0))
        remain_c = float(remaining_counts.get(tile_code, 0))
        opp_disc_c = float(opponent_discard_counts.get(tile_code, 0))
        per_tile_features[_feature_key_for_tile("hand_t", tile_code)] = hand_c
        per_tile_features[_feature_key_for_tile("remain_t", tile_code)] = remain_c
        per_tile_features[_feature_key_for_tile("opp_disc_t", tile_code)] = opp_disc_c
        safety_flag = 1.0 if opp_disc_c > 0 else 0.0
        per_tile_features[_feature_key_for_tile("safety_t", tile_code)] = safety_flag
        if hand_c > 0:
            if safety_flag > 0:
                safe_in_hand_count += hand_c
            elif remain_c > 0:
                raw_in_hand_count += hand_c
        if opp_disc_c > opp_max_tile_disc:
            opp_max_tile_disc = int(opp_disc_c)

    # 验收 (spec §4.4): per-tile suji (筋) + kabe (壁) safety signals.
    # 必须与 engine/share/linhai_search_v3.cpp::build_state_features 同步加。
    # 由 tests/python/test_feature_parity.py 兜底名字一致性。
    suji_set: set = set()
    for suit_suffix in ("w", "t"):
        for mid in (4, 5, 6):
            mid_tile = f"{mid}{suit_suffix}"
            if opponent_discard_counts.get(mid_tile, 0) > 0:
                if mid == 4:
                    suji_set.update({f"1{suit_suffix}", f"7{suit_suffix}"})
                elif mid == 5:
                    suji_set.update({f"2{suit_suffix}", f"8{suit_suffix}"})
                elif mid == 6:
                    suji_set.update({f"3{suit_suffix}", f"9{suit_suffix}"})

    # Kabe (壁): visible = 4 - remaining_counts. visible >= 3 marks adjacent
    # tiles as safer (opponent unlikely to be waiting on this neighbour).
    kabe_set: set = set()
    for suit_suffix in ("w", "t"):
        for r in range(1, 10):
            tile = f"{r}{suit_suffix}"
            visible_count = 4 - int(remaining_counts.get(tile, 4))
            if visible_count >= 3:
                if r > 1:
                    kabe_set.add(f"{r-1}{suit_suffix}")
                if r < 9:
                    kabe_set.add(f"{r+1}{suit_suffix}")

    for tile_code in PER_TILE_FEATURE_ORDER:
        per_tile_features[_feature_key_for_tile("safety_suji_t", tile_code)] = (
            1.0 if tile_code in suji_set else 0.0
        )
        per_tile_features[_feature_key_for_tile("safety_kabe_t", tile_code)] = (
            1.0 if tile_code in kabe_set else 0.0
        )

    # A-2c-3: aggregate defensive signals. These let the GBDT's houjuu/betaori
    # heads see opponent-threat at a glance instead of reconstructing it from
    # 25 per-tile counts. Kept cheap to compute.
    opp_honor_disc_count = 0
    opp_terminal_disc_count = 0
    opp_middle_disc_count = 0  # 4/5/6 in w/t suits -- discarding these usually means hon/char-itsu push
    for tile_code, cnt in opponent_discard_counts.items():
        if cnt <= 0:
            continue
        tile = require_tile(tile_code)
        if tile.suit is Suit.HONOR:
            opp_honor_disc_count += cnt
        else:
            rank = tile.rank or 0
            if rank in {1, 9}:
                opp_terminal_disc_count += cnt
            elif rank in {4, 5, 6}:
                opp_middle_disc_count += cnt
    opp_disc_total = sum(opponent_discard_counts.values())
    opp_disc_per_meld = float(opp_disc_total) / float(max(1, opponent_meld_count))

    aggregate = {
        "wall_remaining": float(wall_remaining),
        "from_player": float(from_player),
        "has_target_hai": 1.0 if target_code else 0.0,
        "target_in_hand_count": float(target_in_hand_count),
        "target_rank": float(target_rank),
        "target_is_honor": target_is_honor,
        "target_is_terminal": target_is_terminal,
        "target_is_white": target_is_white,
        "hand_size": float(hand_size),
        "meld_count": float(meld_count),
        "discard_count": float(discard_count),
        "distinct_tile_count": float(distinct_tile_count),
        "pair_count": float(pair_count),
        "triplet_count": float(triplet_count),
        "quad_count": float(quad_count),
        "white_tiles_in_hand": float(hand_counts["white"]),
        "suited_count": float(suited_count),
        "honor_count": float(honor_count),
        "terminal_count": float(terminal_count),
        "simple_count": float(simple_count),
        "wind_count": float(wind_count),
        "dragon_count": float(dragon_count),
        "tree_active": 1.0 if tree_active else 0.0,
        "grab_charge_active": 1.0 if grab_charge_active else 0.0,
        "grab_charge_hits": float(active_player["grab_charge_hits"]),
        "grab_charge_limit": float(active_player["grab_charge_limit"]),
        "contract_target_count": float(contract_target_count),
        "contract_counter": float(active_player["contract_counter"]),
        "opponent_meld_count": float(opponent_meld_count),
        "opponent_discard_count": float(opponent_discard_count),
        # A-2c-3: defensive aggregate signals for houjuu/betaori heads.
        "safe_in_hand_count": safe_in_hand_count,
        "raw_in_hand_count": raw_in_hand_count,
        "opp_honor_disc_count": float(opp_honor_disc_count),
        "opp_terminal_disc_count": float(opp_terminal_disc_count),
        "opp_middle_disc_count": float(opp_middle_disc_count),
        "opp_max_tile_disc": float(opp_max_tile_disc),
        "opp_disc_per_meld": float(opp_disc_per_meld),
        "remaining_total": float(remaining_total),
        "remaining_white": float(remaining_counts["white"]),
        "can_win": 1.0 if can_win else 0.0,
        "available_action_count": float(len(available_actions)),
        "can_pass": 1.0 if "pass" in action_set else 0.0,
        "can_peng": 1.0 if "peng" in action_set else 0.0,
        "can_chi": 1.0 if "chi" in action_set else 0.0,
        "can_gang": 1.0 if "gang" in action_set else 0.0,
        "can_hu": 1.0 if "hu" in action_set else 0.0,
    }

    aggregate.update(per_tile_features)
    return aggregate


def build_canonical_record(payload: Dict, record_index: int) -> Dict:
    active_wind = normalize_wind(payload.get("active_wind"))
    round_wind = normalize_wind(payload.get("round_wind"))
    dealer_wind = normalize_wind(payload.get("dealer_wind"), default=round_wind)
    players_input = payload.get("players", {})
    players = {wind: parse_player(players_input.get(wind, {}), wind) for wind in WIND_ORDER}
    active_player = players[active_wind]

    visible_counts = build_visible_counts(players)
    remaining_counts = build_remaining_counts(visible_counts)

    opponent_discards: List[str] = []
    opponent_meld_count = 0
    opponent_discard_count = 0
    for wind, player in players.items():
        if wind == active_wind:
            continue
        opponent_discards.extend(player["discards"])
        opponent_meld_count += len(player["melds"])
        opponent_discard_count += len(player["discards"])

    contract_mapping = payload.get("contract_mapping", {})
    contract_targets = active_player["contract_targets"] or [
        normalize_wind(value, default="east")
        for value in contract_mapping.get(active_wind, [])
    ]

    can_win = not bool(
        active_player["passed_hu_this_round"]
        or payload.get("passed_hu_this_round", False)
        or payload.get("pass_hu_this_round", False)
    )
    wall_remaining = int(payload.get("wall_remaining", 0))
    available_actions = [str(action).strip().lower() for action in payload.get("available_actions", []) if str(action).strip()]
    task_labels = build_task_labels(payload, available_actions)
    tree_active = bool(payload.get("tree_active", False) or active_player["tree_revealed"])
    grab_charge_active = bool(payload.get("grab_charge_active", False))
    model_features = build_model_features(
        active_player=active_player,
        remaining_counts=remaining_counts,
        available_actions=available_actions,
        can_win=can_win,
        wall_remaining=wall_remaining,
        from_player=int(payload.get("from_player", 0)),
        target_hai=payload.get("target_hai"),
        tree_active=tree_active,
        grab_charge_active=grab_charge_active,
        contract_target_count=len(contract_targets),
        opponent_meld_count=opponent_meld_count,
        opponent_discard_count=opponent_discard_count,
        opponent_discards=opponent_discards,
    )
    canonical = {
        "record_id": payload.get("record_id", f"record-{record_index:06d}"),
        "context_version": "v3",
        "round_wind": round_wind,
        "dealer_wind": dealer_wind,
        "active_wind": active_wind,
        "wall_remaining": wall_remaining,
        "current_player": {
            "wind": active_wind,
            "hand": active_player["hand"],
            "hand_counts": active_player["hand_counts"],
            "melds": active_player["melds"],
            "discard_count": len(active_player["discards"]),
        },
        "player_snapshots": build_player_snapshots(players),
        "opponent_discards": opponent_discards,
        "visible_counts": visible_counts,
        "remaining_counts": remaining_counts,
        "can_win": can_win,
        "target_hai": normalize_code(payload["target_hai"]) if payload.get("target_hai") else None,
        "from_player": int(payload.get("from_player", 0)),
        "available_actions": available_actions,
        "white_tiles_in_hand": active_player["hand_counts"]["white"],
        "tree_active": tree_active,
        "grab_charge_active": grab_charge_active,
        "grab_charge_hits": active_player["grab_charge_hits"],
        "grab_charge_limit": active_player["grab_charge_limit"],
        "contract_target_count": len(contract_targets),
        "contract_counter": active_player["contract_counter"],
        "opponent_meld_count": opponent_meld_count,
        "opponent_discard_count": opponent_discard_count,
        "label": payload.get("label"),
        "task_labels": task_labels,
        "model_features": model_features,
        "source_meta": payload.get("source_meta", {}),
    }
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract canonical linhai states from JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for index, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record = build_canonical_record(payload, written)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({"written": written, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
