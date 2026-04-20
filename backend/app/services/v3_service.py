from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..core.state import GameState, Wind
from ..core.tiles import normalize_code


REPO_ROOT = Path(__file__).resolve().parents[3]


def _unique_existing_dirs(paths: List[Path]) -> List[Path]:
    seen: List[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        if resolved in seen:
            continue
        seen.append(resolved)
    return seen


def _candidate_module_dirs() -> List[Path]:
    env_override = os.environ.get("LINHAI_V3_ENGINE_MODULE_DIR")
    candidates: List[Path] = []
    if env_override:
        candidates.extend(Path(item) for item in env_override.split(os.pathsep) if item.strip())
    candidates.extend(
        [
            REPO_ROOT / "engine",
            REPO_ROOT / "engine" / "build",
            REPO_ROOT / "backend",
            # Legacy macOS dev paths kept only as last resort for back-compat.
            Path("/Users/wf/Documents/wb/linhai-majiang-v3/engine"),
            Path("/Users/wf/Documents/wb/linhai-majiang-v3/backend"),
        ]
    )
    return _unique_existing_dirs(candidates)


def _candidate_params_dirs() -> List[Path]:
    env_override = os.environ.get("LINHAI_V3_PARAMS_DIR")
    candidates: List[Path] = []
    if env_override:
        candidates.extend(Path(item) for item in env_override.split(os.pathsep) if item.strip())
    candidates.extend(
        [
            REPO_ROOT / "engine" / "params",
            Path("/Users/wf/Documents/wb/linhai-majiang-v3/engine/params"),
        ]
    )
    return _unique_existing_dirs(candidates)


class V3SearchService:
    def __init__(self) -> None:
        self._mod = None
        self._engine = None
        self._load_reason: str = "not_attempted"
        self._module_dir: Optional[Path] = None
        self._params_dir: Optional[Path] = None
        self._load()

    def available(self) -> bool:
        return self._engine is not None

    def status(self) -> Dict[str, object]:
        return {
            "available": self.available(),
            "reason": self._load_reason,
            "module_dir": str(self._module_dir) if self._module_dir else None,
            "params_dir": str(self._params_dir) if self._params_dir else None,
        }

    def _load(self) -> None:
        for path in _candidate_module_dirs():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        try:
            self._mod = importlib.import_module("linhai_v3")
        except ImportError as exc:
            self._load_reason = f"import_failed:{exc}"
            self._mod = None
            self._engine = None
            return
        except Exception as exc:
            self._load_reason = f"import_exception:{exc}"
            self._mod = None
            self._engine = None
            return
        self._module_dir = Path(getattr(self._mod, "__file__", "")).resolve().parent if getattr(self._mod, "__file__", None) else None
        try:
            self._engine = self._mod.LinhaiSearchEngineV3()
            cfg = self._mod.SearchConfig()
            cfg.max_self_draw_depth = 1
            cfg.deep_depth_for_near_ready = 2
            cfg.beam_width_after_draw = 2
            cfg.beam_width_far_shanten = 1
            self._engine.set_search_config(cfg)
            params_dirs = _candidate_params_dirs()
            loaded_params = False
            for params_dir in params_dirs:
                if (params_dir / "agari_prob" / "linhai").is_dir() or (params_dir / "v3").is_dir():
                    self._engine.load_model_bundle(str(params_dir))
                    self._params_dir = params_dir
                    loaded_params = True
                    break
            if loaded_params:
                self._load_reason = "ok"
            else:
                self._load_reason = "params_not_found"
        except Exception as exc:
            self._load_reason = f"engine_init_failed:{exc}"
            self._mod = None
            self._engine = None

    @staticmethod
    def _tile_to_mjai(tile: str) -> str:
        t = normalize_code(tile)
        if t == "white":
            return "P"
        if t == "red":
            return "C"
        if t == "green":
            return "F"
        if t == "east":
            return "E"
        if t == "south":
            return "S"
        if t == "west":
            return "W"
        if t == "north":
            return "N"
        return t.replace("w", "m").replace("t", "s")

    @staticmethod
    def _wind_to_int(wind: Wind) -> int:
        return {
            Wind.EAST: 0,
            Wind.SOUTH: 1,
            Wind.WEST: 2,
            Wind.NORTH: 3,
        }.get(wind, 0)

    @staticmethod
    def _display_tile(tile: str) -> str:
        return tile.replace("m", "w").replace("s", "t")

    @staticmethod
    def _set_optional_attr(target: object, attr: str, value: object) -> None:
        if hasattr(target, attr):
            setattr(target, attr, value)

    def _build_canonical_state(
        self,
        state: GameState,
        discarded_tile: Optional[str] = None,
        from_player: int = 0,
        action_buttons: Optional[List[str]] = None,
    ):
        canonical = self._mod.CanonicalGameState()
        player = state.current_player()
        gs = canonical.game_state
        gs.set_hand([self._tile_to_mjai(tile) for tile in player.hand])
        gs.current_player = self._wind_to_int(player.wind)
        gs.jikaze = self._wind_to_int(player.wind)
        gs.wall_remaining = state.wall_remaining
        gs.has_kan = any(getattr(meld, "type", "") in {"ming_gang", "an_gang", "bu_gang"} for meld in player.melds)

        if player.melds:
            meld_types = []
            meld_tiles = []
            for meld in player.melds:
                meld_types.append(str(meld.type))
                meld_tiles.append([self._tile_to_mjai(tile) for tile in meld.tiles])
            gs.set_current_melds(meld_types, meld_tiles)

        wind_order = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
        jikazes = []
        discard_counts = []
        meld_counts = []
        reach_flags = []
        opponent_discards: List[int] = []
        opponent_meld_count = 0
        opponent_discard_count = 0
        for idx, wind in enumerate(wind_order):
            snapshot = state.players.get(wind)
            jikazes.append(idx)
            discard_counts.append(len(snapshot.discards) if snapshot else 0)
            meld_counts.append(len(snapshot.melds) if snapshot else 0)
            reach_flags.append(False)
            if snapshot and wind != player.wind:
                opponent_meld_count += len(snapshot.melds)
                opponent_discard_count += len(snapshot.discards)
                for tile in snapshot.discards:
                    tile_int = self._mod.tile_str_to_int(self._tile_to_mjai(tile))
                    if tile_int > 0:
                        opponent_discards.append(tile_int)
        gs.set_player_snapshots(jikazes, discard_counts, meld_counts, reach_flags)

        canonical.opponent_discards = opponent_discards
        self._set_optional_attr(canonical, "white_tiles_in_hand", sum(1 for tile in player.hand if normalize_code(tile) == "white"))
        self._set_optional_attr(canonical, "tree_active", bool(state.tree_active or player.tree_revealed))
        self._set_optional_attr(canonical, "grab_charge_active", bool(state.grab_charge_active))
        self._set_optional_attr(canonical, "grab_charge_hits", int(getattr(player, "grab_charge_hits", 0)))
        self._set_optional_attr(canonical, "grab_charge_limit", int(getattr(player, "grab_charge_limit", 0)))
        contract_targets = list(getattr(player, "contract_targets", [])) or list(state.contract_mapping.get(player.wind, []))
        self._set_optional_attr(canonical, "contract_target_count", len(contract_targets))
        self._set_optional_attr(canonical, "contract_counter", int(getattr(player, "contract_counter", 0)))
        self._set_optional_attr(canonical, "opponent_meld_count", opponent_meld_count)
        self._set_optional_attr(canonical, "opponent_discard_count", opponent_discard_count)
        canonical.can_win = not bool(
            getattr(player, "passed_hu_this_round", False)
            or getattr(player, "pass_hu_this_round", False)
            or getattr(state, "passed_hu_this_round", False)
            or getattr(state, "pass_hu_this_round", False)
        )
        canonical.target_hai = self._mod.tile_str_to_int(self._tile_to_mjai(discarded_tile)) if discarded_tile else 0
        canonical.from_player = from_player
        canonical.available_actions = action_buttons or []
        canonical.refresh_counts()
        return canonical

    def recommend_discard(self, state: GameState) -> Optional[Dict]:
        if not self.available():
            return None
        try:
            canonical = self._build_canonical_state(state)
            result = self._engine.recommend_discard_v3(canonical)
            tile = self._display_tile(self._mod.tile_int_to_str(result.hai)) if result.hai else None
            return {
                "engine": "v3",
                "chosen_by": result.chosen_by,
                "action": result.action,
                "tile": tile,
                "candidate_scores": {
                    self._display_tile(self._mod.tile_int_to_str(item.hai)): round(item.total_ev, 3)
                    for item in result.candidate_scores[:5]
                    if item.hai
                },
                "total_ev": round(result.total_ev, 3),
                "agari_prob": round(result.agari_prob, 6),
                "tenpai_prob": round(result.tenpai_prob, 6),
                "houjuu_prob": round(result.houjuu_prob, 6),
                "betaori_prob": round(result.betaori_prob, 6),
                "tsumo_num": round(result.tsumo_num, 6),
                "ryukyoku_prob": round(result.ryukyoku_prob, 6),
                "defense_score": round(result.defense_score, 3),
                "search_depth": result.search_depth,
                "nodes_expanded": result.nodes_expanded,
                "search_nodes": result.search_nodes,
                "root_candidates_total": result.root_candidates_total,
                "root_candidates_evaluated": result.root_candidates_evaluated,
                "configured_time_budget_ms": result.configured_time_budget_ms,
                "configured_node_budget": result.configured_node_budget,
                "cache_hits": result.cache_hits,
                "truncated": result.truncated,
                "fallback_reason": result.fallback_reason,
                "truncate_reason": result.truncate_reason,
            }
        except Exception:
            return None

    def recommend_response(
        self,
        state: GameState,
        discarded_tile: str,
        from_player: int,
        action_buttons: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        if not self.available():
            return None
        try:
            canonical = self._build_canonical_state(state, discarded_tile, from_player, action_buttons)
            result = self._engine.recommend_response_v3(canonical)
            tile = self._display_tile(self._mod.tile_int_to_str(result.hai)) if result.hai else None
            candidate_scores = {}
            for item in result.candidate_scores[:5]:
                discard_tile = self._display_tile(self._mod.tile_int_to_str(item.hai)) if item.hai else None
                key = item.action if not discard_tile else f"{item.action}->{discard_tile}"
                candidate_scores[key] = round(item.total_ev, 3)
            return {
                "engine": "v3",
                "chosen_by": result.chosen_by,
                "action": result.action,
                "tile": tile,
                "candidate_scores": candidate_scores,
                "total_ev": round(result.total_ev, 3),
                "agari_prob": round(result.agari_prob, 6),
                "tenpai_prob": round(result.tenpai_prob, 6),
                "houjuu_prob": round(result.houjuu_prob, 6),
                "betaori_prob": round(result.betaori_prob, 6),
                "tsumo_num": round(result.tsumo_num, 6),
                "ryukyoku_prob": round(result.ryukyoku_prob, 6),
                "defense_score": round(result.defense_score, 3),
                "search_depth": result.search_depth,
                "nodes_expanded": result.nodes_expanded,
                "search_nodes": result.search_nodes,
                "root_candidates_total": result.root_candidates_total,
                "root_candidates_evaluated": result.root_candidates_evaluated,
                "configured_time_budget_ms": result.configured_time_budget_ms,
                "configured_node_budget": result.configured_node_budget,
                "cache_hits": result.cache_hits,
                "truncated": result.truncated,
                "fallback_reason": result.fallback_reason,
                "truncate_reason": result.truncate_reason,
            }
        except Exception:
            return None
