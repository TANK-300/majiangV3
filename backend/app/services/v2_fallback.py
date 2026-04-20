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


def _candidate_v2_module_dirs() -> List[Path]:
    env_override = os.environ.get("LINHAI_V2_MODULE_DIR")
    candidates: List[Path] = []
    if env_override:
        candidates.extend(Path(item) for item in env_override.split(os.pathsep) if item.strip())
    candidates.extend(
        [
            REPO_ROOT / "engine",
            REPO_ROOT / "engine" / "build",
            REPO_ROOT / "backend",
            Path("/Users/wf/Documents/wb/linhai-majiang-v2/majiang-linhai-ai/backend"),
        ]
    )
    return _unique_existing_dirs(candidates)


def _candidate_v2_params_dirs() -> List[Path]:
    env_override = os.environ.get("LINHAI_V2_PARAMS_DIR")
    candidates: List[Path] = []
    if env_override:
        candidates.extend(Path(item) for item in env_override.split(os.pathsep) if item.strip())
    candidates.extend(
        [
            REPO_ROOT / "engine" / "params",
            Path("/Users/wf/Documents/wb/linhai-majiang-v2/akochan/params/"),
            Path("/Users/wf/Documents/wb/akochan/params/"),
        ]
    )
    return _unique_existing_dirs(candidates)


class V2FallbackAdapter:
    """直接调用旧版 V2 `.so`，避免导入旧项目的 Python 包。"""

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
        for path in _candidate_v2_module_dirs():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        try:
            self._mod = importlib.import_module("linhai_ai")
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
            self._engine = self._mod.LinHaiEVEngine()
            loaded_params = False
            for path in _candidate_v2_params_dirs():
                if (path / "agari_prob" / "linhai").is_dir():
                    self._engine.load_params(str(path) + os.sep)
                    if self._engine.params_loaded():
                        self._params_dir = path
                        loaded_params = True
                        break
            self._load_reason = "ok" if loaded_params else "params_not_found"
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
    def _display_tile(tile: str) -> str:
        return tile.replace("m", "w").replace("s", "t")

    @staticmethod
    def _wind_to_int(wind: Wind) -> int:
        return {
            Wind.EAST: 0,
            Wind.SOUTH: 1,
            Wind.WEST: 2,
            Wind.NORTH: 3,
        }.get(wind, 0)

    def _build_state(self, state: GameState):
        gs = self._mod.GameState()
        player = state.current_player()
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
        for idx, wind in enumerate(wind_order):
            snapshot = state.players.get(wind)
            jikazes.append(idx)
            discard_counts.append(len(snapshot.discards) if snapshot else 0)
            meld_counts.append(len(snapshot.melds) if snapshot else 0)
            reach_flags.append(False)
        gs.set_player_snapshots(jikazes, discard_counts, meld_counts, reach_flags)

        opponent = next((p for wind, p in state.players.items() if wind != player.wind), None)
        if opponent is not None:
            discards = [self._mod.tile_str_to_int(self._tile_to_mjai(tile)) for tile in opponent.discards]
            gs_discard_ints = [tile for tile in discards if tile > 0]
            if gs_discard_ints:
                self._engine.set_opponent_discards(gs_discard_ints)
        passed_hu = bool(
            getattr(player, "passed_hu_this_round", False)
            or getattr(player, "pass_hu_this_round", False)
            or getattr(state, "passed_hu_this_round", False)
            or getattr(state, "pass_hu_this_round", False)
        )
        self._engine.set_pass_hu(passed_hu)
        return gs

    def recommend_discard(self, state: GameState) -> Optional[Dict]:
        if not self.available():
            return None
        try:
            gs = self._build_state(state)
            recs = self._engine.recommend_discard_ev(gs)
            if not recs:
                return None
            best = recs[0]
            return {
                "engine": "v2",
                "chosen_by": "fallback_v2",
                "action": "discard",
                "tile": self._display_tile(self._mod.tile_int_to_str(best.hai)),
                "candidate_scores": {
                    self._display_tile(self._mod.tile_int_to_str(item.hai)): round(item.ev_total, 3)
                    for item in recs[:5]
                },
                "total_ev": round(best.ev_total, 3),
                "agari_prob": 0.0,
                "houjuu_prob": round(best.risk_houjuu, 6),
                "defense_score": round(best.ev_defense, 3),
                "search_depth": 1,
                "nodes_expanded": len(recs),
                "search_nodes": len(recs),
                "root_candidates_total": len(recs),
                "root_candidates_evaluated": len(recs),
                "configured_time_budget_ms": -1,
                "configured_node_budget": -1,
                "cache_hits": 0,
                "truncated": False,
                "fallback_reason": "v3_unavailable",
                "truncate_reason": "",
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
            gs = self._build_state(state)
            tile_int = self._mod.tile_str_to_int(self._tile_to_mjai(discarded_tile))
            recs = self._engine.recommend_response_v2(gs, tile_int, from_player, action_buttons or ["pass"])
            if not recs:
                return None
            best = recs[0]
            return {
                "engine": "v2",
                "chosen_by": "fallback_v2",
                "action": best.action,
                "tile": self._display_tile(self._mod.tile_int_to_str(best.discard_hai)) if best.discard_hai else None,
                "candidate_scores": {item.action: round(item.score, 3) for item in recs[:5]},
                "total_ev": round(best.score, 3),
                "agari_prob": 0.0,
                "houjuu_prob": 0.0,
                "defense_score": round(-best.risk_value, 3),
                "search_depth": 1,
                "nodes_expanded": len(recs),
                "search_nodes": len(recs),
                "root_candidates_total": len(recs),
                "root_candidates_evaluated": len(recs),
                "configured_time_budget_ms": -1,
                "configured_node_budget": -1,
                "cache_hits": 0,
                "truncated": False,
                "fallback_reason": "v3_unavailable",
                "truncate_reason": "",
            }
        except Exception:
            return None
