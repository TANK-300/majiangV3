#!/usr/bin/env python3
"""Self-play based win-rate regression harness for Linhai Mahjong V3.

The harness wires two arbitrary policies (engines) against each other in a
simplified Linhai 2-player match and reports win rates, tsumo/ron ratios,
houjuu ratio and average game length. The goal is NOT to faithfully reproduce
every tile-by-tile rule of the production server, but to provide an
*apples-to-apples* comparison between two engines so we can tell whether a
code/model change actually improves strength.

Rules used here (intentionally kept simple and symmetric):

* 2 players, seats east vs south, east starts.
* Tile set: 9 manzu (``1w..9w``), 9 tiaozu (``1t..9t``),
  4 winds, 3 dragons + ``white`` acting as universal tile (4 copies each).
* Each player starts with 13 tiles; on their turn they draw one then discard
  one (picked by their policy). No chi/peng/gang (those require response
  logic which is out of scope for A-0).
* Win patterns supported: 4 melds + 1 pair, with up to ``white_tiles_in_hand``
  wildcards substituted for any missing tile. Seven-pairs is supported iff
  the player has >= 1 white tile OR we match it exactly.
* Ron: on discard, opponent may declare ron if the discarded tile completes
  their hand. Policies can opt-out of a ron by returning ``pass_hu`` (which
  also marks them passed_hu_this_round, matching the production "over hu"
  rule).
* Tsumo: on draw, the drawer can declare self-tsumo if complete.
* If the wall runs out before either player wins, the match is a draw.

The deliberately simplified rules keep both engines on the same footing; the
numbers we get out are meaningful *relatively* even if they are not an exact
reflection of production win-rates.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.state import GameState, Meld, PlayerState, Wind
from backend.app.core.tiles import Honor, Suit, all_tile_codes, normalize_code, require_tile


TILE_CODES: List[str] = [code for code in all_tile_codes()]
TILE_INDEX: Dict[str, int] = {code: idx for idx, code in enumerate(TILE_CODES)}
WHITE = "white"


# --------------------------------------------------------------------------- #
# Win detection (4 melds + 1 pair, with white acting as wildcard).            #
# --------------------------------------------------------------------------- #


def _hand_counts(hand: Sequence[str]) -> List[int]:
    counts = [0] * len(TILE_CODES)
    for tile in hand:
        counts[TILE_INDEX[tile]] += 1
    return counts


def _is_consecutive_suit(a: int, b: int, c: int) -> bool:
    ca = TILE_CODES[a]
    cb = TILE_CODES[b]
    cc = TILE_CODES[c]
    if any(code in {WHITE, "east", "south", "west", "north", "red", "green"} for code in (ca, cb, cc)):
        return False
    # Same suit and consecutive rank
    suit_a = ca[-1]
    suit_b = cb[-1]
    suit_c = cc[-1]
    if suit_a != suit_b or suit_b != suit_c:
        return False
    try:
        ra = int(ca[:-1])
        rb = int(cb[:-1])
        rc = int(cc[:-1])
    except ValueError:
        return False
    return rb == ra + 1 and rc == rb + 1


def _try_remove_melds(counts: List[int], melds_needed: int, wildcards: int) -> bool:
    """Return True if we can extract ``melds_needed`` melds using up to
    ``wildcards`` white tiles as wildcards (white tiles are modelled separately
    and NOT part of ``counts``)."""
    if melds_needed == 0:
        return True
    # Find first non-zero tile
    first = -1
    for idx in range(len(counts)):
        if counts[idx] > 0:
            first = idx
            break
    if first == -1:
        # All that's left is wildcards; each meld costs 3 wildcards.
        return wildcards >= 3 * melds_needed

    # Option 1: triplet of `first`
    if counts[first] >= 3:
        counts[first] -= 3
        if _try_remove_melds(counts, melds_needed - 1, wildcards):
            counts[first] += 3
            return True
        counts[first] += 3

    # Option 2: triplet with wildcards (pair + wildcard, or single + 2 wildcards)
    for wc_use in (1, 2):
        real_use = 3 - wc_use
        if counts[first] >= real_use and wildcards >= wc_use:
            counts[first] -= real_use
            if _try_remove_melds(counts, melds_needed - 1, wildcards - wc_use):
                counts[first] += real_use
                return True
            counts[first] += real_use

    # Option 3: sequential run starting at `first` (only for suited tiles)
    if TILE_CODES[first] not in {WHITE, "east", "south", "west", "north", "red", "green"}:
        try:
            rank = int(TILE_CODES[first][:-1])
            suit = TILE_CODES[first][-1]
        except ValueError:
            rank = 0
            suit = ""
        if 1 <= rank <= 7:
            second_code = f"{rank + 1}{suit}"
            third_code = f"{rank + 2}{suit}"
            if second_code in TILE_INDEX and third_code in TILE_INDEX:
                b = TILE_INDEX[second_code]
                c = TILE_INDEX[third_code]
                # Sub-options based on how many of (a, b, c) we take from real tiles vs wildcards
                for b_use in (0, 1):
                    for c_use in (0, 1):
                        wc_use = b_use + c_use
                        real_b = 1 - b_use
                        real_c = 1 - c_use
                        if wildcards < wc_use:
                            continue
                        if counts[first] < 1:
                            continue
                        if counts[b] < real_b:
                            continue
                        if counts[c] < real_c:
                            continue
                        counts[first] -= 1
                        counts[b] -= real_b
                        counts[c] -= real_c
                        if _try_remove_melds(counts, melds_needed - 1, wildcards - wc_use):
                            counts[first] += 1
                            counts[b] += real_b
                            counts[c] += real_c
                            return True
                        counts[first] += 1
                        counts[b] += real_b
                        counts[c] += real_c

    return False


def is_winning_hand(hand: Sequence[str]) -> bool:
    """Check whether ``hand`` (14 tiles) forms a standard 4-melds + 1-pair
    hand, treating ``white`` as a wildcard."""
    if len(hand) != 14:
        return False
    counts = _hand_counts(hand)
    wildcards = counts[TILE_INDEX[WHITE]]
    counts[TILE_INDEX[WHITE]] = 0

    # Enumerate every possible pair:
    #   * a real pair from `counts` (cost 0 wildcards)
    #   * a pair that is real+wildcard (cost 1 wildcard)
    #   * a pair of two wildcards (cost 2 wildcards)
    for pair_idx in range(len(TILE_CODES)):
        if pair_idx == TILE_INDEX[WHITE]:
            continue
        # Real pair
        if counts[pair_idx] >= 2:
            counts[pair_idx] -= 2
            if _try_remove_melds(counts, 4, wildcards):
                counts[pair_idx] += 2
                return True
            counts[pair_idx] += 2
        # Real + wildcard
        if counts[pair_idx] >= 1 and wildcards >= 1:
            counts[pair_idx] -= 1
            if _try_remove_melds(counts, 4, wildcards - 1):
                counts[pair_idx] += 1
                return True
            counts[pair_idx] += 1
    # Pair of two wildcards
    if wildcards >= 2:
        if _try_remove_melds(counts, 4, wildcards - 2):
            return True
    return False


# --------------------------------------------------------------------------- #
# Policies                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class DiscardDecision:
    tile: str
    engine: str
    reason: str = ""


class Policy:
    name: str = "policy"

    def choose_discard(self, game: "SimulatedGame", seat_idx: int) -> DiscardDecision:
        raise NotImplementedError

    def choose_ron(self, game: "SimulatedGame", seat_idx: int, discarded_tile: str) -> bool:
        """Default: always take a ron when offered."""
        return True


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose_discard(self, game: "SimulatedGame", seat_idx: int) -> DiscardDecision:
        hand = list(game.seats[seat_idx].hand)
        tile = self._rng.choice(hand)
        return DiscardDecision(tile=tile, engine="random")


class HeuristicPolicy(Policy):
    name = "heuristic"

    def choose_discard(self, game: "SimulatedGame", seat_idx: int) -> DiscardDecision:
        seat = game.seats[seat_idx]
        counts = Counter(seat.hand)

        def score(tile: str) -> float:
            descriptor = require_tile(tile)
            base = 0.0
            if tile == WHITE:
                return -9999.0  # never throw the universal tile
            if counts[tile] == 1:
                base += 3.0
            elif counts[tile] >= 3:
                base -= 2.0
            if descriptor.is_honor:
                base += 2.0
            elif descriptor.rank in (1, 9):
                base += 1.5
            elif descriptor.rank is not None and 3 <= descriptor.rank <= 7:
                base -= 1.0
            return base

        best_tile = max(seat.hand, key=score)
        return DiscardDecision(tile=best_tile, engine="heuristic")


class OrchestratorPolicy(Policy):
    """Route discards through the production orchestrator (V3 -> V2 -> heuristic)."""

    def __init__(self, name: Optional[str] = None) -> None:
        from backend.app.services.orchestrator import LinhaiV3Orchestrator

        self._orchestrator = LinhaiV3Orchestrator()
        status = self._orchestrator.engine_status()
        self._active = status.get("active_engine", "heuristic")
        self.name = name or f"orchestrator[{self._active}]"

    def choose_discard(self, game: "SimulatedGame", seat_idx: int) -> DiscardDecision:
        state = game.to_core_state(seat_idx)
        result = self._orchestrator.recommend(state)
        tile = result.get("tile")
        if not tile or tile not in game.seats[seat_idx].hand:
            # Fallback to heuristic if the engine returned something unusable
            tile = HeuristicPolicy().choose_discard(game, seat_idx).tile
            return DiscardDecision(tile=tile, engine="heuristic", reason="engine_returned_unusable_tile")
        return DiscardDecision(
            tile=normalize_code(tile),
            engine=str(result.get("engine", self._active)),
            reason=str(result.get("chosen_by", "")),
        )


class NeuralPolicy(Policy):
    """B-2a: ONNX-backed policy that scores every candidate discard with an
    MLP trained on self-play samples.

    Score per candidate (larger is better):
        score = agari_prob - beta * houjuu_prob - alpha * tsumo_num

    The ONNX bundle must contain model.onnx plus model_meta.json (both are
    produced by tools/train_neural.py).

    Call with spec: ``neural:/path/to/bundle_dir`` or provide via env
    ``LINHAI_NEURAL_BUNDLE``.
    """

    def __init__(
        self,
        bundle_dir: Path,
        *,
        alpha: float = 0.02,
        beta: float = 2.5,
        name: Optional[str] = None,
    ) -> None:
        import numpy as np
        import onnxruntime as ort

        bundle_dir = Path(bundle_dir)
        meta_path = bundle_dir / "model_meta.json"
        onnx_path = bundle_dir / "model.onnx"
        if not meta_path.is_file() or not onnx_path.is_file():
            raise FileNotFoundError(f"Neural bundle missing at {bundle_dir} (need model.onnx + model_meta.json)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._feature_names: List[str] = list(meta["feature_names"])
        self._feature_means: List[float] = [float(meta["feature_means"][name]) for name in self._feature_names]
        self._feature_stds: List[float] = [float(meta["feature_stds"][name]) for name in self._feature_names]
        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._np = np
        self.name = name or f"neural[{bundle_dir.name}]"

    def _score_features(self, feature_dict: Dict[str, float]) -> Tuple[float, float, float]:
        np = self._np
        x = np.zeros((1, len(self._feature_names)), dtype=np.float32)
        for i, name in enumerate(self._feature_names):
            raw = float(feature_dict.get(name, 0.0))
            std = self._feature_stds[i] if self._feature_stds[i] > 1e-9 else 1.0
            x[0, i] = (raw - self._feature_means[i]) / std
        logits = self._session.run(None, {"features": x})[0][0]

        def _sig(z: float) -> float:
            if z >= 0:
                e = math.exp(-z)
                return 1.0 / (1.0 + e)
            e = math.exp(z)
            return e / (1.0 + e)

        agari = _sig(float(logits[0]))
        houjuu = _sig(float(logits[1]))
        tsumo_num = float(logits[2])
        return agari, houjuu, tsumo_num

    def choose_discard(self, game: "SimulatedGame", seat_idx: int) -> DiscardDecision:
        from tools.extract_canonical_states import build_model_features  # local import

        seat = game.seats[seat_idx]
        opp = game.seats[1 - seat_idx]
        remaining = {code: 4 for code in (game.__class__.__module__,)}  # placeholder, overwritten below
        # Build remaining counts from scratch
        from tools.extract_canonical_states import ALL_TILE_CODES

        remaining = {code: 4 for code in ALL_TILE_CODES}
        for tile in seat.hand:
            remaining[tile] = max(0, remaining[tile] - 1)
        for tile in opp.hand:
            remaining[tile] = max(0, remaining[tile] - 1)
        for tile in seat.discards:
            remaining[tile] = max(0, remaining[tile] - 1)
        for tile in opp.discards:
            remaining[tile] = max(0, remaining[tile] - 1)

        best_tile: Optional[str] = None
        best_score = -1e30
        candidate_tiles = list(set(seat.hand))
        for candidate in candidate_tiles:
            hypothetical = list(seat.hand)
            hypothetical.remove(candidate)
            active_player = {
                "wind": seat.wind.value,
                "hand": hypothetical,
                "hand_counts": {code: hypothetical.count(code) for code in set(hypothetical)},
                "melds": [],
                "discards": list(seat.discards) + [candidate],
                "tree_revealed": False,
                "grab_charge_hits": 0,
                "grab_charge_limit": 0,
                "contract_targets": [],
                "contract_counter": 0,
                "passed_hu_this_round": seat.passed_hu_this_round,
            }
            try:
                features = build_model_features(
                    active_player=active_player,
                    remaining_counts=remaining,
                    available_actions=["discard"],
                    can_win=not seat.passed_hu_this_round,
                    wall_remaining=len(game.wall),
                    from_player=1 - seat_idx,
                    target_hai=None,
                    tree_active=False,
                    grab_charge_active=False,
                    contract_target_count=0,
                    opponent_meld_count=0,
                    opponent_discard_count=len(opp.discards),
                )
            except Exception:
                continue
            try:
                agari, houjuu, tsumo = self._score_features(features)
            except Exception:
                continue
            if candidate == "white":
                # never throw the universal tile away without a strong reason
                score = agari - self._beta * houjuu - self._alpha * tsumo - 5.0
            else:
                score = agari - self._beta * houjuu - self._alpha * tsumo
            if score > best_score:
                best_score = score
                best_tile = candidate

        if best_tile is None or best_tile not in seat.hand:
            # Degrade to heuristic if the network couldn't score anything
            return HeuristicPolicy().choose_discard(game, seat_idx)

        return DiscardDecision(tile=best_tile, engine="neural", reason=f"score={best_score:.3f}")


def policy_factory(spec: str, rng: random.Random) -> Policy:
    """Build a Policy from a short spec string.

    Supported forms:
        ``heuristic`` / ``rule`` / ``baseline``
        ``random``
        ``orchestrator`` / ``orchestrator:label``
        ``neural:/abs/path/to/bundle_dir``
        ``neural`` (reads LINHAI_NEURAL_BUNDLE from env)
    """
    raw = spec.strip()
    lower = raw.lower()
    if lower in {"heuristic", "rule", "baseline"}:
        return HeuristicPolicy()
    if lower == "random":
        return RandomPolicy(rng)
    if lower.startswith("orchestrator"):
        _, _, label = raw.partition(":")
        return OrchestratorPolicy(name=label.strip() or None)
    if lower.startswith("neural"):
        _, _, bundle_spec = raw.partition(":")
        bundle_spec = bundle_spec.strip()
        if not bundle_spec:
            bundle_spec = os.environ.get("LINHAI_NEURAL_BUNDLE", "")
        if not bundle_spec:
            raise ValueError(
                "neural policy requires a bundle directory (spec 'neural:/path/to/bundle' or "
                "LINHAI_NEURAL_BUNDLE env var)"
            )
        return NeuralPolicy(Path(bundle_spec))
    raise ValueError(f"Unknown policy spec: {spec}")


# --------------------------------------------------------------------------- #
# Simulation                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class Seat:
    wind: Wind
    hand: List[str] = field(default_factory=list)
    discards: List[str] = field(default_factory=list)
    passed_hu_this_round: bool = False


@dataclass
class GameResult:
    winner: Optional[int]
    loser: Optional[int]
    kind: str  # "tsumo", "ron", "draw"
    turns: int
    houjuu_seat: Optional[int] = None
    steps: int = 0


def _build_wall(rng: random.Random) -> List[str]:
    wall: List[str] = []
    for code in TILE_CODES:
        wall.extend([code] * 4)
    rng.shuffle(wall)
    return wall


class SimulatedGame:
    def __init__(self, policies: Sequence[Policy], rng: random.Random, max_turns: int = 200):
        assert len(policies) == 2
        self.policies = list(policies)
        self.rng = rng
        self.max_turns = max_turns
        self.seats: List[Seat] = [Seat(Wind.EAST), Seat(Wind.SOUTH)]
        self.wall: List[str] = []
        self.current_seat: int = 0
        self.result: Optional[GameResult] = None

    # -- setup ----------------------------------------------------------- #

    def deal(self) -> None:
        self.wall = _build_wall(self.rng)
        for seat in self.seats:
            seat.hand = sorted(self.wall.pop() for _ in range(13))
            seat.discards = []
            seat.passed_hu_this_round = False
        self.current_seat = 0

    # -- state bridging ------------------------------------------------- #

    def to_core_state(self, seat_idx: int) -> GameState:
        """Produce a backend.app.core.state.GameState for the orchestrator."""
        players: Dict[Wind, PlayerState] = {}
        for idx, seat in enumerate(self.seats):
            players[seat.wind] = PlayerState(
                wind=seat.wind,
                hand=list(seat.hand) if idx == seat_idx else [],
                discards=list(seat.discards),
            )
        return GameState(
            round_wind=Wind.EAST,
            dealer_wind=Wind.EAST,
            active_wind=self.seats[seat_idx].wind,
            players=players,
            wall_remaining=len(self.wall),
        )

    # -- per-turn logic -------------------------------------------------- #

    def _opponent_can_ron(self, seat_idx: int, tile: str) -> bool:
        opp = self.seats[1 - seat_idx]
        if opp.passed_hu_this_round:
            return False
        return is_winning_hand(list(opp.hand) + [tile])

    def play_one_turn(self) -> Optional[GameResult]:
        seat_idx = self.current_seat
        seat = self.seats[seat_idx]
        if not self.wall:
            return GameResult(winner=None, loser=None, kind="draw", turns=0)
        tile = self.wall.pop()
        seat.hand.append(tile)

        # Tsumo check
        if is_winning_hand(seat.hand):
            return GameResult(winner=seat_idx, loser=None, kind="tsumo", turns=0)

        decision = self.policies[seat_idx].choose_discard(self, seat_idx)
        if decision.tile not in seat.hand:
            # Shouldn't happen but guard against buggy policies
            decision = DiscardDecision(tile=seat.hand[-1], engine="emergency_fallback")
        seat.hand.remove(decision.tile)
        seat.hand.sort()
        seat.discards.append(decision.tile)

        # Ron check against opponent
        if self._opponent_can_ron(seat_idx, decision.tile):
            policy_opp = self.policies[1 - seat_idx]
            take_ron = True
            try:
                take_ron = policy_opp.choose_ron(self, 1 - seat_idx, decision.tile)
            except Exception:
                take_ron = True
            if take_ron:
                return GameResult(
                    winner=1 - seat_idx,
                    loser=seat_idx,
                    kind="ron",
                    turns=0,
                    houjuu_seat=seat_idx,
                )
            self.seats[1 - seat_idx].passed_hu_this_round = True

        # After the drawer finishes their turn, clear their own passed_hu state
        # (they had the option to tsumo, if they reach the next turn the flag
        # should reset).
        self.seats[seat_idx].passed_hu_this_round = False
        self.current_seat = 1 - seat_idx
        return None

    def run(self) -> GameResult:
        self.deal()
        for turn in range(self.max_turns):
            result = self.play_one_turn()
            if result is not None:
                result.turns = turn + 1
                result.steps = turn + 1
                self.result = result
                return result
        self.result = GameResult(winner=None, loser=None, kind="draw", turns=self.max_turns)
        return self.result


# --------------------------------------------------------------------------- #
# Match runner                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class MatchStats:
    games_played: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    tsumo_a: int = 0
    tsumo_b: int = 0
    ron_a: int = 0
    ron_b: int = 0
    houjuu_a: int = 0
    houjuu_b: int = 0
    turns: List[int] = field(default_factory=list)

    def record(self, result: GameResult, swap: bool) -> None:
        self.games_played += 1
        self.turns.append(result.turns)
        if result.kind == "draw" or result.winner is None:
            self.draws += 1
            return
        winner = result.winner
        loser = result.loser
        houjuu = result.houjuu_seat
        if swap:
            winner = 1 - winner
            if loser is not None:
                loser = 1 - loser
            if houjuu is not None:
                houjuu = 1 - houjuu
        if winner == 0:
            self.wins_a += 1
            if result.kind == "tsumo":
                self.tsumo_a += 1
            else:
                self.ron_a += 1
        else:
            self.wins_b += 1
            if result.kind == "tsumo":
                self.tsumo_b += 1
            else:
                self.ron_b += 1
        if houjuu == 0:
            self.houjuu_a += 1
        elif houjuu == 1:
            self.houjuu_b += 1

    def as_dict(self, policy_a_name: str, policy_b_name: str) -> Dict[str, object]:
        n = max(self.games_played, 1)
        return {
            "policy_a": policy_a_name,
            "policy_b": policy_b_name,
            "games_played": self.games_played,
            "draws": self.draws,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "winrate_a": self.wins_a / n,
            "winrate_b": self.wins_b / n,
            "draw_rate": self.draws / n,
            "tsumo_a": self.tsumo_a,
            "tsumo_b": self.tsumo_b,
            "ron_a": self.ron_a,
            "ron_b": self.ron_b,
            "houjuu_rate_a": self.houjuu_a / n,
            "houjuu_rate_b": self.houjuu_b / n,
            "avg_turns": statistics.fmean(self.turns) if self.turns else 0.0,
            "median_turns": statistics.median(self.turns) if self.turns else 0.0,
        }


def run_match(
    policy_a: Policy,
    policy_b: Policy,
    games: int,
    *,
    seed: int = 0,
    swap_sides: bool = True,
    max_turns: int = 200,
    progress: Optional[Callable[[int, int], None]] = None,
) -> MatchStats:
    stats = MatchStats()
    for i in range(games):
        rng = random.Random(seed + i)
        swap = swap_sides and (i % 2 == 1)
        if swap:
            policies: List[Policy] = [policy_b, policy_a]
        else:
            policies = [policy_a, policy_b]
        game = SimulatedGame(policies, rng, max_turns=max_turns)
        result = game.run()
        stats.record(result, swap)
        if progress is not None:
            progress(i + 1, games)
    return stats


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Linhai V3 self-play win-rate regressions.")
    parser.add_argument("--policy-a", default="heuristic", help="Spec for policy A (heuristic|random|orchestrator[:label])")
    parser.add_argument("--policy-b", default="heuristic", help="Spec for policy B (heuristic|random|orchestrator[:label])")
    parser.add_argument("--games", type=int, default=200, help="Number of games to play")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (sequential per game)")
    parser.add_argument("--max-turns", type=int, default=200, help="Max turns per game before declaring a draw")
    parser.add_argument("--no-swap", action="store_true", help="Disable side-swapping between games")
    parser.add_argument("--output", help="Optional JSONL file to append the aggregated result to")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policy_a = policy_factory(args.policy_a, rng)
    policy_b = policy_factory(args.policy_b, rng)

    start = time.time()
    stats = run_match(
        policy_a,
        policy_b,
        args.games,
        seed=args.seed,
        swap_sides=not args.no_swap,
        max_turns=args.max_turns,
    )
    elapsed = time.time() - start
    summary = stats.as_dict(policy_a.name, policy_b.name)
    summary["elapsed_seconds"] = round(elapsed, 3)
    summary["seed"] = args.seed
    summary["max_turns"] = args.max_turns

    # Also expose what engine the orchestrator is actually running (critical for ops)
    if isinstance(policy_a, OrchestratorPolicy) or isinstance(policy_b, OrchestratorPolicy):
        from backend.app.services.orchestrator import LinhaiV3Orchestrator
        summary["orchestrator_status"] = LinhaiV3Orchestrator().engine_status()

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
