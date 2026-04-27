#!/usr/bin/env python3
"""B-1: Self-play sample recorder.

Drives the same match setup as ``tools/selfplay_eval.py`` but writes every
``(state, action, outcome)`` triple to a JSONL file so B-2 (value/policy
network) has something to train on.

Output format (one JSONL record per decision point):

    {
      "record_id": "game-000123-step-017-east",
      "game_index": 123,
      "step_index": 17,
      "seat_wind": "east",
      "seat_index": 0,
      "turn": 17,
      "wall_remaining": 62,
      "policy": "heuristic",
      "engine": "heuristic",
      "action": "discard",
      "discard_tile": "7w",
      "model_features": { ... per-tile aggregate feature dict ... },
      "outcome": {
        "result": "win" | "loss" | "draw",
        "kind": "tsumo" | "ron" | "draw",
        "steps_to_end": 7,
        "winner_seat_index": 0,
        "loser_seat_index": 1,
        "houjuu_seat_index": 1
      },
      "task_labels": {
        "can_win_label": true,
        "houjuu_label": false,
        "tsumo_num_label": 7
      }
    }

The feature keys match the names used by
``tools/extract_canonical_states.build_model_features`` and
``engine/share/linhai_search_v3.cpp build_state_features`` (where the
Python side of that invariant lives in ``tools/extract_canonical_states``).
This is intentional: B-1's JSONL can be fed straight into
``tools/train_model_stub.py`` with --dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.state import Wind  # noqa: E402
from tools.extract_canonical_states import build_model_features  # noqa: E402
from tools.selfplay_eval import (  # noqa: E402
    DiscardDecision,
    GameResult,
    HeuristicPolicy,
    Policy,
    SimulatedGame,
    is_winning_hand,
    policy_factory,
)


# --------------------------------------------------------------------------- #
# Sample recorder                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class SampleStep:
    game_index: int
    step_index: int
    seat_index: int
    seat_wind: str
    turn: int
    wall_remaining: int
    policy: str
    engine: str
    action: str
    discard_tile: Optional[str]
    hand_before: List[str]
    model_features: Dict[str, float]


@dataclass
class RecordedGame:
    game_index: int
    steps: List[SampleStep] = field(default_factory=list)
    result: Optional[GameResult] = None


def _compute_hand_counts(hand: Sequence[str]) -> Dict[str, int]:
    from tools.extract_canonical_states import ALL_TILE_CODES

    counts = {code: 0 for code in ALL_TILE_CODES}
    for tile in hand:
        counts[tile] = counts.get(tile, 0) + 1
    return counts


def _compute_remaining_counts(seats_hands: Sequence[Sequence[str]], discards: Sequence[str]) -> Dict[str, int]:
    from tools.extract_canonical_states import ALL_TILE_CODES

    remaining = {code: 4 for code in ALL_TILE_CODES}
    for hand in seats_hands:
        for tile in hand:
            remaining[tile] = max(0, remaining.get(tile, 0) - 1)
    for tile in discards:
        remaining[tile] = max(0, remaining.get(tile, 0) - 1)
    return remaining


def _seat_wind_name(seat_index: int) -> str:
    return Wind.EAST.value if seat_index == 0 else Wind.SOUTH.value


class SamplingGame(SimulatedGame):
    """A SimulatedGame subclass that records every policy decision so the
    outer driver can later annotate it with the final game outcome."""

    def __init__(self, *args: Any, game_index: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.game_index = game_index
        self.steps: List[SampleStep] = []

    def _record_decision(self, seat_idx: int, turn: int, decision: DiscardDecision) -> None:
        seat = self.seats[seat_idx]
        opp = self.seats[1 - seat_idx]

        # Features are computed on the seat's hand *after* they've drawn but
        # *before* they discard (that's the decision state the policy saw).
        active_player = {
            "wind": seat.wind.value,
            "hand": list(seat.hand),
            "hand_counts": _compute_hand_counts(seat.hand),
            "melds": [],
            "discards": list(seat.discards),
            "tree_revealed": False,
            "grab_charge_hits": 0,
            "grab_charge_limit": 0,
            "contract_targets": [],
            "contract_counter": 0,
            "passed_hu_this_round": seat.passed_hu_this_round,
        }
        all_discards = list(seat.discards) + list(opp.discards)
        remaining = _compute_remaining_counts([seat.hand, opp.hand], all_discards)

        # Integration note: with A-2a merged, passing opponent_discards turns
        # on the 75 per-tile features (hand_t_* / remain_t_* / opp_disc_t_*).
        features = build_model_features(
            active_player=active_player,
            remaining_counts=remaining,
            available_actions=["discard"],
            can_win=not seat.passed_hu_this_round,
            wall_remaining=len(self.wall),
            from_player=1 - seat_idx,
            target_hai=None,
            tree_active=False,
            grab_charge_active=False,
            contract_target_count=0,
            opponent_meld_count=0,
            opponent_discard_count=len(opp.discards),
            opponent_discards=list(opp.discards),
        )

        self.steps.append(
            SampleStep(
                game_index=self.game_index,
                step_index=len(self.steps),
                seat_index=seat_idx,
                seat_wind=seat.wind.value,
                turn=turn,
                wall_remaining=len(self.wall),
                policy=self.policies[seat_idx].name,
                engine=str(decision.engine),
                action="discard",
                discard_tile=decision.tile,
                hand_before=list(seat.hand),
                model_features=features,
            )
        )

    def play_one_turn(self) -> Optional[GameResult]:
        seat_idx = self.current_seat
        seat = self.seats[seat_idx]
        if not self.wall:
            return GameResult(winner=None, loser=None, kind="draw", turns=0)
        tile = self.wall.pop()
        seat.hand.append(tile)

        # Tsumo check first; if we tsumo we never emit a discard sample here.
        if is_winning_hand(seat.hand):
            return GameResult(winner=seat_idx, loser=None, kind="tsumo", turns=0)

        decision = self.policies[seat_idx].choose_discard(self, seat_idx)
        if decision.tile not in seat.hand:
            decision = DiscardDecision(tile=seat.hand[-1], engine="emergency_fallback")

        # Record BEFORE we mutate the hand so `hand_before` captures the
        # state the policy saw.
        turn_number = len(self.steps) + 1
        self._record_decision(seat_idx, turn_number, decision)

        seat.hand.remove(decision.tile)
        seat.hand.sort()
        seat.discards.append(decision.tile)

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

        self.seats[seat_idx].passed_hu_this_round = False
        self.current_seat = 1 - seat_idx
        return None


# --------------------------------------------------------------------------- #
# Annotation: attach outcome labels to each recorded step                     #
# --------------------------------------------------------------------------- #


def _sample_to_record(step: SampleStep, result: GameResult, total_steps: int) -> Dict[str, Any]:
    if result.winner is None:
        own_outcome = "draw"
    elif result.winner == step.seat_index:
        own_outcome = "win"
    else:
        own_outcome = "loss"

    can_win_label = own_outcome == "win"
    houjuu_label = result.houjuu_seat == step.seat_index
    # "tsumo_num" semantics: number of discards it takes until the game ends
    # from this step's point of view. Regression target for the tsumo_num
    # model; smaller == we were closer to tenpai.
    tsumo_num_label = max(0, total_steps - step.step_index)
    # A-2c-2: near-term win flag. Previously tenpai_label == can_win_label which
    # wasted a training head (both models learned the same thing -- confirmed in
    # A-2b/A-2c-1 where agari_prob and tenpai_prob had IDENTICAL val metrics).
    # Defining tenpai as "this seat won AND we are within 5 discards of the end"
    # gives the GBDT a SHORT-horizon signal: "is this state imminently winnable?"
    # while can_win_label keeps the LONG-horizon signal "will we win at all?".
    NEAR_WIN_HORIZON = 5
    tenpai_near_label = can_win_label and tsumo_num_label <= NEAR_WIN_HORIZON

    return {
        "record_id": f"game-{step.game_index:06d}-step-{step.step_index:03d}-{step.seat_wind}",
        "game_index": step.game_index,
        "step_index": step.step_index,
        "seat_index": step.seat_index,
        "seat_wind": step.seat_wind,
        "turn": step.turn,
        "wall_remaining": step.wall_remaining,
        "policy": step.policy,
        "engine": step.engine,
        "action": step.action,
        "discard_tile": step.discard_tile,
        "hand_before": step.hand_before,
        "model_features": step.model_features,
        "outcome": {
            "result": own_outcome,
            "kind": result.kind,
            "steps_to_end": max(0, total_steps - step.step_index),
            "winner_seat_index": result.winner,
            "loser_seat_index": result.loser,
            "houjuu_seat_index": result.houjuu_seat,
        },
        "task_labels": {
            "best_discard_tile": step.discard_tile,
            "can_win_label": can_win_label,
        },
        "source_meta": {
            "houjuu_label": int(bool(houjuu_label)),
            "tsumo_num_label": float(tsumo_num_label),
            "tenpai_label": int(bool(tenpai_near_label)),  # A-2c-2: near-term win (<=5 discards to game end)
            "betaori_label": int(
                bool(
                    own_outcome != "loss"
                    and not houjuu_label
                )
            ),
            "ryukyoku_label": int(result.kind == "draw"),
        },
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class GenerationStats:
    games: int = 0
    steps: int = 0
    wins_by_policy: Dict[str, int] = field(default_factory=dict)
    kinds: Dict[str, int] = field(default_factory=dict)

    def record(self, recorded: RecordedGame) -> None:
        self.games += 1
        self.steps += len(recorded.steps)
        if recorded.result is None:
            return
        kind = recorded.result.kind
        self.kinds[kind] = self.kinds.get(kind, 0) + 1


def run_sample_generation(
    policy_a: Policy,
    policy_b: Policy,
    games: int,
    *,
    seed: int = 0,
    swap_sides: bool = True,
    max_turns: int = 200,
    output_path: Optional[Path] = None,
) -> GenerationStats:
    stats = GenerationStats()
    fout = output_path.open("w", encoding="utf-8") if output_path is not None else None
    try:
        for i in range(games):
            rng = random.Random(seed + i)
            swap = swap_sides and (i % 2 == 1)
            if swap:
                policies: List[Policy] = [policy_b, policy_a]
            else:
                policies = [policy_a, policy_b]
            game = SamplingGame(policies, rng, game_index=i, max_turns=max_turns)
            result = game.run()
            total_steps = len(game.steps)
            if fout is not None:
                for step in game.steps:
                    record = _sample_to_record(step, result, total_steps)
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.record(RecordedGame(game_index=i, steps=game.steps, result=result))
    finally:
        if fout is not None:
            fout.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play (state, action, outcome) samples for B-2 training.")
    parser.add_argument("--policy-a", default="heuristic", help="Policy spec for seat A (heuristic|random|orchestrator[:label])")
    parser.add_argument("--policy-b", default="heuristic", help="Policy spec for seat B")
    parser.add_argument("--games", type=int, default=200, help="Number of games to simulate")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--max-turns", type=int, default=200, help="Max turns per game before draw")
    parser.add_argument("--no-swap", action="store_true", help="Disable side-swapping between games")
    parser.add_argument("--output", required=True, help="Output JSONL path for (state, action, outcome) samples")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policy_a = policy_factory(args.policy_a, rng)
    policy_b = policy_factory(args.policy_b, rng)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    stats = run_sample_generation(
        policy_a,
        policy_b,
        args.games,
        seed=args.seed,
        swap_sides=not args.no_swap,
        max_turns=args.max_turns,
        output_path=output_path,
    )
    elapsed = time.time() - start

    summary = {
        "policy_a": policy_a.name,
        "policy_b": policy_b.name,
        "games_played": stats.games,
        "steps_recorded": stats.steps,
        "output": str(output_path),
        "kinds": stats.kinds,
        "elapsed_seconds": round(elapsed, 3),
        "seed": args.seed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
