from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .tiles import TileDescriptor, normalize_code


class Wind(Enum):
    EAST = "east"
    SOUTH = "south"
    WEST = "west"
    NORTH = "north"


TileSet = List[str]


@dataclass
class Meld:
    tiles: TileSet
    type: str

    def normalized(self) -> List[TileDescriptor]:
        return [normalize_code(tile) for tile in self.tiles]


@dataclass
class PlayerState:
    wind: Wind
    hand: TileSet = field(default_factory=list)
    melds: List[Meld] = field(default_factory=list)
    discards: TileSet = field(default_factory=list)
    red_heads: int = 0
    tree_tiles: List[str] = field(default_factory=list)
    tree_revealed: bool = False
    grab_charge_hits: int = 0
    grab_charge_limit: int = 6
    red_head_count: int = 0
    red_head_limit: int = 0
    red_head_values: List[str] = field(default_factory=list)
    contract_targets: List[Wind] = field(default_factory=list)
    contract_counter: int = 0

    def hand_counts(self) -> Dict[str, int]:
        return {tile: self.hand.count(tile) for tile in self.hand}


@dataclass
class GameState:
    round_wind: Wind
    dealer_wind: Wind
    active_wind: Wind
    players: Dict[Wind, PlayerState]
    wall_remaining: int
    dora_indicators: TileSet = field(default_factory=list)
    red_head_pool: List[Optional[str]] = field(default_factory=list)
    last_discard: Optional[str] = None
    last_discard_wind: Optional[Wind] = None
    latest_action_kind: Optional[str] = None
    tree_active: bool = False
    grab_charge_active: bool = False
    tree_bonuses: Dict[Wind, int] = field(default_factory=dict)
    grab_charge_sequences: Dict[Wind, List[str]] = field(default_factory=dict)
    contract_mapping: Dict[Wind, List[Wind]] = field(default_factory=dict)
    red_head_rules: Dict[str, int] = field(default_factory=dict)
    red_head_draw_order: List[str] = field(default_factory=list)

    def current_player(self) -> PlayerState:
        return self.players[self.active_wind]


def next_wind(wind: Wind) -> Wind:
    order = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
    idx = order.index(wind)
    return order[(idx + 1) % len(order)]


def previous_wind(wind: Wind) -> Wind:
    order = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
    idx = order.index(wind)
    return order[(idx - 1) % len(order)]
