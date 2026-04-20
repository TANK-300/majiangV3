from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional


class Suit(Enum):
    CHARACTERS = "characters"
    BAMBOO = "bamboo"
    HONOR = "honor"


class Honor(Enum):
    EAST = "east"
    SOUTH = "south"
    WEST = "west"
    NORTH = "north"
    RED = "red"
    GREEN = "green"
    WHITE = "white"


@dataclass(frozen=True)
class TileDescriptor:
    code: str
    suit: Suit
    rank: Optional[int]
    honor: Optional[Honor]

    @property
    def is_honor(self) -> bool:
        return self.suit is Suit.HONOR


def _build_tile(code: str, suit: Suit, rank: Optional[int] = None, honor: Optional[Honor] = None) -> TileDescriptor:
    return TileDescriptor(code=code, suit=suit, rank=rank, honor=honor)


TILES: Dict[str, TileDescriptor] = {}

for i in range(1, 10):
    TILES[f"{i}w"] = _build_tile(f"{i}w", Suit.CHARACTERS, rank=i)
    TILES[f"{i}t"] = _build_tile(f"{i}t", Suit.BAMBOO, rank=i)  # t = 条子 (tiao)

TILES["east"] = _build_tile("east", Suit.HONOR, honor=Honor.EAST)
TILES["south"] = _build_tile("south", Suit.HONOR, honor=Honor.SOUTH)
TILES["west"] = _build_tile("west", Suit.HONOR, honor=Honor.WEST)
TILES["north"] = _build_tile("north", Suit.HONOR, honor=Honor.NORTH)
TILES["red"] = _build_tile("red", Suit.HONOR, honor=Honor.RED)
TILES["green"] = _build_tile("green", Suit.HONOR, honor=Honor.GREEN)
TILES["white"] = _build_tile("white", Suit.HONOR, honor=Honor.WHITE)


def require_tile(code: str) -> TileDescriptor:
    try:
        return TILES[code]
    except KeyError as exc:
        raise ValueError(f"Unknown tile code: {code}") from exc


def all_tile_codes() -> List[str]:
    return list(TILES.keys())


def honor_codes() -> List[str]:
    return [code for code, tile in TILES.items() if tile.is_honor]


def suited_codes() -> List[str]:
    return [code for code, tile in TILES.items() if not tile.is_honor]


def normalize_code(code: str) -> str:
    c = code.strip().lower()
    if c not in TILES:
        raise ValueError(f"Unsupported tile code: {code}")
    return c


def suit_to_suffix(suit: Suit) -> str:
    return {
        Suit.CHARACTERS: "w",
        Suit.BAMBOO: "t",  # t = 条子 (tiao)
        Suit.HONOR: "h",
    }[suit]


def bulk_count(codes: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {code: 0 for code in TILES}
    for code in codes:
        normalized = normalize_code(code)
        counts[normalized] += 1
    return counts


def to_chinese_name(code: str) -> str:
    """将代码转换为中文名称"""
    try:
        tile = require_tile(code)
        if tile.is_honor:
            return {
                Honor.EAST: "东风",
                Honor.SOUTH: "南风",
                Honor.WEST: "西风",
                Honor.NORTH: "北风",
                Honor.RED: "红中",
                Honor.GREEN: "发财",
                Honor.WHITE: "白板",
            }[tile.honor]  # type: ignore
        
        num_map = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
        rank_str = num_map.get(tile.rank, str(tile.rank))
        
        suit_str = {
            Suit.CHARACTERS: "万",
            Suit.BAMBOO: "条",
        }.get(tile.suit, "")
        
        return f"{rank_str}{suit_str}"
    except:
        return code
