"""二人临海麻将路由后置适配。

当前 V3 C++ 引擎与参数包是以 4 人临海（接近立直麻将的 yakuhai 偏好）训练的，
在奈苑临海 / 杭州临海二人玩法下会出现：

* 孤张字牌不作 yakuhai，却被引擎估重，应优先打出。
* 3 家估放炮 → 二人只 1 对手，放炮率偏高。
* 缺一门：自家缺某门时该门牌全不能留；对手缺某门时该门任意牌完全安全。

适配器仅重新排序 V3 返回的 candidate_scores，不凭空造任何手牌里没有的牌，以避免
"推荐手里没有的牌"这类回归。

这不是最终方案 — P1 需重训二人临海参数包才能取代。
"""
from __future__ import annotations

from typing import Dict, Optional, Set

from ..core.state import GameState
from ..core.tiles import normalize_code


HONOR_TILES: Set[str] = {"east", "south", "west", "north", "white", "green", "red"}

# 字牌奖励（自适应）：linhai 本体不计 yakuhai，按当前手张数决定弃留倾向。
# 1 张：独张 → +800（强烈弃）；2 张：对子 → 轻微弃 -200（让 V3 EV 决定）；
# 3 张：暗刻 → 强保 -800；4 张：可杠 → -1500。
# 思路是 monotonic：count 越大越要保（负奖励越深），非线性是为了不让对子被无脑保留。
def _honor_bonus_for_count(count: int) -> float:
    if count <= 1:
        return 800.0
    if count == 2:
        return -200.0
    if count == 3:
        return -800.0
    return -1500.0


# 兼容老调用：保留同名常量供测试 / 旧逻辑读取（现在仅在不显式调函数时用）。
LONE_HONOR_BONUS = 800.0

# 缺一门：自家缺某门价牌必须优先打出。加到 EV 上足以压过任何保牌动机。
# 实测（adapter 层调参均失败 — V3 shanten 把缺门牌算作可用牌，根本矛盾在训练而非调参）：
#   - 5000（强制清门）  : houjuu 23.5%、winrate 52%、窗口正率 57.6%
#   - 1000（软偏好）    : houjuu 30%、 winrate 42.5%、窗口正率 36%
#   - 5000 + shanten≤2 : houjuu 38%、 winrate 10%、 窗口正率 0.8%（牌型崩塌：临近听牌
#                       突然要清，V3 的 meld 规划全部作废）
# 三种方案都低于"不开缺一门"基线 (winrate 64%、windows 76.8%)。结论：missing_suit
# 必须在 V3 训练时内化，adapter 层任何参数都救不了。生产保持 missing_suit_enabled=False。
MISSING_SUIT_OWN_BONUS = 5000.0

# 缺一门 + 现物：清门时优先打对手已经河出过的牌（绝对安全），降低被荣和的风险。
# 提到 5000：与 OWN_BONUS 同量级，让"安全缺门"硬性优于"危险缺门"，
# 对齐 reference_human 的二值现物门控（同门池中有现物则只挑现物）。
# 之前 500 太弱，差被 V3 EV 噪声淹没，houjuu% 从 15% 飙到 23.5%。
MISSING_SUIT_OWN_SAFE_BONUS = 5000.0

# 对手缺某门该门任意牌全部绝对安全。加分略小，不涵盖自家缺一门。
MISSING_SUIT_OPP_BONUS = 200.0

# 二人推断：放炮率乘以该系数 ≈ 1 对手 / 3 对手。
TWO_PLAYER_HOUJUU_FACTOR = 0.4

# 现物（genbutsu）安全：禁用——V3 引擎自身已经把 opp_discards 作为防御特征算进 EV。
# 之前设为 1000 重复加分导致 EV 评分被破坏（avg_chong 4.07 → 1.15 / 胜率 -12pp）。
# 仅在 betaori 模式下用作安全牌识别。
SAFE_TILE_BONUS = 0.0

# 止损 / 全弃和（betaori）模式：仅在牌局已基本无望时启用，避免把还有希望的局放弃。
# 之前 shanten≥2 + wall<25 触发太宽，把中盘还能逆转的局直接弃和。
BETAORI_SHANTEN_THRESHOLD = 3  # shanten ≥ 此值进入 betaori（确实远离听牌）
BETAORI_WALL_THRESHOLD = 15    # wall_remaining < 此值进入 betaori（确实进入终盘）
BETAORI_SAFE_BONUS = 10000.0   # 绝对优先级，安全牌一定排第一

# valid suit 代码
_SUIT_W = "w"  # 万
_SUIT_T = "t"  # 条/索
_SUIT_Z = "z"  # 字牌
_VALID_SUITS = {_SUIT_W, _SUIT_T, _SUIT_Z}


def is_two_player_state(state: GameState) -> bool:
    """启发式判别 2 人状态：除本人外最多 1 个座位有牌河 / 副露。

    不是 100% 准（4 人开局第 1 圣还没人出牌会误判），但字牌独张优先打本身
    在 4p linhai 也是合理的，误判代价低。
    """
    me = state.active_wind
    active_others = 0
    for wind, p in state.players.items():
        if wind == me:
            continue
        if p.discards or p.melds:
            active_others += 1
    return active_others <= 1


def _suit_of(tile: str) -> str:
    """返回 tile 属于 w / t / z。"""
    if tile in HONOR_TILES:
        return _SUIT_Z
    if tile.endswith(_SUIT_W):
        return _SUIT_W
    if tile.endswith(_SUIT_T):
        return _SUIT_T
    return ""


def _normalize_suit_code(suit: Optional[str]) -> Optional[str]:
    """接受 'w' / 'wan' / '万' / 'm' 等别名。"""
    if not suit:
        return None
    s = suit.strip().lower()
    if s in _VALID_SUITS:
        return s
    aliases = {
        "m": _SUIT_W, "wan": _SUIT_W, "char": _SUIT_W, "characters": _SUIT_W,
        "万": _SUIT_W,
        "s": _SUIT_T, "tiao": _SUIT_T, "bamboo": _SUIT_T,
        "条": _SUIT_T, "索": _SUIT_T,
        "honor": _SUIT_Z, "zi": _SUIT_Z,
        "字": _SUIT_Z, "风": _SUIT_Z,
    }
    return aliases.get(s)


def adjust_for_2p_linhai(
    result: Optional[Dict],
    state: GameState,
    *,
    mode: Optional[str] = None,
    missing_suit_self: Optional[str] = None,
    missing_suit_opp: Optional[str] = None,
) -> Optional[Dict]:
    """后置调整 V3 返回结果。

    参数：
        result: orchestrator V3 上一跳返回的 dict（可为 None，原样返回）
        state: 当前请求的 GameState
        mode: 前端显式传过来的模式（如 "linhai_2p"）；None 时走启发式推断
        missing_suit_self: 自家缺一门（w / t / z 或别名）
        missing_suit_opp: 对手缺一门

    返回：调整后的 dict。不修改原输入（返回 shallow copy）。
    """
    if not result or not isinstance(result, dict):
        return result

    is_2p = (mode == "linhai_2p") or (mode is None and is_two_player_state(state))
    own_suit = _normalize_suit_code(missing_suit_self)
    opp_suit = _normalize_suit_code(missing_suit_opp)

    me = state.players.get(state.active_wind)
    hand_counts: Dict[str, int] = {}
    if me:
        for tile in me.hand:
            try:
                norm = normalize_code(tile)
            except ValueError:
                continue
            hand_counts[norm] = hand_counts.get(norm, 0) + 1

    # 现物（genbutsu）：对手已经打过的牌，对手不可能荣和这些牌。
    # 仅看非本人的座位的 discards；本人 discards 与对手的荣和无关。
    safe_tiles: Set[str] = set()
    for wind, p in state.players.items():
        if wind == state.active_wind:
            continue
        for tile in (p.discards or []):
            try:
                safe_tiles.add(normalize_code(tile))
            except ValueError:
                continue

    # 止损 / betaori 触发条件：shanten 已经差，且牌山快空，
    # 此时硬冲和牌的期望低于放炮风险，直接全打安全牌。
    shanten = result.get("shanten")
    try:
        shanten_int = int(shanten) if shanten is not None else 99
    except (TypeError, ValueError):
        shanten_int = 99
    wall_remaining = int(getattr(state, "wall_remaining", 0) or 0)
    betaori_mode = (
        is_2p
        and shanten_int >= BETAORI_SHANTEN_THRESHOLD
        and wall_remaining < BETAORI_WALL_THRESHOLD
        and bool(safe_tiles)  # 没有安全牌则 betaori 没意义，回退正常逻辑
    )

    raw_candidates = result.get("candidate_scores")
    out = dict(result)
    if not isinstance(raw_candidates, dict) or not raw_candidates:
        if is_2p:
            out["houjuu_prob"] = round(
                float(result.get("houjuu_prob", 0.0) or 0.0) * TWO_PLAYER_HOUJUU_FACTOR, 6
            )
        return out

    def _bonus_for(norm: str, suit: str) -> float:
        b = 0.0
        if norm in HONOR_TILES:
            cnt = hand_counts.get(norm, 0)
            if cnt >= 1:
                b += _honor_bonus_for_count(cnt)
        if own_suit and suit == own_suit:
            b += MISSING_SUIT_OWN_BONUS
            # 现物缺门牌：在被强制清门的多张候选里优先选最安全的那张。
            if norm in safe_tiles:
                b += MISSING_SUIT_OWN_SAFE_BONUS
        if opp_suit and suit == opp_suit:
            b += MISSING_SUIT_OPP_BONUS
        if norm in safe_tiles:
            b += SAFE_TILE_BONUS
        if betaori_mode and norm in safe_tiles:
            b += BETAORI_SAFE_BONUS
        return b

    new_scores: Dict[str, float] = {}
    bonuses_applied = False
    seen_norm: Set[str] = set()
    for tile_key, raw_score in raw_candidates.items():
        if not isinstance(raw_score, (int, float)):
            continue
        score = float(raw_score)
        try:
            norm = normalize_code(tile_key)
        except ValueError:
            new_scores[tile_key] = round(score, 3)
            continue
        seen_norm.add(norm)
        suit = _suit_of(norm)
        bonus = _bonus_for(norm, suit)
        if bonus != 0.0:
            bonuses_applied = True
        new_scores[tile_key] = round(score + bonus, 3)

    # 注入手牌里但 V3 candidate_scores 未返回的牌。必要——
    # V3 仅返回 top-N 候选；若缺一门 / 安全牌被修剪掉，adapter 给 0 个目标加分不生效。
    if new_scores and (own_suit or opp_suit or bonuses_applied or betaori_mode):
        existing_min = min(new_scores.values())
        for norm, _count in list(hand_counts.items()):
            if norm in seen_norm:
                continue
            suit = _suit_of(norm)
            bonus = _bonus_for(norm, suit)
            if bonus <= 0:
                continue
            # 注入牌以 existing_min 为基线 + bonus。
            # missing_suit (5000) / betaori_safe (10000) 足以 dominate；
            # lone_honor (800) / safe_tile (1000) 仅在接近最差候选时生效。
            new_scores[norm] = round(existing_min + bonus, 3)
            bonuses_applied = True

    if new_scores:
        out["candidate_scores"] = new_scores
        out["tile"] = max(new_scores, key=new_scores.get)
        if bonuses_applied:
            out["chosen_by"] = (result.get("chosen_by") or "search") + "+2p_adapter"

    if is_2p:
        out["houjuu_prob"] = round(
            float(result.get("houjuu_prob", 0.0) or 0.0) * TWO_PLAYER_HOUJUU_FACTOR, 6
        )

    return out
