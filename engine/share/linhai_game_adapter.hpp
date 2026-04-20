#ifndef LINHAI_GAME_ADAPTER_HPP
#define LINHAI_GAME_ADAPTER_HPP

#include "types.hpp"
#include "../share/json11.hpp"
#include "linhai_ai_engine.hpp"
#include <vector>
#include <string>

namespace linhai {

// 临海麻将 → Akochan 状态转换器
// 处理：2人→4人适配、两花色（无筒子）、MJAI Moves构建

// 临海牌编码到Akochan牌编码的映射（一致的，但需要确认筒子区间被跳过）
// 万子: 1-9, 条子: 21-29, 字牌: 31-37  （和Akochan一样）
// 筒子: 11-19 在临海中不存在

// 临海使用的有效牌种 (25种)
static const int LINHAI_VALID_TILES[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9,         // 万子
    21, 22, 23, 24, 25, 26, 27, 28, 29, // 条子
    31, 32, 33, 34, 35, 36, 37          // 字牌
};
static const int LINHAI_VALID_TILE_COUNT = 25;
static const int LINHAI_TOTAL_TILES = 100;  // 25种 × 4张

// 检查是否为临海有效牌
bool is_linhai_valid_tile(int hai);

// 将 linhai::GameState 转换为 Akochan Game_State
// 2人→4人: player 0,1 = 真实玩家, player 2,3 = 虚拟玩家(空手牌)
Game_State to_akochan_game_state(const GameState& linhai_state);

// 构建最小化的 Moves (MJAI JSON格式游戏记录)
// Akochan 的许多函数需要 Moves 来获取游戏历史信息
// 我们构建: start_kyoku → (dahai序列) → tsumo (当前状态)
Moves build_minimal_moves(const GameState& linhai_state);

// 构建带有对手牌河信息的 Moves (更精确的防守分析)
Moves build_moves_with_discards(
    const GameState& linhai_state,
    const std::vector<int>& my_discards,      // 我的牌河
    const std::vector<int>& opponent_discards  // 对手牌河
);

// 从 linhai::GameState 提取可见牌数组
Hai_Array get_linhai_visible_tiles(const GameState& linhai_state);

// 生成临海的初始牌山（去掉筒子）用于虚拟玩家
Hai_Array get_linhai_remaining_pool(const GameState& linhai_state);

} // namespace linhai

#endif
