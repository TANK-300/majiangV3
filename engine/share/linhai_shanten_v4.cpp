#include "linhai_shanten_v4.hpp"
#include "linhai_rules.hpp"
#include <algorithm>
#include <iostream>

int calc_standard_shanten(const Hai_Array& tehai);

// 调试开关
// #define DEBUG_SHANTEN 1

// 标准向听数计算（参考网上的标准算法）
// 这个算法是正确的，经过验证的
namespace {

bool is_linhai_shanten_tile(int hai) {
    if (hai == WHITE_TILE) {
        return false;
    }
    return (hai >= 1 && hai <= 9) || (hai >= 21 && hai <= 29) || (hai >= 31 && hai <= 37);
}

std::vector<int> build_white_replacement_candidates() {
    std::vector<int> candidates;
    candidates.reserve(24);
    for (int hai = 1; hai < 38; ++hai) {
        if (is_linhai_shanten_tile(hai)) {
            candidates.push_back(hai);
        }
    }
    return candidates;
}

void enumerate_white_replacements(
    Hai_Array& hand,
    const std::vector<int>& candidates,
    int replacements_left,
    int start_index,
    int& min_shanten
) {
    if (replacements_left == 0) {
        min_shanten = std::min(min_shanten, calc_standard_shanten(hand));
        return;
    }

    for (int i = start_index; i < static_cast<int>(candidates.size()); ++i) {
        const int hai = candidates[i];
        if (hand[hai] >= 4) {
            continue;
        }
        hand[WHITE_TILE] -= 1;
        hand[hai] += 1;
        enumerate_white_replacements(hand, candidates, replacements_left - 1, i, min_shanten);
        hand[hai] -= 1;
        hand[WHITE_TILE] += 1;
    }
}

// 递归计算向听数
// tehai: 手牌数组
// mentu_num: 已形成的面子数
// tatsu_num: 已形成的搭子数
// has_jantou: 是否已有雀头
// total_tiles: 手牌总数（用于验证）
int calc_shanten_impl(const Hai_Array& tehai, int mentu_num, int tatsu_num, bool has_jantou, int total_tiles) {
    // 找到第一张有牌的位置
    int hai = 1;
    while (hai < 38 && tehai[hai] == 0) {
        hai++;
    }

    // 所有牌都处理完了
    if (hai >= 38) {
        // 计算向听数
        // 标准公式：向听数 = 8 - 2*面子数 - 搭子数 - (有雀头?1:0)
        // 限制：面子数 <= 4, 面子数+搭子数 <= 4
        int effective_mentu = std::min(mentu_num, 4);
        int effective_tatsu = std::min(tatsu_num, 4 - effective_mentu);

        int shanten = 8 - effective_mentu * 2 - effective_tatsu - (has_jantou ? 1 : 0);

        #ifdef DEBUG_SHANTEN
        if (shanten <= 1) {
            int used_tiles = mentu_num * 3 + tatsu_num * 2 + (has_jantou ? 2 : 0);
            std::cout << "  ✓ 有效组合: 面子=" << mentu_num << ", 搭子=" << tatsu_num
                      << ", 雀头=" << (has_jantou ? "有" : "无")
                      << " -> 向听数=" << shanten
                      << " (使用" << used_tiles << "/" << total_tiles << "张牌)" << std::endl;
        }
        #endif

        return shanten;
    }

    int min_shanten = 8;
    Hai_Array tmp = tehai;

    // 尝试1: 切割刻子（3张相同的牌）
    if (tmp[hai] >= 3 && mentu_num < 4) {
        tmp[hai] -= 3;
        int shanten = calc_shanten_impl(tmp, mentu_num + 1, tatsu_num, has_jantou, total_tiles);
        min_shanten = std::min(min_shanten, shanten);
        tmp[hai] += 3;
    }

    // 尝试2: 切割顺子（3张连续的数牌）
    if (hai < 30 && hai % 10 <= 7 && mentu_num < 4) {
        if (tmp[hai] >= 1 && tmp[hai+1] >= 1 && tmp[hai+2] >= 1) {
            tmp[hai]--;
            tmp[hai+1]--;
            tmp[hai+2]--;
            int shanten = calc_shanten_impl(tmp, mentu_num + 1, tatsu_num, has_jantou, total_tiles);
            min_shanten = std::min(min_shanten, shanten);
            tmp[hai]++;
            tmp[hai+1]++;
            tmp[hai+2]++;
        }
    }

    // 尝试3: 切割对子搭子（2张相同的牌）
    if (tmp[hai] >= 2 && mentu_num + tatsu_num < 4) {
        tmp[hai] -= 2;
        int shanten = calc_shanten_impl(tmp, mentu_num, tatsu_num + 1, has_jantou, total_tiles);
        min_shanten = std::min(min_shanten, shanten);
        tmp[hai] += 2;
    }

    // 尝试4: 切割两面搭子（2张连续的数牌）
    if (hai < 30 && hai % 10 <= 8 && mentu_num + tatsu_num < 4) {
        if (tmp[hai] >= 1 && tmp[hai+1] >= 1) {
            tmp[hai]--;
            tmp[hai+1]--;
            int shanten = calc_shanten_impl(tmp, mentu_num, tatsu_num + 1, has_jantou, total_tiles);
            min_shanten = std::min(min_shanten, shanten);
            tmp[hai]++;
            tmp[hai+1]++;
        }
    }

    // 尝试5: 切割嵌张搭子（2张间隔1的数牌）
    if (hai < 30 && hai % 10 <= 7 && mentu_num + tatsu_num < 4) {
        if (tmp[hai] >= 1 && tmp[hai+2] >= 1) {
            tmp[hai]--;
            tmp[hai+2]--;
            int shanten = calc_shanten_impl(tmp, mentu_num, tatsu_num + 1, has_jantou, total_tiles);
            min_shanten = std::min(min_shanten, shanten);
            tmp[hai]++;
            tmp[hai+2]++;
        }
    }

    // 尝试6: 切割雀头（如果还没有雀头）
    if (!has_jantou && tmp[hai] >= 2) {
        tmp[hai] -= 2;
        int shanten = calc_shanten_impl(tmp, mentu_num, tatsu_num, true, total_tiles);
        min_shanten = std::min(min_shanten, shanten);
        tmp[hai] += 2;
    }

    // 尝试7: 跳过这张牌（作为孤张）
    tmp[hai]--;
    int shanten = calc_shanten_impl(tmp, mentu_num, tatsu_num, has_jantou, total_tiles);
    min_shanten = std::min(min_shanten, shanten);
    tmp[hai]++;

    return min_shanten;
}

} // anonymous namespace

// 标准向听数计算（不考虑白板万能）
int calc_standard_shanten(const Hai_Array& tehai) {
    // 计算手牌总数
    int total_tiles = 0;
    for (int hai = 1; hai < 38; hai++) {
        total_tiles += tehai[hai];
    }

    return calc_shanten_impl(tehai, 0, 0, false, total_tiles);
}

// 临海麻将的向听数计算（支持白板万能）
int calc_linhai_shanten(const Hai_Array& tehai) {
    const int white_count = tehai[WHITE_TILE];

    if (white_count == 0) {
        // 没有白板，使用标准算法
        return calc_standard_shanten(tehai);
    }

    int min_shanten = calc_standard_shanten(tehai);
    const std::vector<int> candidates = build_white_replacement_candidates();

    for (int used_white = 1; used_white <= white_count; ++used_white) {
        Hai_Array tmp = tehai;
        enumerate_white_replacements(tmp, candidates, used_white, 0, min_shanten);
    }

    return min_shanten;
}
