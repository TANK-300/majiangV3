#include "linhai_rules.hpp"
#include "calc_shanten.hpp"
#include "calc_agari.hpp"
#include <algorithm>

namespace {

bool is_linhai_wait_tile(int hai) {
    return (hai >= 1 && hai <= 9) || (hai >= 21 && hai <= 29) || (hai >= 31 && hai <= 37);
}

bool can_draw_wait_tile(const Hai_Array& using_array, int hai) {
    return is_linhai_wait_tile(hai) && using_array[hai] < 4;
}

void push_unique_agari(
    const int bakaze_hai,
    const int jikaze_hai,
    const Hai_Array& using_array,
    const Hai_Array& tehai,
    const Hai_Array& tehai_tate_cut,
    const Hai_Array& tehai_tmp,
    const Fuuro_Vector& fuuro,
    const int machi_hai,
    const Machi_Type machi_type,
    const bool titoi_flag,
    Tenpai_Info& tenpai_info
) {
    if (!can_draw_wait_tile(using_array, machi_hai)) {
        return;
    }
    Agari_Info agari = calc_agari(
        bakaze_hai, jikaze_hai,
        tehai, tehai_tate_cut, tehai_tmp, fuuro,
        machi_hai, machi_type, titoi_flag
    );
    auto same = [&](const Agari_Info& existing) {
        return existing.hai == agari.hai &&
               existing.han_tsumo == agari.han_tsumo &&
               existing.fu_tsumo == agari.fu_tsumo &&
               existing.han_ron == agari.han_ron &&
               existing.fu_ron == agari.fu_ron;
    };
    if (std::find_if(tenpai_info.agari_vec.begin(), tenpai_info.agari_vec.end(), same) == tenpai_info.agari_vec.end()) {
        tenpai_info.mentu_shanten_num = 0;
        tenpai_info.agari_vec.push_back(agari);
    }
}

} // namespace

// 临海麻将的听牌检查（支持白板万能）
void linhai_tenpai_check(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array& using_array, const Hai_Array& tehai, const Hai_Array& tehai_tate_cut, Hai_Array& tehai_tmp,
    const Fuuro_Vector& fuuro, Tenpai_Info& tenpai_info
) {
    int rest = 0;
    int white_count = 0;

    for(int hai = 0; hai < 38; hai++) {
        rest += tehai_tmp[hai];
        if (hai == WHITE_TILE) {
            white_count = tehai_tmp[hai];
        }
    }

    if (rest == 0) {
        // 已经是胡牌形式
        return;
    } else if (rest == 1) {
        if (white_count == 1 && tehai_tmp[WHITE_TILE] == 1) {
            for (int hai = 1; hai < 38; ++hai) {
                if (!is_linhai_wait_tile(hai)) {
                    continue;
                }
                push_unique_agari(
                    bakaze_hai, jikaze_hai,
                    using_array, tehai, tehai_tate_cut, tehai_tmp, fuuro,
                    hai, MT_TANKI, false, tenpai_info
                );
            }
            return;
        }
        // 单骑听牌
        for (int hai = 0; hai < 38; hai++) {
            if (tehai_tmp[hai] == 1 && can_draw_wait_tile(using_array, hai)) {
                push_unique_agari(
                    bakaze_hai, jikaze_hai,
                    using_array,
                    tehai, tehai_tate_cut, tehai_tmp, fuuro,
                    hai, MT_TANKI, false, tenpai_info
                );
            }
        }
    } else if (rest == 2) {
        // 对子听牌或两面/嵌张/边张听牌

        // 1. 对子听牌（双碰）
        for (int hai = 0; hai < 38; hai++) {
            if (tehai_tmp[hai] == 2 && can_draw_wait_tile(using_array, hai)) {
                push_unique_agari(
                    bakaze_hai, jikaze_hai,
                    using_array,
                    tehai, tehai_tate_cut, tehai_tmp, fuuro,
                    hai, MT_SHABO, false, tenpai_info
                );
            }
        }

        // 2. 顺子听牌
        for (int hai = 0; hai < 30; hai++) {
            // 两张牌的情况
            if (tehai_tmp[hai] == 1 && tehai_tmp[hai+1] == 1) {
                // 边张（12听3或89听7）
                if (hai % 10 == 1) {
                    if (can_draw_wait_tile(using_array, hai + 2)) {
                        push_unique_agari(
                            bakaze_hai, jikaze_hai,
                            using_array,
                            tehai, tehai_tate_cut, tehai_tmp, fuuro,
                            hai + 2, MT_PENCHAN, false, tenpai_info
                        );
                    }
                } else if (hai % 10 == 8) {
                    if (can_draw_wait_tile(using_array, hai - 1)) {
                        push_unique_agari(
                            bakaze_hai, jikaze_hai,
                            using_array,
                            tehai, tehai_tate_cut, tehai_tmp, fuuro,
                            hai - 1, MT_PENCHAN, false, tenpai_info
                        );
                    }
                } else {
                    // 两面听牌
                    if (can_draw_wait_tile(using_array, hai - 1)) {
                        push_unique_agari(
                            bakaze_hai, jikaze_hai,
                            using_array,
                            tehai, tehai_tate_cut, tehai_tmp, fuuro,
                            hai - 1, MT_RYANMEN, false, tenpai_info
                        );
                    }
                    if (can_draw_wait_tile(using_array, hai + 2)) {
                        push_unique_agari(
                            bakaze_hai, jikaze_hai,
                            using_array,
                            tehai, tehai_tate_cut, tehai_tmp, fuuro,
                            hai + 2, MT_RYANMEN, false, tenpai_info
                        );
                    }
                }
            }

            // 嵌张（13听2，24听3等）
            if (tehai_tmp[hai] == 1 && tehai_tmp[hai+2] == 1) {
                if (hai % 10 != 9) {
                    if (can_draw_wait_tile(using_array, hai + 1)) {
                        push_unique_agari(
                            bakaze_hai, jikaze_hai,
                            using_array,
                            tehai, tehai_tate_cut, tehai_tmp, fuuro,
                            hai + 1, MT_KANCHAN, false, tenpai_info
                        );
                    }
                }
            }
        }

        // 3. 白板万能的特殊情况
        if (white_count == 1) {
            // 如果有一张白板，可以和任意一张牌组成对子
            for (int hai = 0; hai < 38; hai++) {
                if (hai != WHITE_TILE && tehai_tmp[hai] == 1 && can_draw_wait_tile(using_array, hai)) {
                    // 白板+任意牌可以听该牌（组成对子）
                    push_unique_agari(
                        bakaze_hai, jikaze_hai,
                        using_array,
                        tehai, tehai_tate_cut, tehai_tmp, fuuro,
                        hai, MT_SHABO, false, tenpai_info
                    );
                }
            }
        }
        if (white_count == 2 && tehai_tmp[WHITE_TILE] == 2) {
            for (int hai = 1; hai < 38; ++hai) {
                if (!is_linhai_wait_tile(hai)) {
                    continue;
                }
                push_unique_agari(
                    bakaze_hai, jikaze_hai,
                    using_array, tehai, tehai_tate_cut, tehai_tmp, fuuro,
                    hai, MT_SHABO, false, tenpai_info
                );
            }
        }
    }
}

// 临海麻将的顺子切割（支持白板万能）
void linhai_cut_syuntu(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array& using_array, const Hai_Array& tehai, const Hai_Array& tehai_tate_cut, Hai_Array& tehai_tmp,
    const int start, const Fuuro_Vector& fuuro, Tenpai_Info& tenpai_info
) {
    int white_count = tehai_tmp[WHITE_TILE];

    for (int hai = start; hai < 30; hai++) {
        // 标准顺子（不使用白板）
        if (tehai_tmp[hai] >= 1 && tehai_tmp[hai+1] >= 1 && tehai_tmp[hai+2] >= 1) {
            tehai_tmp[hai]--;
            tehai_tmp[hai+1]--;
            tehai_tmp[hai+2]--;
            linhai_cut_syuntu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tate_cut, tehai_tmp, hai, fuuro, tenpai_info);
            tehai_tmp[hai]++;
            tehai_tmp[hai+1]++;
            tehai_tmp[hai+2]++;
        }

        // 使用白板组成顺子
        if (white_count > 0) {
            // 白板替代第一张
            if (tehai_tmp[hai+1] >= 1 && tehai_tmp[hai+2] >= 1) {
                tehai_tmp[WHITE_TILE]--;
                tehai_tmp[hai+1]--;
                tehai_tmp[hai+2]--;
                linhai_cut_syuntu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tate_cut, tehai_tmp, hai, fuuro, tenpai_info);
                tehai_tmp[WHITE_TILE]++;
                tehai_tmp[hai+1]++;
                tehai_tmp[hai+2]++;
            }

            // 白板替代第二张
            if (tehai_tmp[hai] >= 1 && tehai_tmp[hai+2] >= 1) {
                tehai_tmp[hai]--;
                tehai_tmp[WHITE_TILE]--;
                tehai_tmp[hai+2]--;
                linhai_cut_syuntu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tate_cut, tehai_tmp, hai, fuuro, tenpai_info);
                tehai_tmp[hai]++;
                tehai_tmp[WHITE_TILE]++;
                tehai_tmp[hai+2]++;
            }

            // 白板替代第三张
            if (tehai_tmp[hai] >= 1 && tehai_tmp[hai+1] >= 1) {
                tehai_tmp[hai]--;
                tehai_tmp[hai+1]--;
                tehai_tmp[WHITE_TILE]--;
                linhai_cut_syuntu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tate_cut, tehai_tmp, hai, fuuro, tenpai_info);
                tehai_tmp[hai]++;
                tehai_tmp[hai+1]++;
                tehai_tmp[WHITE_TILE]++;
            }
        }
    }

    // 切割完顺子后，分析搭子
    linhai_analyze_tatu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tate_cut, tehai_tmp, fuuro, tenpai_info);
}

// 临海麻将的刻子切割（支持白板万能）
void linhai_cut_kotu(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array& using_array, const Hai_Array& tehai, Hai_Array& tehai_tmp,
    const int start, const Fuuro_Vector& fuuro, Tenpai_Info& tenpai_info
) {
    int white_count = tehai_tmp[WHITE_TILE];

    for (int hai = start; hai < 38; hai++) {
        // 标准刻子（3张相同）
        if (tehai_tmp[hai] >= 3) {
            tehai_tmp[hai] -= 3;
            linhai_cut_kotu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tmp, hai, fuuro, tenpai_info);
            tehai_tmp[hai] += 3;
        }

        // 使用白板组成刻子
        if (white_count >= 1 && tehai_tmp[hai] >= 2 && hai != WHITE_TILE) {
            // 2张牌 + 1张白板 = 刻子
            tehai_tmp[hai] -= 2;
            tehai_tmp[WHITE_TILE]--;
            linhai_cut_kotu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tmp, hai, fuuro, tenpai_info);
            tehai_tmp[hai] += 2;
            tehai_tmp[WHITE_TILE]++;
        }

        if (white_count >= 2 && tehai_tmp[hai] >= 1 && hai != WHITE_TILE) {
            // 1张牌 + 2张白板 = 刻子
            tehai_tmp[hai]--;
            tehai_tmp[WHITE_TILE] -= 2;
            linhai_cut_kotu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tmp, hai, fuuro, tenpai_info);
            tehai_tmp[hai]++;
            tehai_tmp[WHITE_TILE] += 2;
        }
    }

    // 切割完刻子后，切割顺子
    Hai_Array tehai_tate_cut = tehai_tmp;
    linhai_cut_syuntu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tate_cut, tehai_tmp, 0, fuuro, tenpai_info);
}

// 临海麻将的搭子分析（支持白板万能）
void linhai_analyze_tatu(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array& using_array, const Hai_Array& tehai, const Hai_Array& tehai_tate_cut, Hai_Array& tehai_tmp,
    const Fuuro_Vector& fuuro, Tenpai_Info& tenpai_info
) {
    // 检查是否有明显的面子（剪枝优化）
    for(int hai = 0; hai < 38; hai++) {
        if(tehai_tmp[hai] >= 3) {
            return;
        }
    }

    for (int j = 0; j < 3; j++) {
        for (int i = 1; i <= 7; i++) {
            if (tehai_tmp[10*j+i] > 0 && tehai_tmp[10*j+i+1] > 0 && tehai_tmp[10*j+i+2] > 0) {
                return;
            }
        }
    }

    linhai_tenpai_check(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tate_cut, tehai_tmp, fuuro, tenpai_info);
}

// 临海麻将的手牌分析（支持白板万能）
void linhai_analyze_tehai(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array using_array, const Hai_Array tehai, const Fuuro_Vector fuuro, Tenpai_Info& tenpai_info
) {
    Hai_Array tehai_tmp = tehai;

    // 先尝试切割雀头（对子）
    for (int hai = 0; hai < 38; hai++) {
        if (tehai_tmp[hai] >= 2) {
            tehai_tmp[hai] -= 2;
            linhai_cut_kotu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tmp, 0, fuuro, tenpai_info);
            tehai_tmp[hai] += 2;
        }

        // 白板+任意牌组成雀头
        if (tehai_tmp[WHITE_TILE] >= 1 && tehai_tmp[hai] >= 1 && hai != WHITE_TILE) {
            tehai_tmp[WHITE_TILE]--;
            tehai_tmp[hai]--;
            linhai_cut_kotu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tmp, 0, fuuro, tenpai_info);
            tehai_tmp[WHITE_TILE]++;
            tehai_tmp[hai]++;
        }
    }

    // 不切割雀头的情况（用于计算向听数）
    linhai_cut_kotu(bakaze_hai, jikaze_hai, using_array, tehai, tehai_tmp, 0, fuuro, tenpai_info);

    // 七对子向听数
    if (fuuro.size() == 0) {
        titoi_shanten(bakaze_hai, jikaze_hai, tehai, fuuro, tenpai_info);
    }
}

// 计算临海麻将的向听数（支持白板万能）
Tenpai_Info linhai_cal_tenpai_info(const int bakaze, const int jikaze, const Hai_Array& tehai, const Fuuro_Vector& fuuro) {
    Tenpai_Info tenpai_info;
    tenpai_info.mentu_shanten_num = 8;
    tenpai_info.titoi_shanten_num = 8;

    Hai_Array using_array = using_hai_array(tehai, fuuro);

    // 标准型向听数
    linhai_analyze_tehai(bakaze, jikaze, using_array, tehai, fuuro, tenpai_info);

    // 七对子向听数（临海麻将也支持七对子）
    titoi_shanten(bakaze, jikaze, tehai, fuuro, tenpai_info);

    return tenpai_info;
}

// 抓冲牌判断
bool is_grab_charge_tile(int hai, int jikaze) {
    // jikaze: 0=东, 1=南, 2=西, 3=北
    // hai编号: 1-9万(1-9), 11-19筒(11-19), 21-29索(21-29), 31-34风(东南西北), 35-37箭(白发中)

    switch(jikaze) {
        case 0: // 东风家: 东风 + 1/5/9万条筒
            return hai == 31 || hai == 1 || hai == 5 || hai == 9 ||
                   hai == 11 || hai == 15 || hai == 19 ||
                   hai == 21 || hai == 25 || hai == 29;
        case 1: // 南风家: 南风 + 2/6万条筒 + 中
            return hai == 32 || hai == 2 || hai == 6 ||
                   hai == 12 || hai == 16 ||
                   hai == 22 || hai == 26 || hai == 37;
        case 2: // 西风家: 西风 + 3/7万条筒 + 发
            return hai == 33 || hai == 3 || hai == 7 ||
                   hai == 13 || hai == 17 ||
                   hai == 23 || hai == 27 || hai == 36;
        case 3: // 北风家: 北风 + 4/8万条筒 + 白板
            return hai == 34 || hai == 4 || hai == 8 ||
                   hai == 14 || hai == 18 ||
                   hai == 24 || hai == 28 || hai == 35;
        default:
            return false;
    }
}

// 获取抓冲牌列表
std::vector<int> get_grab_charge_tiles(int jikaze) {
    std::vector<int> tiles;
    for (int hai = 1; hai < 38; hai++) {
        if (is_grab_charge_tile(hai, jikaze)) {
            tiles.push_back(hai);
        }
    }
    return tiles;
}

// 计算抓冲期望值
float calc_grab_charge_expected_value(const Hai_Array& tehai, int jikaze) {
    float value = 0.0f;
    std::vector<int> grab_tiles = get_grab_charge_tiles(jikaze);

    for (int hai : grab_tiles) {
        // 每张抓冲牌的期望值
        // 假设胡牌后有1/6概率抓到该牌，每张翻倍
        value += tehai[hai] * 50.0f; // 每张抓冲牌价值50分
    }

    return value;
}
