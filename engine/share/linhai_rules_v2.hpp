#pragma once

#include "types.hpp"
#include "calc_shanten.hpp"

// 临海麻将规则扩展

// 白板牌的编号
// 当前项目的字牌映射是: 31-34 风牌, 35=白, 36=发, 37=中。
const int WHITE_TILE = 35;

// 检查是否是白板
inline bool is_white_tile(int hai) {
    return hai == WHITE_TILE;
}

// 向听数计算辅助结构
struct ShantenResult {
    int shanten;        // 向听数
    int mentu_num;      // 面子数
    int tatsu_num;      // 搭子数
    int toitsu_num;     // 对子数
    bool has_jantou;    // 是否有雀头

    ShantenResult() : shanten(8), mentu_num(0), tatsu_num(0), toitsu_num(0), has_jantou(false) {}
};

// 计算向听数
int calc_shanten_from_result(const ShantenResult& result);

// 临海麻将的向听数计算（支持白板万能）
void linhai_calc_shanten_recursive(
    const Hai_Array& tehai_tmp,
    int mentu_num,      // 已有面子数
    int tatsu_num,      // 已有搭子数
    bool has_jantou,    // 是否已有雀头
    int& min_shanten    // 最小向听数（引用传递）
);

// 临海麻将的听牌检查（支持白板万能）
void linhai_tenpai_check(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array& using_array, const Hai_Array& tehai, const Hai_Array& tehai_tate_cut, Hai_Array& tehai_tmp,
    const Fuuro_Vector& fuuro, Tenpai_Info& tenpai_info
);

// 临海麻将的手牌分析（支持白板万能）
void linhai_analyze_tehai(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array using_array, const Hai_Array tehai, const Fuuro_Vector fuuro, Tenpai_Info& tenpai_info
);

// 计算临海麻将的向听数（支持白板万能）
Tenpai_Info linhai_cal_tenpai_info(const int bakaze, const int jikaze, const Hai_Array& tehai, const Fuuro_Vector& fuuro);

// 临海麻将的顺子切割（支持白板万能）
void linhai_cut_syuntu(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array& using_array, const Hai_Array& tehai, const Hai_Array& tehai_tate_cut, Hai_Array& tehai_tmp,
    const int start, const Fuuro_Vector& fuuro, Tenpai_Info& tenpai_info,
    int mentu_num, int tatsu_num, bool has_jantou, int& min_shanten
);

// 临海麻将的刻子切割（支持白板万能）
void linhai_cut_kotu(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array& using_array, const Hai_Array& tehai, Hai_Array& tehai_tmp,
    const int start, const Fuuro_Vector& fuuro, Tenpai_Info& tenpai_info,
    int mentu_num, int tatsu_num, bool has_jantou, int& min_shanten
);

// 临海麻将的搭子分析（支持白板万能）
void linhai_analyze_tatu(
    const int bakaze_hai, const int jikaze_hai,
    const Hai_Array& using_array, const Hai_Array& tehai, const Hai_Array& tehai_tate_cut, Hai_Array& tehai_tmp,
    const Fuuro_Vector& fuuro, Tenpai_Info& tenpai_info,
    int mentu_num, int tatsu_num, bool has_jantou, int& min_shanten
);

// 抓冲牌判断
bool is_grab_charge_tile(int hai, int jikaze);

// 获取抓冲牌列表
std::vector<int> get_grab_charge_tiles(int jikaze);

// 计算抓冲期望值
float calc_grab_charge_expected_value(const Hai_Array& tehai, int jikaze);
