#pragma once

#include "types.hpp"
#include "calc_shanten.hpp"
#include <vector>

// 白板牌的编号
// 当前项目的字牌映射是: 31-34 风牌, 35=白, 36=发, 37=中。
extern const int WHITE_TILE;

// 向听数计算辅助函数
int calc_shanten_from_counts(int mentu_num, int tatsu_num, bool has_jantou);

// 临海麻将的向听数计算（递归，支持白板万能）
void linhai_calc_shanten_recursive(
    const Hai_Array& tehai_tmp,
    int start,              // 从哪张牌开始扫描
    int mentu_num,          // 已有面子数
    int tatsu_num,          // 已有搭子数
    bool has_jantou,        // 是否已有雀头
    int& min_shanten        // 最小向听数（引用传递）
);

// 临海麻将的向听数计算（支持白板万能）
Tenpai_Info linhai_cal_tenpai_info(const int bakaze, const int jikaze, const Hai_Array& tehai, const Fuuro_Vector& fuuro);

// 抓冲牌判断
bool is_grab_charge_tile(int hai, int jikaze);

// 获取抓冲牌列表
std::vector<int> get_grab_charge_tiles(int jikaze);

// 计算抓冲期望值
float calc_grab_charge_expected_value(const Hai_Array& tehai, int jikaze);
