#ifndef LINHAI_BONUS_HPP
#define LINHAI_BONUS_HPP

#include "types.hpp"

namespace linhai {

// 翻屁股（抓码加分）规则：
// 胡牌后抓码，根据牌面加分
// - 1/5/9万筒条: 加1番
// - 2/6万筒条 + 中: 加2番
// - 3/7万筒条 + 发: 加3番
// - 4/8万筒条 + 白: 加4番
// - 东南西北: 加对应番数（东1、南2、西3、北4）

// 计算某张牌的翻屁股价值（番数）
int calc_flip_value(int hai);

// 计算手牌的翻屁股期望值
// 参数：
//   tehai: 手牌
//   is_tenpai: 是否已经听牌
// 返回：翻屁股期望值（分数）
float calc_flip_expected_value(const Hai_Array& tehai, bool is_tenpai);

// 杠头（杠后胡牌加倍）规则：
// 杠后胡牌翻倍

// 计算杠头期望值
// 参数：
//   tehai: 手牌
//   has_kan: 是否已经有杠
//   is_tenpai: 是否已经听牌
// 返回：杠头期望值（分数）
float calc_kan_bonus_value(const Hai_Array& tehai, bool has_kan, bool is_tenpai);

// 计算手牌中可以杠的牌数量
int count_kan_potential(const Hai_Array& tehai);

// 综合期望值计算
// 整合抓冲、翻屁股、杠头的综合期望值
struct BonusValue {
    float grab_charge_value;  // 抓冲价值
    float flip_value;         // 翻屁股价值
    float kan_bonus_value;    // 杠头价值
    float total_value;        // 总价值
};

// 计算综合期望值
BonusValue calc_total_bonus_value(
    const Hai_Array& tehai,
    int jikaze,
    bool has_kan,
    bool is_tenpai
);

} // namespace linhai

#endif
