#include "linhai_bonus.hpp"
#include "linhai_grab_charge.hpp"

namespace linhai {

// 牌编号定义
// 万子(m): 1-9
// 筒子(p): 11-19
// 条子(s): 21-29
// 东南西北白发中(E/S/W/N/P/F/C): 31-37

int calc_flip_value(int hai) {
    // 1/5/9万筒条: 加1番
    if (hai == 1 || hai == 5 || hai == 9 ||
        hai == 11 || hai == 15 || hai == 19 ||
        hai == 21 || hai == 25 || hai == 29) {
        return 1;
    }

    // 2/6万筒条: 加2番
    if (hai == 2 || hai == 6 ||
        hai == 12 || hai == 16 ||
        hai == 22 || hai == 26) {
        return 2;
    }

    // 3/7万筒条: 加3番
    if (hai == 3 || hai == 7 ||
        hai == 13 || hai == 17 ||
        hai == 23 || hai == 27) {
        return 3;
    }

    // 4/8万筒条: 加4番
    if (hai == 4 || hai == 8 ||
        hai == 14 || hai == 18 ||
        hai == 24 || hai == 28) {
        return 4;
    }

    // 字牌
    if (hai == 31) return 1;  // 东
    if (hai == 32) return 2;  // 南
    if (hai == 33) return 3;  // 西
    if (hai == 34) return 4;  // 北
    if (hai == 35) return 4;  // 白
    if (hai == 36) return 3;  // 发
    if (hai == 37) return 2;  // 中

    return 0;
}

float calc_flip_expected_value(const Hai_Array& tehai, bool is_tenpai) {
    if (!is_tenpai) {
        // 未听牌时，翻屁股价值较低
        return 0.0f;
    }

    // 计算手牌中所有牌的翻屁股价值���和
    int total_fan = 0;
    int total_tiles = 0;

    for (int hai = 1; hai < 38; hai++) {
        if (tehai[hai] > 0) {
            int fan = calc_flip_value(hai);
            total_fan += fan * tehai[hai];
            total_tiles += tehai[hai];
        }
    }

    if (total_tiles == 0) {
        return 0.0f;
    }

    // 平均每张牌的番数
    float avg_fan = static_cast<float>(total_fan) / total_tiles;

    // 翻屁股规则：胡牌后抓码，假设抓1张码
    // 期望值 = 基础分 × 平均番数
    const float BASE_SCORE = 1000.0f;
    const float FAN_MULTIPLIER = 1.5f;  // 每番增加50%

    float expected_value = BASE_SCORE * avg_fan * FAN_MULTIPLIER;

    return expected_value;
}

int count_kan_potential(const Hai_Array& tehai) {
    int kan_count = 0;

    for (int hai = 1; hai < 38; hai++) {
        if (tehai[hai] >= 4) {
            kan_count++;  // 暗杠
        } else if (tehai[hai] == 3) {
            kan_count++;  // 可能的明杠（如果别人打出）
        }
    }

    return kan_count;
}

float calc_kan_bonus_value(const Hai_Array& tehai, bool has_kan, bool is_tenpai) {
    if (!is_tenpai) {
        // 未听牌时，杠头价值较低
        return 0.0f;
    }

    if (has_kan) {
        // 已经有杠，杠后胡牌翻倍
        const float BASE_SCORE = 1000.0f;
        const float KAN_MULTIPLIER = 2.0f;  // 翻倍
        return BASE_SCORE * (KAN_MULTIPLIER - 1.0f);  // 额外收益
    }

    // 评估杠的潜力
    int kan_potential = count_kan_potential(tehai);

    if (kan_potential == 0) {
        return 0.0f;
    }

    // 有杠的潜力，但还没杠
    // 期望值 = 基础分 × 杠的概率 × 翻倍收益
    const float BASE_SCORE = 1000.0f;
    const float KAN_PROBABILITY = 0.3f;  // 假设30%概率能杠
    const float KAN_MULTIPLIER = 2.0f;

    float expected_value = BASE_SCORE * KAN_PROBABILITY * (KAN_MULTIPLIER - 1.0f) * kan_potential;

    return expected_value;
}

BonusValue calc_total_bonus_value(
    const Hai_Array& tehai,
    int jikaze,
    bool has_kan,
    bool is_tenpai
) {
    BonusValue result;

    // 计算抓冲价值
    result.grab_charge_value = calc_grab_charge_value(tehai, jikaze, is_tenpai);

    // 计算翻屁股价值
    result.flip_value = calc_flip_expected_value(tehai, is_tenpai);

    // 计算杠头价值
    result.kan_bonus_value = calc_kan_bonus_value(tehai, has_kan, is_tenpai);

    // 总价值
    result.total_value = result.grab_charge_value + result.flip_value + result.kan_bonus_value;

    return result;
}

} // namespace linhai
