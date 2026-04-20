#include "linhai_grab_charge.hpp"
#include <algorithm>

namespace linhai {

// 牌编号定义（与akochan一致）
// 万子(m): 1-9
// 筒子(p): 11-19
// 条子(s): 21-29
// 东南西北白发中(E/S/W/N/P/F/C): 31-37

static const int DONG = 31;   // 东 E
static const int NAN_HAI = 32;    // 南 S (避免与math.h的NAN宏冲突)
static const int XI = 33;     // 西 W
static const int BEI = 34;    // 北 N
static const int BAI = 35;    // 白 P
static const int FA = 36;     // 发 F
static const int ZHONG = 37;  // 中 C

std::vector<int> get_grab_charge_tiles(int jikaze) {
    std::vector<int> tiles;

    switch (jikaze) {
        case 0: // 东风家
            tiles.push_back(DONG);  // 东风
            // 1万筒条
            tiles.push_back(1);   // 1万
            tiles.push_back(11);  // 1筒
            tiles.push_back(21);  // 1条
            // 5万筒条
            tiles.push_back(5);   // 5万
            tiles.push_back(15);  // 5筒
            tiles.push_back(25);  // 5条
            // 9万筒条
            tiles.push_back(9);   // 9万
            tiles.push_back(19);  // 9筒
            tiles.push_back(29);  // 9条
            break;

        case 1: // 南风家
            tiles.push_back(NAN_HAI);   // 南风
            // 2万筒条
            tiles.push_back(2);   // 2万
            tiles.push_back(12);  // 2筒
            tiles.push_back(22);  // 2条
            // 6万筒条
            tiles.push_back(6);   // 6万
            tiles.push_back(16);  // 6筒
            tiles.push_back(26);  // 6条
            // 中
            tiles.push_back(ZHONG);  // 中
            break;

        case 2: // 西风家
            tiles.push_back(XI);    // 西风
            // 3万筒条
            tiles.push_back(3);   // 3万
            tiles.push_back(13);  // 3筒
            tiles.push_back(23);  // 3条
            // 7万筒条
            tiles.push_back(7);   // 7万
            tiles.push_back(17);  // 7筒
            tiles.push_back(27);  // 7条
            // 发
            tiles.push_back(FA);  // 发
            break;

        case 3: // 北风家
            tiles.push_back(BEI);   // 北风
            // 4万筒条
            tiles.push_back(4);   // 4万
            tiles.push_back(14);  // 4筒
            tiles.push_back(24);  // 4条
            // 8万筒条
            tiles.push_back(8);   // 8万
            tiles.push_back(18);  // 8筒
            tiles.push_back(28);  // 8条
            // 白板
            tiles.push_back(BAI);  // 白板
            break;
    }

    return tiles;
}

bool is_grab_charge_tile(int hai, int jikaze) {
    std::vector<int> tiles = get_grab_charge_tiles(jikaze);
    return std::find(tiles.begin(), tiles.end(), hai) != tiles.end();
}

int count_grab_charge_tiles(const Hai_Array& tehai, int jikaze) {
    std::vector<int> tiles = get_grab_charge_tiles(jikaze);
    int count = 0;

    for (int hai : tiles) {
        if (hai < 48) {  // 确保索引有效
            count += tehai[hai];
        }
    }

    return count;
}

float calc_grab_charge_value(const Hai_Array& tehai, int jikaze, bool is_tenpai) {
    if (!is_tenpai) {
        // 未听牌时，抓冲牌价值较低
        return 0.0f;
    }

    int grab_charge_count = count_grab_charge_tiles(tehai, jikaze);

    // 抓冲规则：胡牌后抓6张特定牌，每张翻倍
    // 假设：
    // - 基础分数为1000分
    // - 每张抓冲牌有1/6概率被抓到（6张中抓1张）
    // - 每张抓冲牌翻倍（×2）
    // - 期望值 = 基础分 × 抓冲牌数 × (1/6) × 2

    const float BASE_SCORE = 1000.0f;
    const float GRAB_PROBABILITY = 1.0f / 6.0f;  // 抓6张，每张概率1/6
    const float MULTIPLIER = 2.0f;  // 翻倍

    float expected_value = BASE_SCORE * grab_charge_count * GRAB_PROBABILITY * MULTIPLIER;

    return expected_value;
}

float calc_grab_charge_loss(int discard_hai, const Hai_Array& tehai, int jikaze, bool is_tenpai) {
    if (!is_grab_charge_tile(discard_hai, jikaze)) {
        return 0.0f;  // 不是抓冲牌，无损失
    }

    if (!is_tenpai) {
        // 未听牌时，损失较小
        return 50.0f;  // 固定小损失
    }

    // 听牌时，损失一张抓冲牌的期望值
    const float BASE_SCORE = 1000.0f;
    const float GRAB_PROBABILITY = 1.0f / 6.0f;
    const float MULTIPLIER = 2.0f;

    float loss = BASE_SCORE * GRAB_PROBABILITY * MULTIPLIER;

    // 如果手牌中有多张同样的抓冲牌，损失减少
    int count = tehai[discard_hai];
    if (count > 1) {
        loss *= 0.5f;  // 还有其他张，损失减半
    }

    return loss;
}

} // namespace linhai
