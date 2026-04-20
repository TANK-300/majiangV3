#ifndef LINHAI_GRAB_CHARGE_HPP
#define LINHAI_GRAB_CHARGE_HPP

#include "types.hpp"
#include <vector>

namespace linhai {

// 抓冲牌规则：
// 东风家: 东风 + 1/5/9万条筒 (共7张)
// 南风家: 南风 + 2/6万条筒 + 中 (共7张)
// 西风家: 西风 + 3/7万条筒 + 发 (共7张)
// 北风家: 北风 + 4/8万条筒 + 白板 (共7张)

// 获取指定风位的抓冲牌列表
std::vector<int> get_grab_charge_tiles(int jikaze);

// 计算手牌中抓冲牌的数量
int count_grab_charge_tiles(const Hai_Array& tehai, int jikaze);

// 计算抓冲期望值
// 参数：
//   tehai: 手牌
//   jikaze: 自风（0=东, 1=南, 2=西, 3=北）
//   is_tenpai: 是否已经听牌
// 返回：抓冲期望值（分数）
float calc_grab_charge_value(const Hai_Array& tehai, int jikaze, bool is_tenpai);

// 判断某张牌是否是抓冲牌
bool is_grab_charge_tile(int hai, int jikaze);

// 计算打出某张牌后损失的抓冲价值
float calc_grab_charge_loss(int discard_hai, const Hai_Array& tehai, int jikaze, bool is_tenpai);

} // namespace linhai

#endif
