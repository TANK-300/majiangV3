#ifndef LINHAI_SHANTEN_V4_HPP
#define LINHAI_SHANTEN_V4_HPP

#include "types.hpp"

// 标准向听数计算（不考虑白板万能）
int calc_standard_shanten(const Hai_Array& tehai);

// 临海麻将的向听数计算（支持白板万能）
int calc_linhai_shanten(const Hai_Array& tehai);

#endif // LINHAI_SHANTEN_V4_HPP
