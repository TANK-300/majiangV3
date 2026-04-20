#include "linhai_rules_v3.hpp"
#include "calc_shanten.hpp"
#include "calc_agari.hpp"

// 白板牌的编号
const int WHITE_TILE = 35;

// 标准向听数计算（不考虑白板万能）
// 参考标准算法实现
void calc_shanten_recursive(
    const Hai_Array& tehai,
    int start,
    int mentu_num,      // 面子数
    int tatsu_num,      // 搭子数
    bool has_jantou,    // 是否有雀头
    int& min_shanten
) {
    // 找到第一张有牌的位置
    int hai = start;
    while (hai < 38 && tehai[hai] == 0) {
        hai++;
    }

    if (hai >= 38) {
        // 没有牌了，计算向听数
        // 向听数 = 8 - 2*面子数 - 搭子数 - (有雀头?1:0)
        int shanten = 8 - mentu_num * 2 - tatsu_num - (has_jantou ? 1 : 0);

        // 调试输出
        std::cout << "面子数=" << mentu_num << ", 搭子数=" << tatsu_num
                  << ", 雀头=" << (has_jantou ? "有" : "无")
                  << ", 向听数=" << shanten << std::endl;

        if (shanten < min_shanten) {
            min_shanten = shanten;
        }
        return;
    }

    Hai_Array tmp = tehai;

    // 如果面子数已经达到4个，不再切割面子和搭子
    if (mentu_num >= 4) {
        // 只能跳过剩余的牌
        tmp[hai]--;
        calc_shanten_recursive(tmp, hai, mentu_num, tatsu_num, has_jantou, min_shanten);
        tmp[hai]++;
        return;
    }

    // 尝试1: 切割刻子
    if (tmp[hai] >= 3) {
        tmp[hai] -= 3;
        calc_shanten_recursive(tmp, hai, mentu_num + 1, tatsu_num, has_jantou, min_shanten);
        tmp[hai] += 3;
    }

    // 尝试2: 切割顺子（只对数牌）
    if (hai < 30 && hai % 10 <= 7) {
        if (tmp[hai] >= 1 && tmp[hai+1] >= 1 && tmp[hai+2] >= 1) {
            tmp[hai]--;
            tmp[hai+1]--;
            tmp[hai+2]--;
            calc_shanten_recursive(tmp, hai, mentu_num + 1, tatsu_num, has_jantou, min_shanten);
            tmp[hai]++;
            tmp[hai+1]++;
            tmp[hai+2]++;
        }
    }

    // 如果面子+搭子数未达到4个，可以切割搭子
    // 注意：如果面子数已经是4个，就不应该再切割搭子了
    if (mentu_num + tatsu_num < 4 && mentu_num < 4) {
        // 尝试3: 切割对子搭子
        if (tmp[hai] >= 2) {
            tmp[hai] -= 2;
            calc_shanten_recursive(tmp, hai, mentu_num, tatsu_num + 1, has_jantou, min_shanten);
            tmp[hai] += 2;
        }

        // 尝试4: 切割两面/嵌张搭子（只对数牌）
        if (hai < 30) {
            // 两面搭子
            if (hai % 10 <= 8 && tmp[hai] >= 1 && tmp[hai+1] >= 1) {
                tmp[hai]--;
                tmp[hai+1]--;
                calc_shanten_recursive(tmp, hai, mentu_num, tatsu_num + 1, has_jantou, min_shanten);
                tmp[hai]++;
                tmp[hai+1]++;
            }
            // 嵌张搭子
            if (hai % 10 <= 7 && tmp[hai] >= 1 && tmp[hai+2] >= 1) {
                tmp[hai]--;
                tmp[hai+2]--;
                calc_shanten_recursive(tmp, hai, mentu_num, tatsu_num + 1, has_jantou, min_shanten);
                tmp[hai]++;
                tmp[hai+2]++;
            }
        }
    }

    // 尝试5: 切割雀头（如果还没有雀头）
    if (!has_jantou && tmp[hai] >= 2) {
        tmp[hai] -= 2;
        calc_shanten_recursive(tmp, hai, mentu_num, tatsu_num, true, min_shanten);
        tmp[hai] += 2;
    }

    // 尝试6: 跳过这张牌（孤张）
    tmp[hai]--;
    calc_shanten_recursive(tmp, hai, mentu_num, tatsu_num, has_jantou, min_shanten);
    tmp[hai]++;
}

int calc_standard_shanten(const Hai_Array& tehai) {
    int min_shanten = 8;
    calc_shanten_recursive(tehai, 1, 0, 0, false, min_shanten);
    return min_shanten;
}

// 临海麻将的向听数计算（支持白板万能）
Tenpai_Info linhai_cal_tenpai_info(const int bakaze, const int jikaze, const Hai_Array& tehai, const Fuuro_Vector& fuuro) {
    Tenpai_Info info;

    // 统计白板数量
    int white_count = tehai[WHITE_TILE];

    if (white_count == 0) {
        // 没有白板，使用原始算法
        return cal_tenpai_info(bakaze, jikaze, tehai, fuuro);
    }

    // 有白板，枚举所有可能的替代方案
    int min_shanten = 8;

    // 方案1: 白板不替代任何牌（作为普通牌使用）
    {
        int shanten = calc_standard_shanten(tehai);
        if (shanten < min_shanten) {
            min_shanten = shanten;
        }
    }

    // 方案2-N: 白板替代其他牌
    // 枚举所有可能的牌型（1-37，跳过0和10的倍数）
    for (int replace_hai = 1; replace_hai < 38; replace_hai++) {
        if (replace_hai % 10 == 0) continue;
        if (replace_hai == WHITE_TILE) continue;

        // 尝试用1-white_count张白板替代这张牌
        for (int use_white = 1; use_white <= white_count && use_white <= 4 - tehai[replace_hai]; use_white++) {
            Hai_Array tmp = tehai;
            tmp[WHITE_TILE] -= use_white;
            tmp[replace_hai] += use_white;

            int shanten = calc_standard_shanten(tmp);
            if (shanten < min_shanten) {
                min_shanten = shanten;
            }
        }
    }

    info.mentu_shanten_num = min_shanten;

    // 如果是听牌，调用原始的听牌检查获取详细信息
    if (min_shanten == 0) {
        // TODO: 实现白板万能的听牌检查
        // 暂时使用原始算法
        Tenpai_Info detail_info = cal_tenpai_info(bakaze, jikaze, tehai, fuuro);
        if (detail_info.mentu_shanten_num == 0) {
            return detail_info;
        }
    }

    return info;
}

// 抓冲牌判断
bool is_grab_charge_tile(int hai, int jikaze) {
    // jikaze: 0=东, 1=南, 2=西, 3=北
    switch (jikaze) {
        case 0: // 东风家
            return hai == 31 || hai == 1 || hai == 5 || hai == 9;
        case 1: // 南风家
            return hai == 32 || hai == 2 || hai == 6 || hai == 36; // 南、2万、6万、中
        case 2: // 西风家
            return hai == 33 || hai == 3 || hai == 7 || hai == 35; // 西、3万、7万、发
        case 3: // 北风家
            return hai == 34 || hai == 4 || hai == 8 || hai == 37; // 北、4万、8万、白
        default:
            return false;
    }
}

// 获取抓冲牌列表
std::vector<int> get_grab_charge_tiles(int jikaze) {
    std::vector<int> tiles;
    switch (jikaze) {
        case 0: // 东风家
            tiles.push_back(31); // 东
            tiles.push_back(1);  // 1万
            tiles.push_back(5);  // 5万
            tiles.push_back(9);  // 9万
            break;
        case 1: // 南风家
            tiles.push_back(32); // 南
            tiles.push_back(2);  // 2万
            tiles.push_back(6);  // 6万
            tiles.push_back(36); // 中
            break;
        case 2: // 西风家
            tiles.push_back(33); // 西
            tiles.push_back(3);  // 3万
            tiles.push_back(7);  // 7万
            tiles.push_back(35); // 发
            break;
        case 3: // 北风家
            tiles.push_back(34); // 北
            tiles.push_back(4);  // 4万
            tiles.push_back(8);  // 8万
            tiles.push_back(37); // 白
            break;
    }
    return tiles;
}

// 计算抓冲期望值
float calc_grab_charge_expected_value(const Hai_Array& tehai, int jikaze) {
    std::vector<int> grab_tiles = get_grab_charge_tiles(jikaze);
    int count = 0;
    for (size_t i = 0; i < grab_tiles.size(); i++) {
        count += tehai[grab_tiles[i]];
    }

    // 基础期望值：每张抓冲牌价值50分
    // 如果有多张，额外加成
    float value = count * 50.0f;
    if (count >= 2) {
        value += count * 20.0f; // 多张加成
    }

    return value;
}
