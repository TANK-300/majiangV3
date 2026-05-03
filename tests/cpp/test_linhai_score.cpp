// 番数引擎单元测试。覆盖 spec §3.2 番数表全部条款：
//   普通胡 / 硬碰硬 / 树掉还原 / 混一色 (有树/硬碰硬) / 清一色 / 字一色
//   抢杠胡 / 抓冲 / 翻屁股（一万..九万 / 风字 +5）/ 三键承包

#include "share/linhai_score.hpp"
#include "share/types.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <utility>

using namespace linhai_score;

static Hai_Array make_hai(std::initializer_list<std::pair<int,int>> kv) {
    Hai_Array a{};
    for (auto& p : kv) a[p.first] = p.second;
    return a;
}

// 1. 普通胡：白板 1 张（非硬碰硬），无番数加成
static void test_ordinary_hu_with_white() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1},{2,1}});
    in.white_count_in_hand = 1;
    auto b = calc_chong(in, t);
    assert(b.hu_type == HuType::Ordinary);
    assert(b.base_chong == 1);
    assert(b.multiplier_2x == 0);
    assert(b.capped_base == 1);
    assert(b.final_chong == 1);
    assert(b.score == 1);
    std::printf("test_ordinary_hu_with_white PASS\n");
}

// 2. 硬碰硬普通胡：无白板 → ×2
static void test_ordinary_hard() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1},{2,1},{3,1}});
    in.white_count_in_hand = 0;
    auto b = calc_chong(in, t);
    assert(b.hu_type == HuType::Ordinary);
    assert(b.base_chong == 1);
    assert(b.multiplier_2x == 1);
    assert(b.capped_base == 2);
    assert(b.final_chong == 2);
    std::printf("test_ordinary_hard PASS\n");
}

// 3. 清一色：全万 + 无字
static void test_qingyise() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1},{2,1},{3,1},{4,1},{5,1},{6,1},{7,1},{8,1},{9,1}});
    in.white_count_in_hand = 0;  // 硬碰硬
    auto b = calc_chong(in, t);
    assert(b.hu_type == HuType::QingYise);
    assert(b.base_chong == 8);
    // 硬碰硬层叠加，但 base 已封顶 8
    assert(b.capped_base == 8);
    assert(b.final_chong == 8);
    std::printf("test_qingyise PASS\n");
}

// 4. 字一色：全字
static void test_ziyise() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{31,1},{32,1},{33,1},{34,1},{35,1},{36,1},{37,1}});
    in.white_count_in_hand = 1;
    auto b = calc_chong(in, t);
    assert(b.hu_type == HuType::ZiYise);
    assert(b.base_chong == 8);
    assert(b.capped_base == 8);
    std::printf("test_ziyise PASS\n");
}

// 5. 混一色（有树掉，即 white_count > 0）：base = 2
static void test_hunyise_with_tree() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1},{2,1},{3,1},{31,1},{35,1}});  // 万 + 东 + 白板
    in.white_count_in_hand = 1;
    auto b = calc_chong(in, t);
    assert(b.hu_type == HuType::HunYise);
    assert(b.base_chong == 2);
    assert(b.multiplier_2x == 0);  // 混一色不重复翻倍
    assert(b.capped_base == 2);
    std::printf("test_hunyise_with_tree PASS\n");
}

// 6. 混一色（硬碰硬，无白板）：base = 4
static void test_hunyise_hard() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1},{2,1},{3,1},{31,1}});  // 万 + 东，无白板
    in.white_count_in_hand = 0;
    auto b = calc_chong(in, t);
    assert(b.hu_type == HuType::HunYise);
    assert(b.base_chong == 4);
    assert(b.multiplier_2x == 0);  // 混一色不重复翻倍
    assert(b.capped_base == 4);
    std::printf("test_hunyise_hard PASS\n");
}

// 7. 抢杠胡：普通胡 + qianggang → +2 冲
static void test_qianggang() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1}});
    in.is_qiang_gang = true;
    in.white_count_in_hand = 1;  // 非硬碰硬
    auto b = calc_chong(in, t);
    assert(b.qianggang_bonus == 2);
    assert(b.final_chong == 1 + 2);
    std::printf("test_qianggang PASS\n");
}

// 8. 树掉还原：白板暗刻 + has_tree_active → ×4
static void test_tree_restore() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1},{35,3}});  // 白板 ×3 暗刻
    in.has_tree_active = true;
    in.white_count_in_hand = 3;
    auto b = calc_chong(in, t);
    assert(b.multiplier_4x == 1);
    // base 1 * 4 = 4，封顶 8 内
    assert(b.capped_base == 4);
    std::printf("test_tree_restore PASS\n");
}

// 9. 抓冲：清一色 + 抓 2 张红头 → 8 + 2 = 10（封顶只对基础冲，extra 不计入）
static void test_qingyise_with_grab_charge() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1},{2,1},{3,1},{4,1},{5,1},{6,1},{7,1},{8,1},{9,1}});
    in.grab_charge_caught_count = 2;
    auto b = calc_chong(in, t);
    assert(b.extra_chong == 2);
    assert(b.final_chong == 8 + 2);  // base 8 capped + extra 2
    std::printf("test_qingyise_with_grab_charge PASS\n");
}

// 10. 翻屁股：抓到 1m → +1 冲
static void test_redhead_yiwan() {
    ScoreTable t;
    // 默认 redhead 表为空（load_score_table 在文件不存在时回退默认）；
    // 但 ScoreTable() 默认构造没填表，手动塞默认值（与 score_table.json 一致）
    for (int i = 1; i <= 9; ++i) t.redhead_tile_to_chong[i] = i;
    for (int i = 31; i <= 37; ++i) t.redhead_tile_to_chong[i] = 5;

    ChongInputs in;
    in.tehai = make_hai({{1,1}});
    in.white_count_in_hand = 1;
    in.redhead.caught_tiles = {1};  // 1m
    auto b = calc_chong(in, t);
    assert(b.redhead_bonus == 1);
    assert(b.final_chong == 1 + 1);
    std::printf("test_redhead_yiwan PASS\n");
}

// 11. 翻屁股：抓到 9m → +9 冲
static void test_redhead_jiuwan() {
    ScoreTable t;
    for (int i = 1; i <= 9; ++i) t.redhead_tile_to_chong[i] = i;
    for (int i = 31; i <= 37; ++i) t.redhead_tile_to_chong[i] = 5;

    ChongInputs in;
    in.tehai = make_hai({{1,1}});
    in.white_count_in_hand = 1;
    in.redhead.caught_tiles = {9};  // 9m
    auto b = calc_chong(in, t);
    assert(b.redhead_bonus == 9);
    std::printf("test_redhead_jiuwan PASS\n");
}

// 12. 翻屁股：抓到中(C, hai=37) → +5 冲
static void test_redhead_zhong() {
    ScoreTable t;
    for (int i = 1; i <= 9; ++i) t.redhead_tile_to_chong[i] = i;
    for (int i = 31; i <= 37; ++i) t.redhead_tile_to_chong[i] = 5;

    ChongInputs in;
    in.tehai = make_hai({{1,1}});
    in.white_count_in_hand = 1;
    in.redhead.caught_tiles = {37};  // C = 中
    auto b = calc_chong(in, t);
    assert(b.redhead_bonus == 5);
    std::printf("test_redhead_zhong PASS\n");
}

// 13. 翻屁股累加：抓到 1m + 9m + 东 → 1 + 9 + 5 = 15
static void test_redhead_multi() {
    ScoreTable t;
    for (int i = 1; i <= 9; ++i) t.redhead_tile_to_chong[i] = i;
    for (int i = 31; i <= 37; ++i) t.redhead_tile_to_chong[i] = 5;

    ChongInputs in;
    in.tehai = make_hai({{1,1}});
    in.white_count_in_hand = 1;
    in.redhead.caught_tiles = {1, 9, 31};  // 1m + 9m + E
    auto b = calc_chong(in, t);
    assert(b.redhead_bonus == 1 + 9 + 5);
    std::printf("test_redhead_multi PASS\n");
}

// 14. 三键承包：自摸 + active + counter >= target → contract_split = final_chong / 2
static void test_contract_split() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1},{2,1},{3,1}});
    in.white_count_in_hand = 0;  // 硬碰硬 ×2
    in.is_tsumo = true;
    in.contract.active = true;
    in.contract.counter = 3;
    in.contract.target_count = 3;
    auto b = calc_chong(in, t);
    // base 1 * 2 = 2，capped 2，final 2 → split 1
    assert(b.contract_split == 1);
    std::printf("test_contract_split PASS\n");
}

// 15. 三键承包：counter 不到 target → 不触发
static void test_contract_inactive_below_target() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1}});
    in.white_count_in_hand = 1;
    in.is_tsumo = true;
    in.contract.active = true;
    in.contract.counter = 2;          // 不到 3
    in.contract.target_count = 3;
    auto b = calc_chong(in, t);
    assert(b.contract_split == 0);
    std::printf("test_contract_inactive_below_target PASS\n");
}

// 16. 树掉还原 + 清一色 + 抢杠：组合复杂场景
static void test_combination_qingyise_tree_qianggang() {
    ScoreTable t;
    ChongInputs in;
    in.tehai = make_hai({{1,1},{2,1},{3,1},{4,1},{5,1},{6,1},{7,1},{8,1},{9,1}});
    in.has_tree_active = false;  // 清一色不带白板，不触发 tree restore
    in.is_qiang_gang = true;
    auto b = calc_chong(in, t);
    assert(b.hu_type == HuType::QingYise);
    assert(b.qianggang_bonus == 2);
    assert(b.final_chong == 8 + 2);  // 清一色封顶 8 + 抢杠 2
    std::printf("test_combination_qingyise_tree_qianggang PASS\n");
}

// 17. score = final_chong * chong_to_score（默认 1:1）
static void test_score_equals_chong_default() {
    ScoreTable t;  // chong_to_score = 1
    ChongInputs in;
    in.tehai = make_hai({{1,1}});
    in.white_count_in_hand = 1;
    auto b = calc_chong(in, t);
    assert(b.score == b.final_chong);
    std::printf("test_score_equals_chong_default PASS\n");
}

// 18. score 比例：chong_to_score = 2
static void test_score_with_custom_ratio() {
    ScoreTable t;
    t.chong_to_score = 2;
    ChongInputs in;
    in.tehai = make_hai({{1,1}});
    in.white_count_in_hand = 1;
    auto b = calc_chong(in, t);
    assert(b.score == b.final_chong * 2);
    std::printf("test_score_with_custom_ratio PASS\n");
}

int main() {
    test_ordinary_hu_with_white();
    test_ordinary_hard();
    test_qingyise();
    test_ziyise();
    test_hunyise_with_tree();
    test_hunyise_hard();
    test_qianggang();
    test_tree_restore();
    test_qingyise_with_grab_charge();
    test_redhead_yiwan();
    test_redhead_jiuwan();
    test_redhead_zhong();
    test_redhead_multi();
    test_contract_split();
    test_contract_inactive_below_target();
    test_combination_qingyise_tree_qianggang();
    test_score_equals_chong_default();
    test_score_with_custom_ratio();
    std::printf("\nAll 18 linhai_score tests PASS\n");
    return 0;
}
