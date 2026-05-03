#include "linhai_score.hpp"
#include "json11.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>

namespace linhai_score {

// ---------- 38-array 索引段（对应 engine/share/types.cpp::hai_str_to_int） ----------
// 1..9   : 1m..9m
// 10     : 5mr (akadora，临海不用)
// 11..19 : 1p..9p
// 20     : 5pr
// 21..29 : 1s..9s
// 30     : 5sr
// 31..34 : E S W N
// 35     : P (白)
// 36     : F (发)
// 37     : C (中)

static bool is_man(int hai)   { return hai >= 1  && hai <= 9; }
static bool is_pin(int hai)   { return hai >= 11 && hai <= 19; }
static bool is_sou(int hai)   { return hai >= 21 && hai <= 29; }
static bool is_honor(int hai) { return hai >= 31 && hai <= 37; }

// 临海有效 25 牌：9 万 + 9 条 + 7 字（筒子 11-19 在临海中不存在）。
// 与 linhai_game_adapter.hpp::LINHAI_VALID_TILES 保持一致。
static const std::array<int, 25> LINHAI_VALID_TILES = {
    1,2,3,4,5,6,7,8,9,            // 万子
    21,22,23,24,25,26,27,28,29,   // 条子
    31,32,33,34,35,36,37          // 字牌
};

ScoreTable load_score_table(const std::string& path) {
    ScoreTable t;
    std::ifstream f(path);
    if (!f.is_open()) return t;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string err;
    auto j = json11::Json::parse(ss.str(), err);
    if (!err.empty()) return t;

    auto get_int = [](const json11::Json& v, int fallback) {
        return v.is_number() ? v.int_value() : fallback;
    };

    t.chong_to_score          = get_int(j["chong_to_score"], t.chong_to_score);
    t.base_ordinary           = get_int(j["base_chong"]["ordinary"], t.base_ordinary);
    t.base_qingyise           = get_int(j["base_chong"]["qingyise"], t.base_qingyise);
    t.base_ziyise             = get_int(j["base_chong"]["ziyise"], t.base_ziyise);
    t.base_hunyise_tree       = get_int(j["base_chong"]["hunyise_with_tree"], t.base_hunyise_tree);
    t.base_hunyise_hard       = get_int(j["base_chong"]["hunyise_hard"], t.base_hunyise_hard);
    t.qianggang_bonus         = get_int(j["bonuses"]["qianggang_hu"], t.qianggang_bonus);
    t.grab_charge_per_tile    = get_int(j["bonuses"]["grab_charge_per_tile"], t.grab_charge_per_tile);
    t.max_base_chong          = get_int(j["cap"]["max_base_chong"], t.max_base_chong);
    if (j["cap"]["extra_chong_uncapped"].is_bool()) {
        t.extra_chong_uncapped = j["cap"]["extra_chong_uncapped"].bool_value();
    }
    t.redhead_per_player_count_default = get_int(j["redhead"]["per_player_count_default"], t.redhead_per_player_count_default);
    t.contract_target_count_default    = get_int(j["contract"]["target_count_default"], t.contract_target_count_default);

    auto rh = j["redhead"]["tile_to_chong"];
    if (rh.is_object()) {
        for (const auto& kv : rh.object_items()) {
            int hai = hai_str_to_int(kv.first);
            if (hai > 0) t.redhead_tile_to_chong[hai] = kv.second.int_value();
        }
    } else {
        // 默认表（与 score_table.json 一致）
        for (int i = 1; i <= 9; ++i) t.redhead_tile_to_chong[i] = i;     // 1m..9m → +1..+9
        for (int i = 31; i <= 37; ++i) t.redhead_tile_to_chong[i] = 5;   // E/S/W/N/P/F/C → +5
    }

    // === Phase A: 推理侧防守加固（spec §3.5）===
    auto get_float = [](const json11::Json& v, float fallback) {
        return v.is_number() ? static_cast<float>(v.number_value()) : fallback;
    };

    if (j["ev_risk_weights"].is_object()) {
        const auto& evr = j["ev_risk_weights"];
        t.ev_risk_weights.lambda_base           = get_float(evr["lambda_base"],           t.ev_risk_weights.lambda_base);
        t.ev_risk_weights.lambda_pressure_step  = get_float(evr["lambda_pressure_step"],  t.ev_risk_weights.lambda_pressure_step);
        t.ev_risk_weights.lambda_max            = get_float(evr["lambda_max"],            t.ev_risk_weights.lambda_max);
        t.ev_risk_weights.mu_base               = get_float(evr["mu_base"],               t.ev_risk_weights.mu_base);
        t.ev_risk_weights.mu_pressure_step      = get_float(evr["mu_pressure_step"],      t.ev_risk_weights.mu_pressure_step);
        t.ev_risk_weights.mu_max                = get_float(evr["mu_max"],                t.ev_risk_weights.mu_max);
        t.ev_risk_weights.houjuu_base_coeff     = get_float(evr["houjuu_base_coeff"],     t.ev_risk_weights.houjuu_base_coeff);
        t.ev_risk_weights.high_chong_threshold  = get_float(evr["high_chong_threshold"],  t.ev_risk_weights.high_chong_threshold);
    }
    if (j["betaori_thresholds"].is_object()) {
        const auto& bt = j["betaori_thresholds"];
        t.betaori_thresholds.base                = get_float(bt["base"],                t.betaori_thresholds.base);
        t.betaori_thresholds.meld_drop           = get_float(bt["meld_drop"],           t.betaori_thresholds.meld_drop);
        t.betaori_thresholds.qingyise_drop       = get_float(bt["qingyise_drop"],       t.betaori_thresholds.qingyise_drop);
        t.betaori_thresholds.ev_loss_multiplier  = get_float(bt["ev_loss_multiplier"],  t.betaori_thresholds.ev_loss_multiplier);
    }
    if (j["feature_weights"].is_object()) {
        const auto& fw = j["feature_weights"];
        t.feature_weights.opp_meld_pressure_alpha    = get_float(fw["opp_meld_pressure_alpha"],    t.feature_weights.opp_meld_pressure_alpha);
        t.feature_weights.opp_qingyise_alarm_alpha   = get_float(fw["opp_qingyise_alarm_alpha"],   t.feature_weights.opp_qingyise_alarm_alpha);
    }

    return t;
}

HuType classify_hu(const Hai_Array& tehai, const Fuuro_Vector& fuuro) {
    Hai_Array all = tehai;
    for (const auto& m : fuuro) {
        if (m.hai > 0 && m.hai < (int)all.size()) all[m.hai]++;
        for (int t : m.consumed) {
            if (t > 0 && t < (int)all.size()) all[t]++;
        }
    }
    bool has_man = false, has_pin = false, has_sou = false, has_honor = false;
    for (int i = 1; i <= 9; ++i)   if (all[i] > 0) has_man = true;
    for (int i = 11; i <= 19; ++i) if (all[i] > 0) has_pin = true;
    for (int i = 21; i <= 29; ++i) if (all[i] > 0) has_sou = true;
    for (int i = 31; i <= 37; ++i) if (all[i] > 0) has_honor = true;
    int suit_count = (has_man ? 1 : 0) + (has_pin ? 1 : 0) + (has_sou ? 1 : 0);

    if (suit_count == 0 && has_honor) return HuType::ZiYise;
    if (suit_count == 1 && !has_honor) return HuType::QingYise;
    if (suit_count == 1 && has_honor)  return HuType::HunYise;
    return HuType::Ordinary;
}

static int calc_redhead_bonus(const RedHeadCatch& rh, const ScoreTable& table) {
    int sum = 0;
    for (int hai : rh.caught_tiles) {
        auto it = table.redhead_tile_to_chong.find(hai);
        if (it != table.redhead_tile_to_chong.end()) sum += it->second;
    }
    return sum;
}

ChongBreakdown calc_chong(const ChongInputs& in, const ScoreTable& table) {
    ChongBreakdown b;
    b.hu_type = classify_hu(in.tehai, in.melds);

    int total_white = in.white_count_in_hand + in.white_count_in_melds;

    // 1. 基础冲
    switch (b.hu_type) {
    case HuType::QingYise:
        b.base_chong = table.base_qingyise;
        break;
    case HuType::ZiYise:
        b.base_chong = table.base_ziyise;
        break;
    case HuType::HunYise:
        // 有树（白板）→ 2 冲；硬碰硬（无白板）→ 4 冲
        b.base_chong = (total_white > 0) ? table.base_hunyise_tree : table.base_hunyise_hard;
        break;
    case HuType::Ordinary:
    default:
        b.base_chong = table.base_ordinary;
    }

    // 2. 硬碰硬（无白板时翻倍）
    // 注：混一色已经在 base_chong 区分 with_tree/hard，避免重复翻倍
    bool hard = (total_white == 0);
    if (hard && b.hu_type != HuType::HunYise) {
        b.multiplier_2x++;
    }

    // 3. 树掉还原（白板暗刻 = 3 张全在手且 has_tree_active）
    if (in.has_tree_active && in.white_count_in_hand >= 3) {
        b.multiplier_4x = 1;
    }

    // 4. 计算应用倍数后的 base
    int mult = 1;
    for (int i = 0; i < b.multiplier_2x; ++i) mult *= 2;
    if (b.multiplier_4x) mult *= 4;
    b.capped_base = std::min(b.base_chong * mult, table.max_base_chong);

    // 5. 抢杠胡
    if (in.is_qiang_gang) b.qianggang_bonus = table.qianggang_bonus;

    // 6. 抓冲（不计入封顶）
    b.extra_chong = in.grab_charge_caught_count * table.grab_charge_per_tile;

    // 7. 翻屁股
    b.redhead_bonus = calc_redhead_bonus(in.redhead, table);

    // 8. 最终冲数
    b.final_chong = b.capped_base + b.extra_chong + b.qianggang_bonus + b.redhead_bonus;

    // 9. 三键承包（仅自摸触发，按 split_ratio 平摊）
    if (in.is_tsumo && in.contract.active && in.contract.counter >= in.contract.target_count) {
        b.contract_split = b.final_chong / 2;  // 每家承担一半
    }

    // 10. 转换为分数
    b.score = b.final_chong * table.chong_to_score;

    std::ostringstream os;
    os << "type=" << (int)b.hu_type
       << " base=" << b.base_chong
       << " mult=" << mult
       << " capped=" << b.capped_base
       << " extra=" << b.extra_chong
       << " qianggang=" << b.qianggang_bonus
       << " redhead=" << b.redhead_bonus
       << " contract_split=" << b.contract_split
       << " final=" << b.final_chong
       << " score=" << b.score;
    b.explain = os.str();

    return b;
}

} // namespace linhai_score
