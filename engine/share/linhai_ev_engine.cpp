#include "linhai_ev_engine.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <numeric>

namespace linhai {

// ============================================================================
// 工具函数
// ============================================================================

float LinHaiEVEngine::logistic(const float* w, const float* x, int dim) {
    float a = 0.0f;
    for (int i = 0; i < dim; i++) {
        a += w[i] * x[i];
    }
    return 1.0f / (1.0f + expf(-a));
}

static float my_logit(float x) {
    if (x >= 1.0f) return 10.0f;
    if (x <= 0.0f) return -10.0f;
    return logf(x / (1.0f - x));
}

static void read_params_file(float* w, int count, const std::string& path) {
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp) return;
    double tmp;
    for (int i = 0; i < count; i++) {
        if (fscanf(fp, "%lf", &tmp) == EOF) break;
        w[i] = static_cast<float>(tmp);
    }
    fclose(fp);
}

// ============================================================================
// 构造 & 参数加载
// ============================================================================

LinHaiEVEngine::LinHaiEVEngine() : config(), params_dir("params/"), can_win_(true) {}

LinHaiEVEngine::LinHaiEVEngine(const AIConfig& cfg) : config(cfg), params_dir("params/"), can_win_(true) {}

bool LinHaiEVEngine::load_params(const std::string& dir) {
    params_dir = dir;

    // 加载 agari_prob 参数
    for (int turn = 1; turn <= 17; turn++) {
        std::string path = dir + "agari_prob/linhai/agari_para" + std::to_string(turn) + ".txt";
        read_params_file(prob_params.agari_w[turn], 4, path);
    }

    // 加载 betaori 参数
    for (int turn = 2; turn <= 18; turn++) {
        std::string path = dir + "betaori/linhai/betaori_houjuu_para" + std::to_string(turn) + "_3000.txt";
        read_params_file(prob_params.betaori_w[turn - 1], 3, path);
    }

    // 加载 houjuu_prob 参数
    for (int turn = 6; turn <= 14; turn++) {
        std::string path = dir + "houjuu_prob/linhai/para" + std::to_string(turn) + "_4000.txt";
        read_params_file(prob_params.houjuu_w[turn - 1], 3, path);
    }

    // 加载 tsumo_num 参数
    for (int turn = 1; turn <= 16; turn++) {
        std::string path = dir + "tsumo_num/linhai/para" + std::to_string(turn) + "_4000.txt";
        read_params_file(prob_params.tsumo_num_w[turn - 1], 4, path);
    }

    // 加载 ryukyoku 参数
    for (int turn = 7; turn <= 18; turn++) {
        std::string path = dir + "ryukyoku_prob/linhai/para" + std::to_string(turn) + "_4000.txt";
        read_params_file(prob_params.ryukyoku_w[turn - 1], 4, path);
    }

    prob_params.loaded = true;
    return true;
}

// ============================================================================
// 概率计算
// ============================================================================

float LinHaiEVEngine::calc_solo_agari_prob(int shanten, int ukeire_count, int remaining_turns, int pool_size) const {
    if (shanten < 0) return 1.0f;
    if (remaining_turns <= 0 || pool_size <= 0 || ukeire_count <= 0) return 0.0f;

    // 每次摸牌命中有效牌的概率 (使用实际剩余牌数)
    float p = std::min(0.92f, (float)ukeire_count / (float)std::max(1, pool_size));

    if (shanten == 0) {
        // P(n巡内至少摸到一张和了牌) = 1-(1-p)^n  [几何分布]
        float not_win = 1.0f;
        for (int t = 0; t < std::min(remaining_turns, 25); t++) {
            not_win *= (1.0f - p);
        }
        return 1.0f - not_win;
    }

    if (shanten == 1) {
        // 阶段1: 前60%巡数内进听牌
        int turns_to_tenpai = std::max(1, (int)(remaining_turns * 0.6f));
        float not_improve = 1.0f;
        for (int t = 0; t < std::min(turns_to_tenpai, 15); t++) {
            not_improve *= (1.0f - p);
        }
        float p_reach_tenpai = 1.0f - not_improve;

        // 阶段2: 剩余巡数内和了 (保守估算 4-10 张受入)
        int est_waiters = std::min(ukeire_count / 2 + 2, 10);
        int turns_after = std::max(1, remaining_turns - turns_to_tenpai);
        float p_win_after = calc_solo_agari_prob(0, est_waiters, turns_after, pool_size);

        return p_reach_tenpai * p_win_after * 0.85f;
    }

    // shanten >= 2: 链式改善模型
    float expected_turns = (float)shanten / std::max(0.01f, p);
    if (expected_turns >= (float)remaining_turns * 1.5f) {
        return std::max(0.01f, 0.08f / (float)shanten);
    }
    float survival = 1.0f - expected_turns / (float)(remaining_turns + 1);
    return std::max(0.01f, std::min(0.50f, survival * 0.35f));
}

float LinHaiEVEngine::calc_agari_prob(float solo_prob, int turn, float opp_tenpai_prob) const {
    if (!prob_params.loaded) return solo_prob * 0.7f;

    int idx = std::max(1, std::min(turn, 15));
    float x[4] = {1.0f, 0.0f, opp_tenpai_prob, my_logit(solo_prob)};
    return logistic(prob_params.agari_w[idx], x, 4);
}

int LinHaiEVEngine::resolve_opponent_index(const GameState& state) const {
    if (state.all_players.empty()) return -1;

    int current = state.current_player;
    if (current < 0 || current >= static_cast<int>(state.all_players.size())) {
        current = state.jikaze;
    }
    if (current < 0 || current >= static_cast<int>(state.all_players.size())) {
        current = 0;
    }

    int best_idx = -1;
    int best_score = -1;
    for (int i = 0; i < static_cast<int>(state.all_players.size()); i++) {
        if (i == current) continue;
        const auto& candidate = state.all_players[i];
        int score = static_cast<int>(candidate.kawa.size()) * 4 +
                    static_cast<int>(candidate.fuuro.size()) * 3 +
                    (candidate.reach_declared ? 2 : 0);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx >= 0 && best_score > 0) {
        return best_idx;
    }

    if (static_cast<int>(state.all_players.size()) > 1) {
        return current == 0 ? 1 : 0;
    }
    return -1;
}

float LinHaiEVEngine::estimate_opponent_tenpai_prob(const GameState& state) const {
    // 基于对手信息估算听牌概率
    float prob = 0.1f;  // 基础概率

    int opp_idx = resolve_opponent_index(state);
    if (opp_idx >= 0 && opp_idx < static_cast<int>(state.all_players.size())) {
        const auto& opp = state.all_players[opp_idx];

        // 对手是否立直
        if (opp.reach_declared) return 0.95f;

        // 基于牌河长度
        int discard_count = static_cast<int>(opp.kawa.size());
        if (discard_count >= 12) prob += 0.3f;
        else if (discard_count >= 8) prob += 0.2f;
        else if (discard_count >= 5) prob += 0.1f;

        // 基于副露数量
        int meld_count = static_cast<int>(opp.fuuro.size());
        if (meld_count >= 3) prob += 0.25f;
        else if (meld_count >= 2) prob += 0.15f;
        else if (meld_count >= 1) prob += 0.08f;

        // 墙牌少时概率上升
        if (state.wall_remaining <= 15) prob += 0.2f;
        else if (state.wall_remaining <= 30) prob += 0.1f;
    }

    return std::min(0.95f, std::max(0.05f, prob));
}

int LinHaiEVEngine::estimate_remaining_turns(const GameState& state) const {
    // 2人临海: 每人轮流摸牌, 剩余摸牌次数 ≈ wall_remaining / 2
    return std::max(1, state.wall_remaining / 2);
}

float LinHaiEVEngine::calc_ryukyoku_prob(int turn, float my_agari_prob, float opp_tenpai_prob) const {
    // 简化流局概率
    float base = 1.0f - 0.75f * std::max(0.0f, (18.0f - turn) / 18.0f);
    return std::max(0.0f, std::min(1.0f, base * (1.0f - my_agari_prob)));
}

// ============================================================================
// 效率计算
// ============================================================================

EfficiencyResult LinHaiEVEngine::calc_efficiency(const Hai_Array& tehai, const Hai_Array& visible) const {
    EfficiencyResult result;
    result.shanten = calc_linhai_shanten(tehai);
    result.ukeire = 0;
    result.ukeire_count = 0;
    result.win_ukeire_count = 0;
    result.effective_tiles.clear();

    int current_shanten = result.shanten;

    for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
        int hai = LINHAI_VALID_TILES[i];
        int remaining = 4 - visible[hai];
        if (remaining <= 0) continue;

        Hai_Array tmp = tehai;
        tmp[hai]++;
        int new_shanten = calc_linhai_shanten(tmp);
        if (new_shanten < current_shanten) {
            result.ukeire++;
            result.ukeire_count += remaining;
            result.effective_tiles.push_back(hai);

            // 直接胡牌判定 (shanten=0 → -1)
            if (new_shanten == -1) {
                result.win_ukeire_count += remaining;
            }
        }
    }

    return result;
}

EfficiencyResult LinHaiEVEngine::calc_efficiency_after_discard(
    const Hai_Array& tehai, int discard_hai, const Hai_Array& visible) const {
    Hai_Array tmp = tehai;
    if (tmp[discard_hai] > 0) {
        tmp[discard_hai]--;
    }
    Hai_Array new_visible = visible;
    new_visible[discard_hai]++;
    return calc_efficiency(tmp, new_visible);
}

// ============================================================================
// 防守计算
// ============================================================================

std::array<float, 38> LinHaiEVEngine::calc_houjuu_probs(const GameState& state) const {
    std::array<float, 38> probs = {};
    float opp_tenpai = estimate_opponent_tenpai_prob(state);

    // 危险度随墙牌减少上升
    float danger_scale = (state.wall_remaining <= 20) ? 1.4f :
                         (state.wall_remaining <= 30) ? 1.2f : 1.0f;
    float base_prob = opp_tenpai * 0.05f * danger_scale;

    for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
        probs[LINHAI_VALID_TILES[i]] = base_prob;
    }

    // 现物分析: 对手实际打牌 (opponent_discards_ 来自Python传入)
    for (int hai : opponent_discards_) {
        if (hai > 0 && hai < 38 && is_linhai_valid_tile(hai)) {
            probs[hai] *= 0.05f;  // 现物: 几乎安全
        }
    }

    // 同时检查 state kawa (如果有实际数据)
    int opp_idx = resolve_opponent_index(state);
    if (opp_idx >= 0 && opp_idx < static_cast<int>(state.all_players.size())) {
        for (const auto& sutehai : state.all_players[opp_idx].kawa) {
            int hai = sutehai.hai;
            if (hai > 0 && hai < 38) {
                probs[hai] = std::min(probs[hai], base_prob * 0.05f);
            }
        }

        // 筋牌分析 (基于 opponent_discards_)
        for (int hai : opponent_discards_) {
            if (hai <= 0 || hai >= 30) continue;
            int rank = hai % 10;
            int base_suit = hai - rank;
            if (rank == 4 || rank == 5 || rank == 6) {
                if (rank - 3 >= 1) probs[base_suit + rank - 3] *= 0.5f;
                if (rank + 3 <= 9) probs[base_suit + rank + 3] *= 0.5f;
            }
        }

        // 壁分析
        Hai_Array visible = get_linhai_visible_tiles(state);
        for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
            int hai = LINHAI_VALID_TILES[i];
            if (hai >= 31) continue;
            if (visible[hai] >= 3) {
                int rank = hai % 10;
                int base_suit = hai - rank;
                if (rank >= 2) probs[base_suit + rank - 1] *= 0.6f;
                if (rank <= 8) probs[base_suit + rank + 1] *= 0.6f;
            }
        }
    }

    // 白板放铳概率极低
    probs[35] *= 0.1f;

    return probs;
}

float LinHaiEVEngine::calc_betaori_ev(
    const Hai_Array& tehai,
    const std::array<float, 38>& houjuu_probs,
    float other_value,
    int ori_turns
) const {
    // 使用 Akochan 的 Betaori 算法
    // 按危险度排序手牌，计算弃牌序列的期望损失

    struct RiskTile {
        int hai;
        int count;
        float risk_coeff;
    };

    std::vector<RiskTile> tiles;
    float houjuu_decay = 0.9f;

    for (int hai = 1; hai < 38; hai++) {
        if (tehai[hai] <= 0) continue;
        if (!is_linhai_valid_tile(hai)) continue;

        float prob = houjuu_probs[hai];
        float value = -8000.0f;  // 平均放铳损失 (临海约8000分)
        float beta = 1.0f - powf(houjuu_decay, static_cast<float>(tehai[hai]));

        float coeff = prob * (other_value - value) / (prob + beta - prob * beta);
        tiles.push_back({hai, tehai[hai], coeff});
    }

    // 按风险系数排序 (低风险优先弃出)
    std::sort(tiles.begin(), tiles.end(), [](const RiskTile& a, const RiskTile& b) {
        return a.risk_coeff < b.risk_coeff;
    });

    float total_houjuu_prob = 0.0f;
    float total_ev = 0.0f;
    float coeff_acc = 1.0f;
    int num = 0;

    for (const auto& tile : tiles) {
        if (num >= ori_turns) break;
        total_houjuu_prob += coeff_acc * houjuu_probs[tile.hai];
        total_ev += coeff_acc * houjuu_probs[tile.hai] * (-8000.0f);
        coeff_acc *= (1.0f - houjuu_probs[tile.hai]) * powf(houjuu_decay, static_cast<float>(tile.count));
        num += tile.count;
    }

    total_ev += (1.0f - total_houjuu_prob) * other_value;
    return total_ev;
}

// ============================================================================
// 价值计算
// ============================================================================

float LinHaiEVEngine::estimate_agari_value(const Hai_Array& tehai, int jikaze, bool has_kan) const {
    // 估算和了时的期望得分
    // 基于手牌结构估算番数

    float base_value = 2000.0f;  // 1番30符基础

    // 白板(得牌): 百搭加成
    int white_count = tehai[35];
    if (white_count > 0) {
        base_value *= (1.0f + 0.35f * std::min(white_count, 2));
    }

    // 门前清 → +1番 (立直等效)
    // 简化: 手牌总数 > 10 表示门前
    int hand_count = 0;
    for (int hai = 1; hai < 38; hai++) hand_count += tehai[hai];
    if (hand_count >= 10) base_value += 1000.0f;

    // 自风番
    if (jikaze == 0) base_value += 500.0f;  // 东家加分

    // 杠
    if (has_kan) base_value += 800.0f;

    // 使用抓冲加值
    BonusValue bonus = calc_total_bonus_value(tehai, jikaze, has_kan, true);
    base_value += bonus.total_value * 0.3f;

    return base_value;
}

float LinHaiEVEngine::calc_linhai_modifier(int discard_hai, const GameState& state) {
    float modifier = 0.0f;

    // 1. 白板保护
    if (discard_hai == 35) {
        // 白板(得牌)是百搭，绝不能打
        modifier -= 50000.0f;
    }

    // 2. 抓冲价值损失
    int shanten = calc_linhai_shanten(state.tehai);
    bool is_tenpai = (shanten <= 0);
    BonusValue bonus_before = calc_total_bonus_value(
        state.tehai, state.jikaze, state.has_kan, is_tenpai);

    Hai_Array tmp = state.tehai;
    if (tmp[discard_hai] > 0) tmp[discard_hai]--;
    BonusValue bonus_after = calc_total_bonus_value(
        tmp, state.jikaze, state.has_kan, is_tenpai);

    float bonus_loss = bonus_before.total_value - bonus_after.total_value;
    modifier -= bonus_loss * config.bonus_weight;

    // 3. 承包风险
    float contract_risk = risk_evaluator.evaluate_discard_risk(
        discard_hai, state.all_players, state.current_player).total_risk;
    modifier -= contract_risk * config.risk_weight;

    return modifier;
}

// ============================================================================
// 多巡DP期望值
// ============================================================================

float LinHaiEVEngine::calc_multi_turn_ev(
    const Hai_Array& tehai,
    const Hai_Array& visible,
    const GameState& state,
    int remaining_turns,
    float opp_tenpai_prob,
    int pool_size
) const {
    EfficiencyResult eff = calc_efficiency(tehai, visible);

    // 一人麻将和了概率 (使用实际 pool_size)
    float solo_prob = calc_solo_agari_prob(eff.shanten, eff.ukeire_count, remaining_turns, pool_size);

    // 考虑对手后的和了概率
    int turn_approx = std::max(1, 18 - remaining_turns);
    float my_agari_prob = calc_agari_prob(solo_prob, turn_approx, opp_tenpai_prob);

    // 和了价值
    float agari_value = estimate_agari_value(tehai, state.jikaze, state.has_kan);

    // 对手和了概率
    float opp_agari_prob = (1.0f - my_agari_prob) * opp_tenpai_prob * 0.5f;

    // 流局概率
    float ryuukyoku_prob = calc_ryukyoku_prob(turn_approx, my_agari_prob, opp_tenpai_prob);
    ryuukyoku_prob = std::min(ryuukyoku_prob, 1.0f - my_agari_prob - opp_agari_prob);
    ryuukyoku_prob = std::max(0.0f, ryuukyoku_prob);

    float agari_ev = my_agari_prob * agari_value;
    float opp_agari_ev = opp_agari_prob * (-8000.0f);
    float ryuukyoku_ev = ryuukyoku_prob * (eff.shanten <= 0 ? 1500.0f : -500.0f);

    float total_ev = agari_ev + opp_agari_ev + ryuukyoku_ev;

    if (config.aggressive_mode) {
        total_ev += my_agari_prob * agari_value * 0.3f;
    }

    if (config.defensive_mode || opp_tenpai_prob > 0.5f) {
        float defense_weight = config.defensive_mode ? 0.5f : opp_tenpai_prob - 0.3f;
        defense_weight = std::max(0.0f, std::min(0.8f, defense_weight));

        std::array<float, 38> houjuu_probs = calc_houjuu_probs(state);
        float betaori_ev = calc_betaori_ev(tehai, houjuu_probs, 0.0f, remaining_turns);
        total_ev = total_ev * (1.0f - defense_weight) + betaori_ev * defense_weight;
    }

    return total_ev;
}

// ============================================================================
// 核心推荐接口
// ============================================================================

std::vector<TileEV> LinHaiEVEngine::recommend_discard_ev(const GameState& state) {
    std::vector<TileEV> results;

    Hai_Array visible = get_linhai_visible_tiles(state);
    float opp_tenpai = estimate_opponent_tenpai_prob(state);
    int remaining_turns = estimate_remaining_turns(state);
    std::array<float, 38> houjuu_probs = calc_houjuu_probs(state);

    // 计算实际剩余牌数 (pool_size): 总剩余 - 自己手牌
    int pool_size = 0;
    for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
        pool_size += std::max(0, 4 - visible[LINHAI_VALID_TILES[i]]);
    }
    for (int hai = 1; hai < 38; hai++) pool_size -= state.tehai[hai];
    pool_size = std::max(1, pool_size);

    // 对每张可以打的牌计算EV
    for (int hai = 1; hai < 38; hai++) {
        if (state.tehai[hai] <= 0) continue;
        if (!is_linhai_valid_tile(hai)) continue;

        TileEV ev;
        ev.hai = hai;

        Hai_Array after = state.tehai;
        after[hai]--;
        Hai_Array visible_after = visible;
        visible_after[hai]++;

        EfficiencyResult eff = calc_efficiency(after, visible_after);
        ev.shanten_after = eff.shanten;
        ev.ukeire_after = eff.ukeire_count;

        // 多巡期望值 (使用 pool_size)
        ev.ev_offense = calc_multi_turn_ev(after, visible_after, state, remaining_turns, opp_tenpai, pool_size);

        ev.risk_houjuu = houjuu_probs[hai];
        ev.ev_defense = -houjuu_probs[hai] * 8000.0f;
        ev.ev_linhai_bonus = calc_linhai_modifier(hai, state);

        ev.ev_total = ev.ev_offense + ev.ev_linhai_bonus;

        if (opp_tenpai > 0.4f) {
            float def_weight = std::min(0.6f, opp_tenpai - 0.2f);
            ev.ev_total = ev.ev_total * (1.0f - def_weight) + ev.ev_defense * def_weight;
        }

        ev.explanation = make_explanation(ev);
        results.push_back(ev);
    }

    // 按EV排序
    std::sort(results.begin(), results.end(), [](const TileEV& a, const TileEV& b) {
        return a.ev_total > b.ev_total;
    });

    return results;
}

DiscardRecommendation LinHaiEVEngine::recommend_best_discard_v2(const GameState& state) {
    auto evs = recommend_discard_ev(state);
    if (evs.empty()) return DiscardRecommendation();

    const TileEV& best = evs[0];
    DiscardRecommendation rec;
    rec.hai = best.hai;
    rec.score = best.ev_total;
    rec.shanten = best.shanten_after;
    rec.bonus_value = best.ev_linhai_bonus;
    rec.risk_value = best.risk_houjuu * 8000.0f;
    rec.explanation = best.explanation;
    return rec;
}

std::vector<DiscardRecommendation> LinHaiEVEngine::recommend_discard_v2(const GameState& state) {
    auto evs = recommend_discard_ev(state);
    std::vector<DiscardRecommendation> results;

    for (const auto& ev : evs) {
        DiscardRecommendation rec;
        rec.hai = ev.hai;
        rec.score = ev.ev_total;
        rec.shanten = ev.shanten_after;
        rec.bonus_value = ev.ev_linhai_bonus;
        rec.risk_value = ev.risk_houjuu * 8000.0f;
        rec.explanation = ev.explanation;
        results.push_back(rec);
    }

    return results;
}

// ============================================================================
// 响应动作推荐
// ============================================================================

std::vector<ResponseRecommendation> LinHaiEVEngine::recommend_response_v2(
    const GameState& state,
    int target_hai,
    int from_player,
    const std::vector<std::string>& available_actions
) {
    std::vector<ResponseRecommendation> recommendations;
    Hai_Array visible = get_linhai_visible_tiles(state);
    float opp_tenpai = estimate_opponent_tenpai_prob(state);
    int remaining_turns = estimate_remaining_turns(state);

    // pool_size
    Hai_Array visible2 = visible;
    int pool_size = 0;
    for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
        pool_size += std::max(0, 4 - visible2[LINHAI_VALID_TILES[i]]);
    }
    for (int hai = 1; hai < 38; hai++) pool_size -= state.tehai[hai];
    pool_size = std::max(1, pool_size);

    // 当前手牌EV (pass的基准)
    EfficiencyResult current_eff = calc_efficiency(state.tehai, visible);
    float current_ev = calc_multi_turn_ev(state.tehai, visible, state, remaining_turns, opp_tenpai, pool_size);

    // Pass
    {
        ResponseRecommendation rec;
        rec.action = "pass";
        rec.target_hai = target_hai;
        rec.shanten = current_eff.shanten;
        rec.score = current_ev;

        // 防守加值: pass 避免暴露信息
        float threat = opp_tenpai;
        rec.score += threat * 500.0f;

        rec.explanation = threat >= 0.45f ? "场面偏紧，保持门前更稳" : "保持门前，等待更优进张";
        recommendations.push_back(rec);
    }

    for (const auto& action : available_actions) {
        if (action == "pass" || action == "guo") continue;

        if (action == "hu") {
            if (!can_win_) continue;  // 过胡不胡: 本局不能再胡
            ResponseRecommendation rec;
            rec.action = "hu";
            rec.target_hai = target_hai;
            rec.score = 1000000.0f;
            rec.shanten = -1;
            rec.explanation = "胡牌优先";
            recommendations.push_back(rec);
            continue;
        }

        if (action == "peng" && state.tehai[target_hai] >= 2) {
            // 模拟碰后状态
            GameState next = state;
            next.tehai[target_hai] = std::max(0, next.tehai[target_hai] - 2);
            Fuuro_Elem fuuro;
            fuuro.type = FT_PON;
            fuuro.hai = target_hai;
            fuuro.consumed = {target_hai, target_hai};
            fuuro.target_relative = 1;
            next.fuuro.push_back(fuuro);

            // 碰后找最佳弃牌
            auto discard_recs = recommend_discard_v2(next);
            if (!discard_recs.empty()) {
                ResponseRecommendation rec;
                rec.action = "peng";
                rec.target_hai = target_hai;
                rec.consumed = {target_hai, target_hai};
                rec.discard_hai = discard_recs[0].hai;
                rec.shanten = discard_recs[0].shanten;
                rec.score = discard_recs[0].score;
                rec.bonus_value = discard_recs[0].bonus_value;

                // 速度加成
                int speed_gain = current_eff.shanten - rec.shanten;
                if (speed_gain > 0) rec.score += speed_gain * 2500.0f;

                // 承包风险
                float contract_risk = risk_evaluator.get_contract_evaluator()
                    .evaluate_meld_risk(from_player);
                rec.risk_value = contract_risk;
                rec.score -= contract_risk * 0.8f;

                rec.explanation = "碰后打" + hai_int_to_str(rec.discard_hai) +
                    "，" + std::to_string(rec.shanten) + "向听";
                recommendations.push_back(rec);
            }
        }

        if (action == "chi") {
            // 生成所有吃组合
            if (target_hai >= 30) continue;  // 字牌不能吃
            int rank = target_hai % 10;
            int base = target_hai - rank;

            auto try_chi = [&](int c1, int c2) {
                if (c1 <= 0 || c1 >= 38 || c2 <= 0 || c2 >= 38) return;
                if (state.tehai[c1] <= 0 || state.tehai[c2] <= 0) return;
                if (!is_linhai_valid_tile(c1) || !is_linhai_valid_tile(c2)) return;

                GameState next = state;
                next.tehai[c1]--;
                next.tehai[c2]--;
                Fuuro_Elem fuuro;
                fuuro.type = FT_CHI;
                fuuro.hai = target_hai;
                fuuro.consumed = {c1, c2};
                fuuro.target_relative = 1;
                next.fuuro.push_back(fuuro);

                auto discard_recs = recommend_discard_v2(next);
                if (!discard_recs.empty()) {
                    ResponseRecommendation rec;
                    rec.action = "chi";
                    rec.target_hai = target_hai;
                    rec.consumed = {c1, c2};
                    rec.discard_hai = discard_recs[0].hai;
                    rec.shanten = discard_recs[0].shanten;
                    rec.score = discard_recs[0].score;
                    rec.bonus_value = discard_recs[0].bonus_value;

                    int speed_gain = current_eff.shanten - rec.shanten;
                    if (speed_gain > 0) rec.score += speed_gain * 2000.0f;
                    else rec.score -= 1500.0f;  // 吃不提速则扣分

                    float contract_risk = risk_evaluator.get_contract_evaluator()
                        .evaluate_meld_risk(from_player);
                    rec.risk_value = contract_risk;
                    rec.score -= contract_risk * 0.8f;

                    rec.explanation = "吃后打" + hai_int_to_str(rec.discard_hai) +
                        "，" + std::to_string(rec.shanten) + "向听";
                    recommendations.push_back(rec);
                }
            };

            // 三种吃法
            if (rank >= 3) try_chi(base + rank - 2, base + rank - 1);
            if (rank >= 2 && rank <= 8) try_chi(base + rank - 1, base + rank + 1);
            if (rank <= 7) try_chi(base + rank + 1, base + rank + 2);
        }

        if (action == "gang" && state.tehai[target_hai] >= 3) {
            GameState next = state;
            next.tehai[target_hai] = std::max(0, next.tehai[target_hai] - 3);
            next.has_kan = true;
            Fuuro_Elem fuuro;
            fuuro.type = FT_DAIMINKAN;
            fuuro.hai = target_hai;
            fuuro.consumed = {target_hai, target_hai, target_hai};
            fuuro.target_relative = 1;
            next.fuuro.push_back(fuuro);

            Hai_Array next_visible = get_linhai_visible_tiles(next);
            EfficiencyResult next_eff = calc_efficiency(next.tehai, next_visible);
            float next_ev = calc_multi_turn_ev(next.tehai, next_visible, next, remaining_turns, opp_tenpai, pool_size);

            ResponseRecommendation rec;
            rec.action = "gang";
            rec.target_hai = target_hai;
            rec.consumed = {target_hai, target_hai, target_hai};
            rec.shanten = next_eff.shanten;
            rec.score = next_ev + 2000.0f;  // 杠的额外价值 (杠头+翻屁股)

            if (opp_tenpai > 0.5f) rec.score -= 2000.0f;  // 高威胁时杠扣分

            rec.explanation = "明杠" + hai_int_to_str(target_hai);
            recommendations.push_back(rec);
        }
    }

    std::sort(recommendations.begin(), recommendations.end(),
        [](const ResponseRecommendation& a, const ResponseRecommendation& b) {
            return a.score > b.score;
        });

    return recommendations;
}

// ============================================================================
// 辅助
// ============================================================================

void LinHaiEVEngine::set_opponent_discards(const std::vector<int>& discards) {
    opponent_discards_ = discards;
}

void LinHaiEVEngine::set_aggressive_mode(bool enabled) {
    config.aggressive_mode = enabled;
    if (enabled) config.defensive_mode = false;
}

void LinHaiEVEngine::set_defensive_mode(bool enabled) {
    config.defensive_mode = enabled;
    if (enabled) config.aggressive_mode = false;
}

std::string LinHaiEVEngine::make_explanation(const TileEV& ev) const {
    std::ostringstream oss;

    if (ev.shanten_after < 0) {
        oss << "已胡牌";
    } else if (ev.shanten_after == 0) {
        oss << "听牌, " << ev.ukeire_after << "种受入";
    } else {
        oss << ev.shanten_after << "向听, " << ev.ukeire_after << "种受入";
    }

    if (ev.ev_linhai_bonus < -10000.0f) {
        oss << ", 白板不可打";
    } else if (ev.ev_linhai_bonus < -2000.0f) {
        oss << ", 损失抓冲价值";
    }

    if (ev.risk_houjuu > 0.1f) {
        oss << ", 放铳风险" << static_cast<int>(ev.risk_houjuu * 100) << "%";
    }

    oss << " (EV:" << std::fixed << std::setprecision(0) << ev.ev_total << ")";

    return oss.str();
}

} // namespace linhai
