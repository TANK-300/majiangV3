#include "linhai_ai_engine.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace linhai {

// ============================================================================
// 辅助函数实现
// ============================================================================

std::vector<int> get_possible_discards(const Hai_Array& tehai) {
    std::vector<int> discards;
    for (int hai = 1; hai < 38; hai++) {
        if (tehai[hai] > 0) {
            discards.push_back(hai);
        }
    }
    return discards;
}

void print_recommendation(const DiscardRecommendation& rec) {
    std::cout << "推荐打: " << hai_int_to_str(rec.hai)
              << " (评分: " << std::fixed << std::setprecision(1) << rec.score << ")"
              << std::endl;
    std::cout << "  向听数: " << rec.shanten << std::endl;
    std::cout << "  价值: " << rec.bonus_value << " 分" << std::endl;
    std::cout << "  风险: " << rec.risk_value << " 分" << std::endl;
    std::cout << "  理由: " << rec.explanation << std::endl;
}

void print_all_recommendations(const std::vector<DiscardRecommendation>& recs) {
    std::cout << "\n所有打牌选择（按评分排序）:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (size_t i = 0; i < recs.size(); i++) {
        std::cout << (i + 1) << ". ";
        std::cout << hai_int_to_str(recs[i].hai) << " ";
        std::cout << "评分:" << std::fixed << std::setprecision(1) << recs[i].score << " ";
        std::cout << "向听:" << recs[i].shanten << " ";
        std::cout << "价值:" << std::setprecision(0) << recs[i].bonus_value << " ";
        std::cout << "风险:" << recs[i].risk_value << std::endl;
    }
}

// ============================================================================
// LinHaiAIEngine 实现
// ============================================================================

LinHaiAIEngine::LinHaiAIEngine() : config() {
}

LinHaiAIEngine::LinHaiAIEngine(const AIConfig& cfg) : config(cfg) {
}

int LinHaiAIEngine::evaluate_shanten_after_discard(const Hai_Array& tehai, int discard_hai) {
    // 创建临时手牌，移除要打出的牌
    Hai_Array tmp = tehai;
    if (tmp[discard_hai] > 0) {
        tmp[discard_hai]--;
    }

    // 计算向听数
    return calc_linhai_shanten(tmp);
}

int LinHaiAIEngine::evaluate_effective_shanten(const Hai_Array& tehai, int fuuro_count) {
    int raw_shanten = calc_linhai_shanten(tehai);
    int adjusted_shanten = raw_shanten - fuuro_count * 2;
    return std::max(-1, adjusted_shanten);
}

float LinHaiAIEngine::evaluate_keep_value(
    const Hai_Array& tehai,
    int keep_hai,
    int jikaze,
    bool has_kan,
    bool is_tenpai
) {
    // 评估保留某张牌的价值
    // 这里简化为：如果打出这张牌，会损失多少价值

    // 创建临时手牌，移除这张牌
    Hai_Array tmp = tehai;
    if (tmp[keep_hai] > 0) {
        tmp[keep_hai]--;
    }

    // 计算原手牌的价值
    BonusValue original_bonus = calc_total_bonus_value(tehai, jikaze, has_kan, is_tenpai);

    // 计算打出后的价值
    BonusValue after_bonus = calc_total_bonus_value(tmp, jikaze, has_kan, is_tenpai);

    // 损失的价值
    float loss = original_bonus.total_value - after_bonus.total_value;

    return loss;
}

float LinHaiAIEngine::evaluate_discard_risk(
    int discard_hai,
    const GameState& state
) {
    // 评估打出某张牌的风险
    RiskValue risk = risk_evaluator.evaluate_discard_risk(
        discard_hai,
        state.all_players,
        state.current_player
    );

    return risk.total_risk;
}

float LinHaiAIEngine::calculate_total_score(
    int discard_hai,
    const GameState& state
) {
    Hai_Array tmp = state.tehai;
    if (tmp[discard_hai] > 0) {
        tmp[discard_hai]--;
    }

    // 1. 计算打出后的向听数
    int shanten = evaluate_effective_shanten(tmp, static_cast<int>(state.fuuro.size()));

    // 2. 计算当前向听数
    int current_shanten = evaluate_effective_shanten(state.tehai, static_cast<int>(state.fuuro.size()));
    bool is_tenpai = (current_shanten == 0);

    // 3. 计算保留价值损失
    float bonus_loss = evaluate_keep_value(
        state.tehai,
        discard_hai,
        state.jikaze,
        state.has_kan,
        is_tenpai
    );

    // 4. 计算打出风险
    float risk = evaluate_discard_risk(discard_hai, state);

    // 5. 综合评分
    // 评分 = -向听数惩罚 - 价值损失 - 风险
    float score = 0.0f;

    // 向听数惩罚
    score -= shanten * config.shanten_weight;

    // 价值损失惩罚
    score -= bonus_loss * config.bonus_weight;

    // 风险惩罚
    score -= risk * config.risk_weight;

    // 策略调整
    if (config.aggressive_mode) {
        // 进攻模式：向听数权重加倍
        score -= shanten * config.shanten_weight;
    }

    if (config.defensive_mode) {
        // 防守模式：风险权重加倍
        score -= risk * config.risk_weight;
    }

    return score;
}

std::string LinHaiAIEngine::generate_explanation(
    int discard_hai,
    int shanten,
    float bonus_value,
    float risk_value,
    float total_score
) {
    std::ostringstream oss;

    // 向听数说明
    if (shanten == -1) {
        oss << "已胡牌";
    } else if (shanten == 0) {
        oss << "听牌";
    } else {
        oss << shanten << "向听";
    }

    // 价值说明
    if (bonus_value > 5000) {
        oss << ", 损失大量价值";
    } else if (bonus_value > 2000) {
        oss << ", 损失较多价值";
    } else if (bonus_value > 500) {
        oss << ", 损失少量价值";
    }

    // 风险说明
    if (risk_value > 8000) {
        oss << ", 极高风险";
    } else if (risk_value > 5000) {
        oss << ", 高风险";
    } else if (risk_value > 2000) {
        oss << ", 中等风险";
    } else if (risk_value > 500) {
        oss << ", 低风险";
    }

    return oss.str();
}

std::vector<std::vector<int>> LinHaiAIEngine::generate_chi_combinations(const Hai_Array& tehai, int target_hai) {
    std::vector<std::vector<int>> combinations;
    if (target_hai <= 0 || target_hai >= 30 || target_hai % 10 == 0) {
        return combinations;
    }

    const int rank = target_hai % 10;

    if (rank >= 3 && tehai[target_hai - 2] > 0 && tehai[target_hai - 1] > 0) {
        combinations.push_back({target_hai - 2, target_hai - 1});
    }
    if (rank >= 2 && rank <= 8 && tehai[target_hai - 1] > 0 && tehai[target_hai + 1] > 0) {
        combinations.push_back({target_hai - 1, target_hai + 1});
    }
    if (rank <= 7 && tehai[target_hai + 1] > 0 && tehai[target_hai + 2] > 0) {
        combinations.push_back({target_hai + 1, target_hai + 2});
    }

    return combinations;
}

GameState LinHaiAIEngine::simulate_peng(const GameState& state, int target_hai) {
    GameState next = state;
    next.tehai[target_hai] = std::max(0, next.tehai[target_hai] - 2);

    Fuuro_Elem fuuro;
    fuuro.type = FT_PON;
    fuuro.hai = target_hai;
    fuuro.consumed = {target_hai, target_hai};
    fuuro.target_relative = 1;
    next.fuuro.push_back(fuuro);

    return next;
}

GameState LinHaiAIEngine::simulate_chi(const GameState& state, int target_hai, const std::vector<int>& consumed) {
    GameState next = state;
    for (int hai : consumed) {
        next.tehai[hai] = std::max(0, next.tehai[hai] - 1);
    }

    Fuuro_Elem fuuro;
    fuuro.type = FT_CHI;
    fuuro.hai = target_hai;
    fuuro.consumed = consumed;
    fuuro.target_relative = 1;
    next.fuuro.push_back(fuuro);

    return next;
}

int LinHaiAIEngine::select_virtual_replacement_tile(const GameState& state) {
    int best_hai = 0;
    float best_score = -1.0e30f;

    for (int hai = 1; hai < 38; hai++) {
        if (hai % 10 == 0) {
            continue;
        }
        if (state.tehai[hai] >= 4) {
            continue;
        }

        Hai_Array tmp = state.tehai;
        tmp[hai]++;
        int shanten = evaluate_effective_shanten(tmp, static_cast<int>(state.fuuro.size()));
        BonusValue bonus = calc_total_bonus_value(tmp, state.jikaze, state.has_kan, shanten == 0);
        float score = -shanten * config.shanten_weight + bonus.total_value * 0.35f;
        if (score > best_score) {
            best_score = score;
            best_hai = hai;
        }
    }

    return best_hai;
}

GameState LinHaiAIEngine::simulate_ming_gang(const GameState& state, int target_hai) {
    GameState next = state;
    next.tehai[target_hai] = std::max(0, next.tehai[target_hai] - 3);
    next.has_kan = true;

    Fuuro_Elem fuuro;
    fuuro.type = FT_DAIMINKAN;
    fuuro.hai = target_hai;
    fuuro.consumed = {target_hai, target_hai, target_hai};
    fuuro.target_relative = 1;
    next.fuuro.push_back(fuuro);

    const int replacement = select_virtual_replacement_tile(next);
    if (replacement > 0) {
        next.tehai[replacement]++;
    }

    return next;
}

float LinHaiAIEngine::estimate_response_threat(const GameState& state, int from_player) {
    float threat = 0.0f;

    if (state.wall_remaining <= 18) {
        threat += 0.42f;
    } else if (state.wall_remaining <= 30) {
        threat += 0.25f;
    } else if (state.wall_remaining <= 45) {
        threat += 0.12f;
    }

    if (from_player >= 0 && from_player < static_cast<int>(state.all_players.size())) {
        const Player_State& source = state.all_players[from_player];
        if (source.reach_declared) {
            threat += 0.55f;
        }

        const float fuuro_pressure = std::min(0.33f, static_cast<float>(source.fuuro.size()) * 0.11f);
        threat += fuuro_pressure;

        const int discard_count = static_cast<int>(source.kawa.size());
        if (discard_count >= 10) {
            threat += 0.16f;
        } else if (discard_count >= 6) {
            threat += 0.08f;
        }
    }

    return std::max(0.0f, std::min(1.0f, threat));
}

int LinHaiAIEngine::estimate_pair_count(const Hai_Array& tehai) {
    int pairs = 0;
    for (int hai = 1; hai < 38; hai++) {
        if (hai % 10 == 0) {
            continue;
        }
        if (tehai[hai] >= 2) {
            pairs++;
        }
    }
    return pairs;
}

int LinHaiAIEngine::estimate_taatsu_count(const Hai_Array& tehai) {
    int taatsu = 0;
    for (int base : {1, 11, 21}) {
        for (int rank = 1; rank <= 8; rank++) {
            const int hai = base + rank - 1;
            if (tehai[hai] > 0 && tehai[hai + 1] > 0) {
                taatsu++;
            }
        }
        for (int rank = 1; rank <= 7; rank++) {
            const int hai = base + rank - 1;
            if (tehai[hai] > 0 && tehai[hai + 2] > 0) {
                taatsu++;
            }
        }
    }
    return taatsu;
}

int LinHaiAIEngine::estimate_isolated_count(const Hai_Array& tehai) {
    int isolated = 0;
    for (int hai = 1; hai < 38; hai++) {
        if (hai % 10 == 0 || tehai[hai] <= 0) {
            continue;
        }

        if (hai >= 31) {
            if (tehai[hai] == 1) {
                isolated++;
            }
            continue;
        }

        const int rank = hai % 10;
        bool connected = false;
        if (tehai[hai] >= 2) {
            connected = true;
        }
        if (rank > 1 && tehai[hai - 1] > 0) {
            connected = true;
        }
        if (rank < 9 && tehai[hai + 1] > 0) {
            connected = true;
        }
        if (rank > 2 && tehai[hai - 2] > 0) {
            connected = true;
        }
        if (rank < 8 && tehai[hai + 2] > 0) {
            connected = true;
        }
        if (!connected) {
            isolated++;
        }
    }
    return isolated;
}

float LinHaiAIEngine::evaluate_shape_value(const Hai_Array& tehai) {
    const int pairs = estimate_pair_count(tehai);
    const int taatsu = estimate_taatsu_count(tehai);
    const int isolated = estimate_isolated_count(tehai);
    return static_cast<float>(pairs) * 180.0f + static_cast<float>(taatsu) * 260.0f - static_cast<float>(isolated) * 140.0f;
}

ResponseRecommendation LinHaiAIEngine::evaluate_pass_response(const GameState& state, int target_hai, int from_player) {
    ResponseRecommendation rec;
    rec.action = "pass";
    rec.target_hai = target_hai;

    const int shanten = evaluate_effective_shanten(state.tehai, static_cast<int>(state.fuuro.size()));
    BonusValue bonus = calc_total_bonus_value(state.tehai, state.jikaze, state.has_kan, shanten == 0);
    const float threat = estimate_response_threat(state, from_player);
    const float defense_bonus =
        threat * (700.0f + static_cast<float>(std::max(0, 2 - shanten)) * 650.0f)
        + (state.wall_remaining <= 24 ? 320.0f : 0.0f);

    rec.shanten = shanten;
    rec.bonus_value = bonus.total_value;
    rec.risk_value = threat * 1000.0f;
    rec.score = -shanten * config.shanten_weight + bonus.total_value * 0.20f - 600.0f + defense_bonus;
    rec.explanation = threat >= 0.45f ? "场面偏紧，保持门前更稳" : "保持门前，等待更优进张";

    return rec;
}

ResponseRecommendation LinHaiAIEngine::evaluate_open_response(
    const GameState& original_state,
    const GameState& simulated_state,
    const std::string& action,
    int target_hai,
    const std::vector<int>& consumed,
    int from_player
) {
    ResponseRecommendation rec;
    rec.action = action;
    rec.target_hai = target_hai;
    rec.consumed = consumed;

    const int current_shanten = evaluate_effective_shanten(
        original_state.tehai,
        static_cast<int>(original_state.fuuro.size())
    );
    const float threat = estimate_response_threat(original_state, from_player);
    const float original_shape = evaluate_shape_value(original_state.tehai);

    DiscardRecommendation best_discard = recommend_best_discard(simulated_state);
    Hai_Array after_discard = simulated_state.tehai;
    if (best_discard.hai > 0 && after_discard[best_discard.hai] > 0) {
        after_discard[best_discard.hai]--;
    }

    const int after_shanten = evaluate_effective_shanten(
        after_discard,
        static_cast<int>(simulated_state.fuuro.size())
    );
    BonusValue bonus = calc_total_bonus_value(
        after_discard,
        simulated_state.jikaze,
        simulated_state.has_kan,
        after_shanten == 0
    );
    const float shape_gain = evaluate_shape_value(after_discard) - original_shape;

    float contract_risk = 0.0f;
    if (action == "chi" || action == "peng") {
        contract_risk = risk_evaluator.get_contract_evaluator().evaluate_meld_risk(from_player) + 350.0f;
    }

    const int speed_gain = current_shanten - after_shanten;
    float action_bonus = 0.0f;
    if (action == "chi") {
        action_bonus = 300.0f;
    } else if (action == "peng") {
        action_bonus = 900.0f;
    } else if (action == "gang") {
        action_bonus = 1800.0f;
    }

    if (action == "chi") {
        if (speed_gain > 0) {
            action_bonus += 600.0f;
        } else {
            action_bonus -= 1400.0f + threat * 700.0f;
        }
        action_bonus += std::max(-900.0f, std::min(1200.0f, shape_gain * 1.8f));
    } else if (action == "peng") {
        if (speed_gain > 0) {
            action_bonus += 950.0f + (1.0f - threat) * 500.0f;
        } else {
            action_bonus -= threat * 800.0f;
        }
        action_bonus += std::max(-700.0f, std::min(900.0f, shape_gain * 1.3f));
    } else if (action == "gang") {
        float gang_window_bonus = (1.0f - threat) * 1700.0f;
        if (original_state.wall_remaining >= 36) {
            gang_window_bonus += 500.0f;
        }
        if (after_shanten <= current_shanten) {
            gang_window_bonus += 700.0f;
        }
        float gang_brake = threat * (speed_gain <= 0 ? 2600.0f : 1400.0f);
        if (after_shanten > current_shanten) {
            gang_brake += 1600.0f;
        }
        if (current_shanten <= 1) {
            gang_brake += threat * 800.0f;
        }
        action_bonus += gang_window_bonus - gang_brake;
        action_bonus += std::max(-600.0f, std::min(600.0f, shape_gain));
    }

    rec.discard_hai = best_discard.hai;
    rec.shanten = after_shanten;
    rec.bonus_value = bonus.total_value;
    rec.risk_value = contract_risk + threat * 1200.0f;
    rec.score =
        best_discard.score * 0.35f +
        (-after_shanten * config.shanten_weight) +
        bonus.total_value * 0.25f +
        speed_gain * 2200.0f +
        shape_gain * 2.0f +
        action_bonus -
        contract_risk * 0.80f;

    if (config.aggressive_mode && speed_gain > 0) {
        rec.score += speed_gain * 1000.0f;
    }
    if (config.defensive_mode) {
        rec.score -= contract_risk * 0.35f;
    }

    std::ostringstream oss;
    oss << action << "后建议打" << hai_int_to_str(best_discard.hai)
        << "，" << after_shanten << "向听";
    if (action == "gang") {
        oss << (threat < 0.35f ? "，低威胁可积极开杠" : "，场面偏紧需谨慎开杠");
    } else if (action == "chi") {
        oss << (speed_gain > 0 ? "，该组合提速更明显" : "，该组合提速有限");
    }
    rec.explanation = oss.str();

    return rec;
}

std::vector<DiscardRecommendation> LinHaiAIEngine::recommend_discard(const GameState& state) {
    std::vector<DiscardRecommendation> recommendations;

    // 获取所有可能打出的牌
    std::vector<int> possible = get_possible_discards(state.tehai);

    // 评估每张牌
    for (int hai : possible) {
        DiscardRecommendation rec;
        rec.hai = hai;

        // 计算向听数
        Hai_Array tmp = state.tehai;
        if (tmp[hai] > 0) {
            tmp[hai]--;
        }
        rec.shanten = evaluate_effective_shanten(tmp, static_cast<int>(state.fuuro.size()));

        // 计算价值损失
        int current_shanten = evaluate_effective_shanten(state.tehai, static_cast<int>(state.fuuro.size()));
        bool is_tenpai = (current_shanten == 0);
        rec.bonus_value = evaluate_keep_value(
            state.tehai,
            hai,
            state.jikaze,
            state.has_kan,
            is_tenpai
        );

        // 计算风险
        rec.risk_value = evaluate_discard_risk(hai, state);

        // 计算综合评分
        rec.score = calculate_total_score(hai, state);

        // 生成说明
        rec.explanation = generate_explanation(
            hai,
            rec.shanten,
            rec.bonus_value,
            rec.risk_value,
            rec.score
        );

        recommendations.push_back(rec);
    }

    // 按评分排序（从高到低）
    std::sort(recommendations.begin(), recommendations.end(),
              [](const DiscardRecommendation& a, const DiscardRecommendation& b) {
                  return a.score > b.score;
              });

    return recommendations;
}

DiscardRecommendation LinHaiAIEngine::recommend_best_discard(const GameState& state) {
    std::vector<DiscardRecommendation> recs = recommend_discard(state);
    if (recs.empty()) {
        return DiscardRecommendation();
    }
    return recs[0];
}

std::vector<ResponseRecommendation> LinHaiAIEngine::recommend_response(
    const GameState& state,
    int target_hai,
    int from_player,
    const std::vector<std::string>& available_actions
) {
    std::vector<ResponseRecommendation> recommendations;

    recommendations.push_back(evaluate_pass_response(state, target_hai, from_player));

    for (const std::string& action : available_actions) {
        if (action == "pass" || action == "guo") {
            continue;
        }
        if (action == "hu") {
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
            recommendations.push_back(
                evaluate_open_response(
                    state,
                    simulate_peng(state, target_hai),
                    "peng",
                    target_hai,
                    {target_hai, target_hai},
                    from_player
                )
            );
            continue;
        }
        if (action == "gang" && state.tehai[target_hai] >= 3) {
            recommendations.push_back(
                evaluate_open_response(
                    state,
                    simulate_ming_gang(state, target_hai),
                    "gang",
                    target_hai,
                    {target_hai, target_hai, target_hai},
                    from_player
                )
            );
            continue;
        }
        if (action == "chi") {
            std::vector<std::vector<int>> combinations = generate_chi_combinations(state.tehai, target_hai);
            for (const std::vector<int>& combo : combinations) {
                recommendations.push_back(
                    evaluate_open_response(
                        state,
                        simulate_chi(state, target_hai, combo),
                        "chi",
                        target_hai,
                        combo,
                        from_player
                    )
                );
            }
        }
    }

    std::sort(
        recommendations.begin(),
        recommendations.end(),
        [](const ResponseRecommendation& a, const ResponseRecommendation& b) {
            return a.score > b.score;
        }
    );

    return recommendations;
}

ResponseRecommendation LinHaiAIEngine::recommend_best_response(
    const GameState& state,
    int target_hai,
    int from_player,
    const std::vector<std::string>& available_actions
) {
    std::vector<ResponseRecommendation> recs = recommend_response(state, target_hai, from_player, available_actions);
    if (recs.empty()) {
        return ResponseRecommendation();
    }
    return recs[0];
}

void LinHaiAIEngine::record_chi(int target_player) {
    risk_evaluator.record_chi(target_player);
}

void LinHaiAIEngine::record_pon(int target_player) {
    risk_evaluator.record_pon(target_player);
}

void LinHaiAIEngine::reset() {
    risk_evaluator.reset();
}

const AIConfig& LinHaiAIEngine::get_config() const {
    return config;
}

void LinHaiAIEngine::set_config(const AIConfig& cfg) {
    config = cfg;
}

void LinHaiAIEngine::set_aggressive_mode(bool enabled) {
    config.aggressive_mode = enabled;
    if (enabled) {
        config.defensive_mode = false;
    }
}

void LinHaiAIEngine::set_defensive_mode(bool enabled) {
    config.defensive_mode = enabled;
    if (enabled) {
        config.aggressive_mode = false;
    }
}

} // namespace linhai
