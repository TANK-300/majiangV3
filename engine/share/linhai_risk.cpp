#include "linhai_risk.hpp"
#include <algorithm>
#include <cmath>

namespace linhai {

// ============================================================================
// ContractRiskEvaluator 实现
// ============================================================================

ContractRiskEvaluator::ContractRiskEvaluator() {
    reset();
}

void ContractRiskEvaluator::record_chi(int target_player) {
    if (target_player >= 0 && target_player < 4) {
        meld_counts[target_player].chi_count++;
        meld_counts[target_player].total_count++;
    }
}

void ContractRiskEvaluator::record_pon(int target_player) {
    if (target_player >= 0 && target_player < 4) {
        meld_counts[target_player].pon_count++;
        meld_counts[target_player].total_count++;
    }
}

MeldCount ContractRiskEvaluator::get_meld_count(int target_player) const {
    if (target_player >= 0 && target_player < 4) {
        return meld_counts[target_player];
    }
    return MeldCount();
}

float ContractRiskEvaluator::evaluate_meld_risk(int target_player) const {
    if (target_player < 0 || target_player >= 4) {
        return 0.0f;
    }

    int total = meld_counts[target_player].total_count;

    // 三摊承包规则：吃碰3次就承包
    if (total == 0) {
        return 0.0f;        // 无风险
    } else if (total == 1) {
        return 1000.0f;     // 低风险
    } else if (total == 2) {
        return 5000.0f;     // 高风险！再吃碰一次就承包
    } else {
        return 10000.0f;    // 极高风险！已经承包
    }
}

float ContractRiskEvaluator::evaluate_discard_contract_risk(int hai, int target_player) const {
    // 评估打出某张牌被某玩家吃碰的承包风险
    float base_risk = evaluate_meld_risk(target_player);

    if (base_risk < 5000.0f) {
        // 风险不高，可以打
        return base_risk * 0.3f;  // 降低权重
    }

    // 已经吃碰2次，风险很高
    // 需要评估这张牌被吃碰的概率

    // 简化：假设所有牌被吃碰的概率相同
    // 实际应该根据牌型、对手手牌等判断
    const float MELD_PROBABILITY = 0.2f;  // 假设20%概率被吃碰

    return base_risk * MELD_PROBABILITY;
}

void ContractRiskEvaluator::reset() {
    for (int i = 0; i < 4; i++) {
        meld_counts[i] = MeldCount();
    }
}

// ============================================================================
// MultiRonRiskEvaluator 实现
// ============================================================================

float MultiRonRiskEvaluator::estimate_tenpai_probability(const Player_State& player_state) const {
    // 基于多个因素估算听牌概率

    // 因素1: 手牌数（副露后手牌变少）
    int tehai_count = 0;
    for (int hai = 1; hai < 38; hai++) {
        tehai_count += player_state.tehai[hai];
    }

    // 因素2: 副露数
    int fuuro_count = player_state.fuuro.size();

    // 因素3: 立直状态
    bool is_reach = player_state.reach_declared;

    // 因素4: 河牌数（打出的牌越多，越可能听牌）
    int kawa_count = player_state.kawa.size();

    // 计算听牌概率
    float probability = 0.0f;

    // 立直必然听牌
    if (is_reach) {
        return 1.0f;
    }

    // 副露多，听牌概率高
    if (fuuro_count >= 3) {
        probability += 0.6f;
    } else if (fuuro_count >= 2) {
        probability += 0.4f;
    } else if (fuuro_count >= 1) {
        probability += 0.2f;
    }

    // 河牌多，听牌概率高
    if (kawa_count >= 15) {
        probability += 0.3f;
    } else if (kawa_count >= 10) {
        probability += 0.2f;
    } else if (kawa_count >= 5) {
        probability += 0.1f;
    }

    // 限制在0-1之间
    probability = std::min(1.0f, probability);
    probability = std::max(0.0f, probability);

    return probability;
}

float MultiRonRiskEvaluator::evaluate_single_ron_risk(
    int hai,
    const Player_State& player_state
) const {
    // 评估某张牌被单个玩家胡的风险

    // 听牌概率
    float tenpai_prob = estimate_tenpai_probability(player_state);

    if (tenpai_prob < 0.1f) {
        return 0.0f;  // 基本不可能听牌
    }

    // 假设听牌后，每张牌被胡的概率
    // 简化：假设听3张牌，每张牌1/3概率
    const float RON_PROBABILITY = 0.33f;

    // 基础风险分数
    const float BASE_RISK = 3000.0f;

    float risk = BASE_RISK * tenpai_prob * RON_PROBABILITY;

    return risk;
}

float MultiRonRiskEvaluator::evaluate_multi_ron_risk(
    int hai,
    const std::vector<Player_State>& player_states,
    int current_player
) const {
    float total_risk = 0.0f;
    int tenpai_count = 0;

    // 评估每个对手的风险
    for (int i = 0; i < player_states.size(); i++) {
        if (i == current_player) {
            continue;  // 跳过自己
        }

        float single_risk = evaluate_single_ron_risk(hai, player_states[i]);
        total_risk += single_risk;

        if (estimate_tenpai_probability(player_states[i]) > 0.5f) {
            tenpai_count++;
        }
    }

    // 一炮多响：如果多家听牌，风险倍增
    if (tenpai_count >= 2) {
        total_risk *= 1.5f;  // 风险增加50%
    }
    if (tenpai_count >= 3) {
        total_risk *= 2.0f;  // 风险翻倍
    }

    return total_risk;
}

// ============================================================================
// RiskEvaluator 实现
// ============================================================================

RiskEvaluator::RiskEvaluator() {
}

void RiskEvaluator::record_chi(int target_player) {
    contract_evaluator.record_chi(target_player);
}

void RiskEvaluator::record_pon(int target_player) {
    contract_evaluator.record_pon(target_player);
}

RiskValue RiskEvaluator::evaluate_discard_risk(
    int hai,
    const std::vector<Player_State>& player_states,
    int current_player
) {
    RiskValue result;

    // 评估三摊承包风险
    result.contract_risk = 0.0f;
    for (int i = 0; i < player_states.size(); i++) {
        if (i == current_player) {
            continue;
        }
        float risk = contract_evaluator.evaluate_discard_contract_risk(hai, i);
        result.contract_risk += risk;
    }

    // 评估一炮多响风险
    result.multi_ron_risk = multi_ron_evaluator.evaluate_multi_ron_risk(
        hai, player_states, current_player
    );

    // 总风险
    result.total_risk = result.contract_risk + result.multi_ron_risk;

    return result;
}

void RiskEvaluator::reset() {
    contract_evaluator.reset();
}

const ContractRiskEvaluator& RiskEvaluator::get_contract_evaluator() const {
    return contract_evaluator;
}

} // namespace linhai
