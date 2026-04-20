#ifndef LINHAI_RISK_HPP
#define LINHAI_RISK_HPP

#include "types.hpp"
#include <vector>

namespace linhai {

// 三摊承包规则：
// 对同一家吃碰3次，该家胡牌你承包（支付全部分数）

// 一炮多响规则：
// 一张牌可以让多家胡牌，都要付钱

// 玩家吃碰统计
struct MeldCount {
    int chi_count;   // 吃的次数
    int pon_count;   // 碰的次数
    int total_count; // 总次数

    MeldCount() : chi_count(0), pon_count(0), total_count(0) {}
};

// 三摊承包风险评估器
class ContractRiskEvaluator {
private:
    // 记录对每个玩家的吃碰次数
    // meld_counts[target_player] = 对该玩家的吃碰次数
    MeldCount meld_counts[4];

public:
    ContractRiskEvaluator();

    // 记录一次吃牌
    void record_chi(int target_player);

    // 记录一次碰牌
    void record_pon(int target_player);

    // 获取对某玩家的吃碰次数
    MeldCount get_meld_count(int target_player) const;

    // 评估对某玩家再次吃碰的风险
    // 返回：风险分数（0-10000，越高越危险）
    float evaluate_meld_risk(int target_player) const;

    // 评估打出某张牌被某玩家吃碰的承包风险
    float evaluate_discard_contract_risk(int hai, int target_player) const;

    // 重置统计（新局开始）
    void reset();
};

// 一炮多响风险评估器
class MultiRonRiskEvaluator {
public:
    // 评估打出某张牌被多家胡的风险
    // 参数：
    //   hai: 要打出的牌
    //   player_states: 所有玩家的状态
    //   current_player: 当前玩家ID
    // 返回：风险分数（0-10000，越高越危险）
    float evaluate_multi_ron_risk(
        int hai,
        const std::vector<Player_State>& player_states,
        int current_player
    ) const;

    // 评估某张牌被单个玩家胡的风险
    float evaluate_single_ron_risk(
        int hai,
        const Player_State& player_state
    ) const;

    // 估算某玩家的听牌概率
    // 基于：手牌数、副露数、河牌等信息
    float estimate_tenpai_probability(const Player_State& player_state) const;
};

// 综合风险评估
struct RiskValue {
    float contract_risk;      // 三摊承包风险
    float multi_ron_risk;     // 一炮多响风险
    float total_risk;         // 总风险
};

// 综合风险评估器
class RiskEvaluator {
private:
    ContractRiskEvaluator contract_evaluator;
    MultiRonRiskEvaluator multi_ron_evaluator;

public:
    RiskEvaluator();

    // 记录吃碰操作
    void record_chi(int target_player);
    void record_pon(int target_player);

    // 评估打出某张牌的综合风险
    RiskValue evaluate_discard_risk(
        int hai,
        const std::vector<Player_State>& player_states,
        int current_player
    );

    // 重置统计
    void reset();

    // 获取承包风险评估器（用于查询）
    const ContractRiskEvaluator& get_contract_evaluator() const;
};

} // namespace linhai

#endif
