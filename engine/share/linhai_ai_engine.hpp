#ifndef LINHAI_AI_ENGINE_HPP
#define LINHAI_AI_ENGINE_HPP

#include "types.hpp"
#include "linhai_shanten_v4.hpp"
#include "linhai_grab_charge.hpp"
#include "linhai_bonus.hpp"
#include "linhai_risk.hpp"
#include <vector>
#include <string>

namespace linhai {

// 打牌推荐结果
struct DiscardRecommendation {
    int hai;                    // 推荐打出的牌
    float score;                // 综合评分
    int shanten;                // 打出后的向听数
    float bonus_value;          // 保留的价值
    float risk_value;           // 打出的风险
    std::string explanation;    // 推荐理由
};

// 响应动作推荐结果
struct ResponseRecommendation {
    std::string action;         // chi / peng / gang / pass
    int target_hai;             // 对手打出的牌
    std::vector<int> consumed;  // 吃/碰/杠消耗的暗手牌
    int discard_hai;            // 动作后建议打出的牌
    float score;                // 综合评分
    int shanten;                // 动作后建议打牌后的向听数
    float bonus_value;          // 动作后的综合价值
    float risk_value;           // 动作带来的风险
    std::string explanation;    // 推荐理由

    ResponseRecommendation()
        : action("pass"),
          target_hai(0),
          discard_hai(0),
          score(0.0f),
          shanten(8),
          bonus_value(0.0f),
          risk_value(0.0f) {}
};

// 游戏状态（完整信息）
struct GameState {
    // 当前玩家信息
    int current_player;         // 当前玩家ID（0-3）
    Hai_Array tehai;            // 手牌
    Fuuro_Vector fuuro;         // 副露
    int jikaze;                 // 自风（0=东, 1=南, 2=西, 3=北）
    bool has_kan;               // 是否有杠

    // 所有玩家状态
    std::vector<Player_State> all_players;

    // 游戏进度
    int wall_remaining;         // 剩余牌数
    int round;                  // 局数

    GameState() : current_player(0), jikaze(0), has_kan(false),
                  wall_remaining(70), round(0) {
        tehai = {0};
        all_players.resize(4);
    }
};

// AI决策引擎配置
struct AIConfig {
    // 权重配置
    float shanten_weight;       // 向听数权重（分/向听）
    float bonus_weight;         // 价值权重（倍率）
    float risk_weight;          // 风险权重（倍率）

    // 策略配置
    bool aggressive_mode;       // 进攻模式（优先降低向听数）
    bool defensive_mode;        // 防守模式（优先避免风险）

    // 默认配置
    AIConfig() :
        shanten_weight(5000.0f),
        bonus_weight(1.0f),
        risk_weight(1.0f),
        aggressive_mode(false),
        defensive_mode(false) {}
};

// 临海麻将AI决策引擎
class LinHaiAIEngine {
private:
    AIConfig config;
    RiskEvaluator risk_evaluator;

    // 评估打出某张牌后的向听数
    int evaluate_shanten_after_discard(const Hai_Array& tehai, int discard_hai);

    // 评估保留某张牌的价值
    float evaluate_keep_value(
        const Hai_Array& tehai,
        int keep_hai,
        int jikaze,
        bool has_kan,
        bool is_tenpai
    );

    // 评估打出某张牌的风险
    float evaluate_discard_risk(
        int discard_hai,
        const GameState& state
    );

    // 计算综合评分
    float calculate_total_score(
        int discard_hai,
        const GameState& state
    );

    // 生成推荐理由
    std::string generate_explanation(
        int discard_hai,
        int shanten,
        float bonus_value,
        float risk_value,
        float total_score
    );

    // 计算考虑副露后的有效向听数
    int evaluate_effective_shanten(const Hai_Array& tehai, int fuuro_count);

    // 生成可吃组合（返回需要从手牌消耗的两张牌）
    std::vector<std::vector<int>> generate_chi_combinations(const Hai_Array& tehai, int target_hai);

    // 模拟碰牌后的状态
    GameState simulate_peng(const GameState& state, int target_hai);

    // 模拟吃牌后的状态
    GameState simulate_chi(const GameState& state, int target_hai, const std::vector<int>& consumed);

    // 模拟明杠后的状态（包含简化补牌）
    GameState simulate_ming_gang(const GameState& state, int target_hai);

    // 找到杠后最优虚拟补牌
    int select_virtual_replacement_tile(const GameState& state);

    // 评估 pass 动作
    ResponseRecommendation evaluate_pass_response(const GameState& state, int target_hai, int from_player);

    // 评估副露动作
    ResponseRecommendation evaluate_open_response(
        const GameState& original_state,
        const GameState& simulated_state,
        const std::string& action,
        int target_hai,
        const std::vector<int>& consumed,
        int from_player
    );

    // 响应动作的整体威胁评估
    float estimate_response_threat(const GameState& state, int from_player);

    // 吃碰杠后牌形估值
    int estimate_pair_count(const Hai_Array& tehai);
    int estimate_taatsu_count(const Hai_Array& tehai);
    int estimate_isolated_count(const Hai_Array& tehai);
    float evaluate_shape_value(const Hai_Array& tehai);

public:
    LinHaiAIEngine();
    explicit LinHaiAIEngine(const AIConfig& cfg);

    // 推荐打牌
    // 返回所有可能打牌的评分，按评分排序
    std::vector<DiscardRecommendation> recommend_discard(const GameState& state);

    // 推荐最佳打牌
    DiscardRecommendation recommend_best_discard(const GameState& state);

    // 推荐响应动作（吃/碰/杠/过）
    std::vector<ResponseRecommendation> recommend_response(
        const GameState& state,
        int target_hai,
        int from_player,
        const std::vector<std::string>& available_actions
    );

    // 推荐最佳响应动作
    ResponseRecommendation recommend_best_response(
        const GameState& state,
        int target_hai,
        int from_player,
        const std::vector<std::string>& available_actions
    );

    // 记录吃碰操作（用于承包风险追踪）
    void record_chi(int target_player);
    void record_pon(int target_player);

    // 重置状态（新局开始）
    void reset();

    // 获取/设置配置
    const AIConfig& get_config() const;
    void set_config(const AIConfig& cfg);

    // 设置策略模式
    void set_aggressive_mode(bool enabled);
    void set_defensive_mode(bool enabled);
};

// 辅助函数：从手牌中获取所有可能打出的牌
std::vector<int> get_possible_discards(const Hai_Array& tehai);

// 辅助函数：打印推荐结果
void print_recommendation(const DiscardRecommendation& rec);
void print_all_recommendations(const std::vector<DiscardRecommendation>& recs);

} // namespace linhai

#endif
