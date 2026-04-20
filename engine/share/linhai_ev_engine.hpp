#ifndef LINHAI_EV_ENGINE_HPP
#define LINHAI_EV_ENGINE_HPP

// 临海麻将期望值引擎 V2
// 基于 Akochan 的概率框架，适配两花色2人临海麻将
// 替代原始 linhai_ai_engine 的简单线性评分

#include "types.hpp"
#include "linhai_ai_engine.hpp"
#include "linhai_game_adapter.hpp"
#include "linhai_shanten_v4.hpp"
#include "linhai_grab_charge.hpp"
#include "linhai_bonus.hpp"
#include "linhai_risk.hpp"
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace linhai {

// ============================================================================
// 概率模型参数 (从 params/ 加载的回归系数)
// ============================================================================

struct ProbParams {
    // agari_prob: 和了概率回归参数 (按巡目索引, 最多18巡)
    float agari_w[18][4];       // w[turn][feature]

    // tenpai_prob: 听牌概率参数
    float tenpai_w[18][5];      // w[turn][feature]

    // betaori: 防守参数
    float betaori_w[18][3];     // w[turn][feature]

    // houjuu: 放铳概率
    float houjuu_w[18][3];      // w[turn][feature]

    // tsumo_num: 剩余摸牌次数
    float tsumo_num_w[18][4];   // w[turn][feature]

    // ryukyoku: 流局概率
    float ryukyoku_w[18][4];    // w[turn][feature]

    bool loaded;

    ProbParams() : loaded(false) {
        memset(agari_w, 0, sizeof(agari_w));
        memset(tenpai_w, 0, sizeof(tenpai_w));
        memset(betaori_w, 0, sizeof(betaori_w));
        memset(houjuu_w, 0, sizeof(houjuu_w));
        memset(tsumo_num_w, 0, sizeof(tsumo_num_w));
        memset(ryukyoku_w, 0, sizeof(ryukyoku_w));
    }
};

// ============================================================================
// 手牌效率分析结果
// ============================================================================

struct EfficiencyResult {
    int shanten;                     // 向听数
    int ukeire;                      // 受入数 (能改善向听的牌种类数)
    int ukeire_count;                // 受入枚数 (考虑剩余张数)
    int win_ukeire_count;            // 直接胡牌的张数 (shanten=0时有效)
    float agari_prob_solo;           // 一人麻将和了概率 (不考虑对手)
    float tenpai_prob;               // 到达听牌的概率
    std::vector<int> effective_tiles; // 有效进张牌
};

// ============================================================================
// 单张牌的详细EV分析
// ============================================================================

struct TileEV {
    int hai;                         // 牌编码
    float ev_total;                  // 综合期望值
    float ev_offense;                // 进攻期望值 (和了概率 × 和了价值)
    float ev_defense;                // 防守期望值 (betaori)
    float ev_linhai_bonus;           // 临海特色加值 (抓冲/白板/承包)
    float risk_houjuu;               // 打出此牌的放铳概率
    int shanten_after;               // 打出后的向听数
    int ukeire_after;                // 打出后的受入枚数
    std::string explanation;
};

// ============================================================================
// LinHai EV Engine V2
// ============================================================================

class LinHaiEVEngine {
private:
    AIConfig config;
    RiskEvaluator risk_evaluator;
    ProbParams prob_params;
    std::string params_dir;
    std::vector<int> opponent_discards_;
    bool can_win_;  // 过胡不胡规则

    // ---- 私有方法 ----
    static float logistic(const float* w, const float* x, int dim);
    int resolve_opponent_index(const GameState& state) const;
    float calc_solo_agari_prob(int shanten, int ukeire_count, int remaining_turns, int pool_size) const;
    float calc_agari_prob(float solo_prob, int turn, float opponent_tenpai_prob) const;
    float estimate_opponent_tenpai_prob(const GameState& state) const;
    int estimate_remaining_turns(const GameState& state) const;
    float calc_ryukyoku_prob(int turn, float my_agari_prob, float opp_tenpai_prob) const;
    EfficiencyResult calc_efficiency(const Hai_Array& tehai, const Hai_Array& visible) const;
    EfficiencyResult calc_efficiency_after_discard(const Hai_Array& tehai, int discard_hai, const Hai_Array& visible) const;
    std::array<float, 38> calc_houjuu_probs(const GameState& state) const;
    float calc_betaori_ev(const Hai_Array& tehai, const std::array<float, 38>& houjuu_probs, float other_value, int ori_turns) const;
    float estimate_agari_value(const Hai_Array& tehai, int jikaze, bool has_kan) const;
    float calc_linhai_modifier(int discard_hai, const GameState& state);
    float calc_multi_turn_ev(const Hai_Array& tehai, const Hai_Array& visible, const GameState& state, int remaining_turns, float opp_tenpai_prob, int pool_size) const;
    std::string make_explanation(const TileEV& ev) const;

public:
    LinHaiEVEngine();
    explicit LinHaiEVEngine(const AIConfig& cfg);

    bool load_params(const std::string& dir = "params/");
    bool params_loaded() const { return prob_params.loaded; }

    // 核心 V2 接口
    std::vector<TileEV> recommend_discard_ev(const GameState& state);
    DiscardRecommendation recommend_best_discard_v2(const GameState& state);
    std::vector<DiscardRecommendation> recommend_discard_v2(const GameState& state);
    std::vector<ResponseRecommendation> recommend_response_v2(const GameState& state, int target_hai, int from_player, const std::vector<std::string>& available_actions);

    // 辅助接口
    void set_opponent_discards(const std::vector<int>& discards);
    void set_pass_hu(bool passed) { can_win_ = !passed; }
    bool get_can_win() const { return can_win_; }
    const AIConfig& get_config() const { return config; }
    void set_config(const AIConfig& cfg) { config = cfg; }
    void set_aggressive_mode(bool enabled);
    void set_defensive_mode(bool enabled);
    void record_chi(int target_player) { risk_evaluator.record_chi(target_player); }
    void record_pon(int target_player) { risk_evaluator.record_pon(target_player); }
    void reset() { risk_evaluator.reset(); opponent_discards_.clear(); can_win_ = true; }
};

} // namespace linhai

#endif
