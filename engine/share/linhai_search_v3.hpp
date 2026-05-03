#ifndef LINHAI_SEARCH_V3_HPP
#define LINHAI_SEARCH_V3_HPP

#include "linhai_ev_engine.hpp"
#include "linhai_game_adapter.hpp"
#include "linhai_score.hpp"
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>

namespace linhai {

struct SearchConfig {
    int max_self_draw_depth = 2;
    int deep_depth_for_near_ready = 3;
    int near_ready_shanten_threshold = 1;
    int beam_width_after_draw = 6;
    int beam_width_far_shanten = 4;
    bool enable_rollout = false;
    int discard_rollout_simulations = 24;
    int response_rollout_simulations = 16;
    float discard_rollout_gap_ratio = 0.05f;
    float response_rollout_gap_ratio = 0.08f;
    float discard_rollout_override_delta = 0.08f;
    float response_rollout_override_delta = 0.12f;
    int time_budget_ms_discard = 80;
    int time_budget_ms_response = 60;
    int max_nodes_discard = -1;
    int max_nodes_response = -1;
};

struct ModelBundle {
    std::string version_dir;
    bool loaded = false;
    std::string reason;
};

struct V3LinearModel {
    bool loaded = false;
    bool logistic = true;
    std::string task;
    std::string label_key;
    float intercept = 0.0f;
    std::vector<std::string> feature_names;
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    std::vector<float> weights;
};

// A-2b: GBDT (LightGBM) model bundle. One struct-of-arrays representation
// backs all trees of one head so the hot predict path is cache-friendly.
// The schema matches exactly what tools/train_model_stub.py emits when
// LightGBM is available (see `_flatten_lightgbm_tree`).
struct V3GBDTNode {
    int feat = -1;          // feature index into feature_names; -1 for leaf
    float thr = 0.0f;       // split threshold; x[feat] <= thr goes left
    int left = -1;
    int right = -1;
    float leaf_value = 0.0f;
};

struct V3GBDTModel {
    bool loaded = false;
    bool logistic = true;   // true iff objective == "binary"
    std::string task;
    std::string label_key;
    float init_score = 0.0f; // constant added to the sum of leaf values (LightGBM's implicit prior)
    std::vector<std::string> feature_names;
    // Flattened trees: trees[t] is the node list for the t-th booster tree.
    // Index 0 inside each list is always the root.
    std::vector<std::vector<V3GBDTNode>> trees;
};

struct SearchCandidate {
    std::string action = "discard";
    int hai = 0;
    int target_hai = 0;
    float total_ev = 0.0f;
    float agari_prob = 0.0f;
    float tenpai_prob = 0.0f;
    float houjuu_prob = 0.0f;
    float betaori_prob = 0.0f;
    float tsumo_num = 0.0f;
    float ryukyoku_prob = 0.0f;
    float defense_score = 0.0f;
    int shanten = 8;
    int ukeire = 0;
    int search_depth = 0;
    std::string explanation;
};

struct SearchResult {
    std::string action = "pass";
    int hai = 0;
    int target_hai = 0;
    float total_ev = 0.0f;
    float agari_prob = 0.0f;
    float tenpai_prob = 0.0f;
    float houjuu_prob = 0.0f;
    float betaori_prob = 0.0f;
    float tsumo_num = 0.0f;
    float ryukyoku_prob = 0.0f;
    float defense_score = 0.0f;
    int shanten = 8;
    int ukeire = 0;
    int search_depth = 0;
    int nodes_expanded = 0;
    int search_nodes = 0;
    int root_candidates_total = 0;
    int root_candidates_evaluated = 0;
    int configured_time_budget_ms = -1;
    int configured_node_budget = -1;
    int cache_hits = 0;
    bool truncated = false;
    std::string chosen_by = "search";
    std::string fallback_reason;
    std::string truncate_reason;
    std::vector<SearchCandidate> candidate_scores;
};

struct CanonicalGameState {
    GameState game_state;
    Hai_Array visible_tiles = {0};
    Hai_Array remaining_counts = {0};
    std::vector<int> opponent_discards;
    bool can_win = true;
    int target_hai = 0;
    int from_player = 0;
    std::vector<std::string> available_actions;
    int white_tiles_in_hand = 0;
    bool tree_active = false;
    bool grab_charge_active = false;
    int grab_charge_hits = 0;
    int grab_charge_limit = 0;
    int contract_target_count = 0;
    int contract_counter = 0;
    int opponent_meld_count = 0;
    int opponent_discard_count = 0;
    // Phase A spec §3.5.3: 副露压力辅助字段。这些是 compute_opp_pressure_score
    // 的内部依赖（**不进 build_state_features**，避免训练特征漂移）。Python 在
    // _build_canonical_state 时根据对手 melds/discards 的实际牌码内容填充。
    //   opponent_honor_triplets: 对手副露中字牌刻子数（碰/明杠/加杠且 hai>=31 && hai<=37）
    //   opp_white_meld_count:    对手副露含白板（hai==35）的副露条数（0 或 1+）
    int opponent_honor_triplets = 0;
    int opp_white_meld_count = 0;

    void refresh_counts();
};

class LinhaiSearchEngineV3 {
public:
    LinhaiSearchEngineV3();

    bool load_model_bundle(const std::string& version_dir);
    void set_search_config(const SearchConfig& cfg) { config_ = cfg; }
    const SearchConfig& get_search_config() const { return config_; }
    ModelBundle get_model_bundle() const { return model_bundle_; }
    SearchResult get_last_search_debug() const { return last_search_; }

    SearchResult recommend_discard_v3(CanonicalGameState state);
    SearchResult recommend_response_v3(CanonicalGameState state);

    // Phase A spec §3.5.3: 副露压力综合标量，0=无威胁，~1.5=清一色+字一色+抓冲全开。
    // A4 将在 EV 公式里以此动态加权 λ/μ；A3 仅暴露此 helper，无外部行为变化。
    float compute_opp_pressure_score(const CanonicalGameState& state) const;

private:
    SearchConfig config_;
    ModelBundle model_bundle_;
    SearchResult last_search_;
    LinHaiEVEngine fallback_engine_;
    V3LinearModel agari_model_;
    V3LinearModel tenpai_model_;
    V3LinearModel houjuu_model_;
    V3LinearModel betaori_model_;
    V3LinearModel tsumo_num_model_;
    V3LinearModel ryukyoku_model_;
    // A-2b: GBDT counterparts. Each head tries to load the GBDT bundle
    // first; if unavailable, the linear bundle above is used as a fallback.
    V3GBDTModel agari_gbdt_;
    V3GBDTModel tenpai_gbdt_;
    V3GBDTModel houjuu_gbdt_;
    V3GBDTModel betaori_gbdt_;
    V3GBDTModel tsumo_num_gbdt_;
    V3GBDTModel ryukyoku_gbdt_;
    // 验收 (spec §3.3): score-regression heads. 学的是带号冲数（E[my_chong] /
    // E[loss_chong]），由 train_model_stub.py --task agari_score / houjuu_score 产出。
    // 加载到这两个 head 后，search 时把 total_ev = agari_score - houjuu_score 作为
    // 弃牌排序主键（替代旧的 prob × 番数估算）。如果 score-head 未加载，回退到
    // 旧的 6-prob-head 路径，保证向后兼容。
    V3LinearModel agari_score_model_;
    V3LinearModel houjuu_score_model_;
    V3GBDTModel agari_score_gbdt_;
    V3GBDTModel houjuu_score_gbdt_;
    // Phase A spec §3.5: 评分/风险加权配置表。从 v3/score_table.json 加载，
    // 缺失时使用 ScoreTable 内的默认值（与 spec §3.5.1 一致：lambda_base=1.5,
    // mu_base=2.0, opp_meld_pressure_alpha=1.0...）。A4 在 build_discard_candidates
    // 中读取 ev_risk_weights / feature_weights 完成动态 λ/μ 加权。
    linhai_score::ScoreTable score_table_;
    boost::unordered_map<std::size_t, float> future_cache_;
    boost::unordered_map<std::size_t, SearchResult> discard_cache_;
    boost::unordered_map<std::size_t, int> shanten_cache_;
    int cache_hits_ = 0;
    std::chrono::steady_clock::time_point search_deadline_;
    bool deadline_enabled_ = false;
    int node_budget_limit_ = -1;
    int search_nodes_ = 0;
    bool search_truncated_ = false;
    std::string truncate_reason_;

    SearchResult make_fallback_result(const std::string& reason) const;
    int cached_shanten(const Hai_Array& tehai);
    // Fuuro-aware shanten: merges the fuuro tiles back into the tehai counts
    // via using_hai_array() so the standard shanten formula operates on the
    // full 13-tile (or 14-tile for post-draw) representation. Using the raw
    // 10/7/4-tile hand directly undercounts melds the player has already
    // locked in and produces systematically wrong shanten for any hand with
    // fuuro — the root cause of the engine undervaluing chi/pon tenpai.
    int cached_shanten(const Hai_Array& tehai, const Fuuro_Vector& fuuro);
    SearchResult search_discard_once(CanonicalGameState& state, int depth);
    float evaluate_future_draws(CanonicalGameState state, int depth, int& nodes_expanded);
    std::vector<SearchCandidate> build_discard_candidates(const CanonicalGameState& state);
    std::vector<SearchCandidate> build_response_candidates(const CanonicalGameState& state, int& nodes_expanded);
    void apply_context(CanonicalGameState& state);
    std::size_t make_state_cache_key(const CanonicalGameState& state, int depth, std::size_t prefix) const;
    void reset_search_cache();
    void begin_search_budget(int budget_ms, int node_budget);
    void add_search_nodes(int count);
    bool is_search_budget_exceeded();
    bool load_v3_model(const std::string& path, V3LinearModel& out_model);
    // A-2b: unified model loader that detects GBDT vs linear schema from the
    // JSON's `model_type` field and populates the correct output struct.
    // Returns true if either branch loaded successfully.
    bool load_v3_head(
        const std::string& path,
        V3LinearModel& linear_out,
        V3GBDTModel& gbdt_out
    );
    bool load_v3_gbdt(const std::string& path, V3GBDTModel& out_model);
    std::unordered_map<std::string, float> build_state_features(const CanonicalGameState& state) const;
    void add_candidate_features(
        std::unordered_map<std::string, float>& features,
        const SearchCandidate& candidate,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    float predict_model(const V3LinearModel& model, const std::unordered_map<std::string, float>& features) const;
    // A-2b: GBDT inference. Internally materializes x[] once from the feature
    // map using gbdt.feature_names, then walks every tree in O(depth) time.
    float predict_gbdt(const V3GBDTModel& model, const std::unordered_map<std::string, float>& features) const;
    // A-2b: unified head prediction. Uses GBDT if loaded, otherwise linear,
    // otherwise returns 0.0f (caller should already have short-circuited).
    float predict_head(
        const V3LinearModel& linear,
        const V3GBDTModel& gbdt,
        const std::unordered_map<std::string, float>& features
    ) const;
    float estimate_agari_prob(
        const CanonicalGameState& state,
        const SearchCandidate& candidate,
        float heuristic_agari_prob,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    float estimate_houjuu_prob(
        const CanonicalGameState& state,
        const SearchCandidate& candidate,
        float heuristic_houjuu_prob,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    float estimate_tenpai_prob(
        const CanonicalGameState& state,
        const SearchCandidate& candidate,
        float heuristic_tenpai_prob,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    float estimate_betaori_prob(
        const CanonicalGameState& state,
        const SearchCandidate& candidate,
        float heuristic_betaori_prob,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    float estimate_tsumo_num(
        const CanonicalGameState& state,
        const SearchCandidate& candidate,
        float heuristic_tsumo_num,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    float estimate_ryukyoku_prob(
        const CanonicalGameState& state,
        const SearchCandidate& candidate,
        float heuristic_ryukyoku_prob,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    // 验收 (spec §3.3): score-aware EV. 返回带号期望冲数（agari 正、houjuu 负）。
    // 仅当对应 score head 加载成功时返回有效值；否则返回 0（caller 应回退到
    // prob × heuristic 路径）。
    float estimate_agari_score(
        const CanonicalGameState& state,
        const SearchCandidate& candidate,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    float estimate_houjuu_score(
        const CanonicalGameState& state,
        const SearchCandidate& candidate,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    bool has_score_heads_loaded() const {
        return (agari_score_model_.loaded || agari_score_gbdt_.loaded)
            && (houjuu_score_model_.loaded || houjuu_score_gbdt_.loaded);
    }
};

} // namespace linhai

#endif
