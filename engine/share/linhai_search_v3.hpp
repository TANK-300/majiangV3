#ifndef LINHAI_SEARCH_V3_HPP
#define LINHAI_SEARCH_V3_HPP

#include "linhai_ev_engine.hpp"
#include "linhai_game_adapter.hpp"
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
    std::unordered_map<std::string, float> build_state_features(const CanonicalGameState& state) const;
    void add_candidate_features(
        std::unordered_map<std::string, float>& features,
        const SearchCandidate& candidate,
        const std::string& action,
        int target_hai,
        bool safe
    ) const;
    float predict_model(const V3LinearModel& model, const std::unordered_map<std::string, float>& features) const;
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
};

} // namespace linhai

#endif
