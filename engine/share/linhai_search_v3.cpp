#include "linhai_search_v3.hpp"

#include <algorithm>
#include <fstream>
#include <set>

namespace linhai {

namespace {

constexpr float kImmediateWeight = 0.45f;
constexpr float kFutureWeight = 0.55f;

bool file_exists(const std::string& path) {
    std::ifstream fin(path.c_str());
    return fin.good();
}

float json_number_or_default(const json11::Json& json, double default_value = 0.0) {
    return static_cast<float>(json.is_number() ? json.number_value() : default_value);
}

int choose_draw_depth(const SearchConfig& cfg, int shanten) {
    if (shanten <= cfg.near_ready_shanten_threshold) {
        return std::max(cfg.max_self_draw_depth, cfg.deep_depth_for_near_ready);
    }
    return cfg.max_self_draw_depth;
}

bool is_safe_against_discards(const std::vector<int>& discards, int hai) {
    return std::find(discards.begin(), discards.end(), hai) != discards.end();
}

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

std::vector<std::vector<int>> enumerate_chi_patterns(const Hai_Array& tehai, int target_hai) {
    std::vector<std::vector<int>> patterns;
    if (target_hai >= 30 || !is_linhai_valid_tile(target_hai)) {
        return patterns;
    }

    const int rank = target_hai % 10;
    const int base = target_hai - rank;
    auto try_add = [&](int c1, int c2) {
        if (c1 <= 0 || c1 >= 38 || c2 <= 0 || c2 >= 38) {
            return;
        }
        if (!is_linhai_valid_tile(c1) || !is_linhai_valid_tile(c2)) {
            return;
        }
        if (tehai[c1] <= 0 || tehai[c2] <= 0) {
            return;
        }
        patterns.push_back({c1, c2});
    };

    if (rank >= 3) {
        try_add(base + rank - 2, base + rank - 1);
    }
    if (rank >= 2 && rank <= 8) {
        try_add(base + rank - 1, base + rank + 1);
    }
    if (rank <= 7) {
        try_add(base + rank + 1, base + rank + 2);
    }
    return patterns;
}

bool is_suited_tile(int hai) {
    return (hai >= 1 && hai <= 9) || (hai >= 11 && hai <= 19) || (hai >= 21 && hai <= 29);
}

int tile_rank(int hai) {
    if (hai >= 1 && hai <= 9) {
        return hai;
    }
    if (hai >= 11 && hai <= 19) {
        return hai - 10;
    }
    if (hai >= 21 && hai <= 29) {
        return hai - 20;
    }
    return 0;
}

int tile_connectivity(const Hai_Array& tehai, int hai) {
    int score = 0;
    score += tehai[hai] * 3;
    if (!is_suited_tile(hai)) {
        return score;
    }
    if (hai - 1 >= 0) {
        score += tehai[hai - 1] * 2;
    }
    if (hai + 1 < 38) {
        score += tehai[hai + 1] * 2;
    }
    if (tile_rank(hai) >= 3 && hai - 2 >= 0) {
        score += tehai[hai - 2];
    }
    if (tile_rank(hai) <= 7 && hai + 2 < 38) {
        score += tehai[hai + 2];
    }
    return score;
}

int estimate_ukeire_after_discard(const Hai_Array& tehai, const Hai_Array& remaining) {
    int total = 0;
    for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
        int hai = LINHAI_VALID_TILES[i];
        if (remaining[hai] <= 0) {
            continue;
        }
        int support = tile_connectivity(tehai, hai);
        if (support > 0) {
            total += remaining[hai];
        }
    }
    return total;
}

int estimate_fast_shanten(const Hai_Array& tehai) {
    int meld_like = 0;
    int pair_like = 0;
    for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
        int hai = LINHAI_VALID_TILES[i];
        if (tehai[hai] >= 3) {
            meld_like += 2;
        } else if (tehai[hai] == 2) {
            pair_like += 1;
        }
        if (is_suited_tile(hai) && tehai[hai] > 0) {
            if (hai + 1 < 38 && tehai[hai + 1] > 0) {
                meld_like += 1;
            }
            if (hai + 2 < 38 && tehai[hai + 2] > 0) {
                meld_like += 1;
            }
        }
    }
    int estimate = 6 - std::min(4, meld_like / 2) - std::min(1, pair_like);
    return std::max(0, estimate);
}

bool is_wind_tile(int hai) {
    return hai >= 31 && hai <= 34;
}

bool is_dragon_tile(int hai) {
    return hai >= 35 && hai <= 37;
}

// A-2a: stable tile-id -> human-readable code map. The names here MUST match
// exactly the names emitted by tools/extract_canonical_states.py and
// PER_TILE_FEATURE_ORDER, otherwise the model weights will be silently
// misaligned between training and inference.
const char* tile_code_for_feature(int hai) {
    switch (hai) {
        case 1:  return "1w"; case 2:  return "2w"; case 3:  return "3w";
        case 4:  return "4w"; case 5:  return "5w"; case 6:  return "6w";
        case 7:  return "7w"; case 8:  return "8w"; case 9:  return "9w";
        case 21: return "1t"; case 22: return "2t"; case 23: return "3t";
        case 24: return "4t"; case 25: return "5t"; case 26: return "6t";
        case 27: return "7t"; case 28: return "8t"; case 29: return "9t";
        case 31: return "east"; case 32: return "south"; case 33: return "west"; case 34: return "north";
        case 35: return "white"; case 36: return "green"; case 37: return "red";
        default: return nullptr;
    }
}

} // namespace

void CanonicalGameState::refresh_counts() {
    visible_tiles = get_linhai_visible_tiles(game_state);
    remaining_counts = get_linhai_remaining_pool(game_state);
    for (int hai : opponent_discards) {
        if (hai > 0 && hai < 38) {
            visible_tiles[hai] += 1;
            remaining_counts[hai] = std::max(0, remaining_counts[hai] - 1);
        }
    }
}

LinhaiSearchEngineV3::LinhaiSearchEngineV3() : config_(), model_bundle_(), last_search_(), fallback_engine_() {}

void LinhaiSearchEngineV3::reset_search_cache() {
    future_cache_.clear();
    discard_cache_.clear();
    cache_hits_ = 0;
    deadline_enabled_ = false;
    node_budget_limit_ = -1;
    search_nodes_ = 0;
    search_truncated_ = false;
    truncate_reason_.clear();
}

void LinhaiSearchEngineV3::begin_search_budget(int budget_ms, int node_budget) {
    search_truncated_ = false;
    truncate_reason_.clear();
    deadline_enabled_ = budget_ms >= 0;
    node_budget_limit_ = node_budget;
    search_nodes_ = 0;
    if (!deadline_enabled_) {
        return;
    }
    search_deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
}

void LinhaiSearchEngineV3::add_search_nodes(int count) {
    if (count <= 0) {
        return;
    }
    search_nodes_ += count;
}

bool LinhaiSearchEngineV3::is_search_budget_exceeded() {
    if (search_truncated_) {
        return true;
    }
    if (node_budget_limit_ >= 0 && search_nodes_ > node_budget_limit_) {
        search_truncated_ = true;
        truncate_reason_ = "node_budget_exceeded";
        return true;
    }
    if (!deadline_enabled_) {
        return false;
    }
    if (std::chrono::steady_clock::now() <= search_deadline_) {
        return false;
    }
    search_truncated_ = true;
    truncate_reason_ = "time_budget_exceeded";
    return true;
}

std::size_t LinhaiSearchEngineV3::make_state_cache_key(
    const CanonicalGameState& state,
    int depth,
    std::size_t prefix
) const {
    std::size_t seed = prefix;
    hash_combine(seed, static_cast<std::size_t>(depth));
    hash_combine(seed, static_cast<std::size_t>(state.game_state.wall_remaining));
    hash_combine(seed, static_cast<std::size_t>(state.game_state.jikaze));
    hash_combine(seed, static_cast<std::size_t>(state.can_win ? 1 : 0));
    hash_combine(seed, static_cast<std::size_t>(state.target_hai));
    hash_combine(seed, static_cast<std::size_t>(state.from_player));
    hash_combine(seed, static_cast<std::size_t>(state.tree_active ? 1 : 0));
    hash_combine(seed, static_cast<std::size_t>(state.grab_charge_active ? 1 : 0));
    hash_combine(seed, static_cast<std::size_t>(state.grab_charge_hits));
    hash_combine(seed, static_cast<std::size_t>(state.grab_charge_limit));
    hash_combine(seed, static_cast<std::size_t>(state.contract_target_count));
    hash_combine(seed, static_cast<std::size_t>(state.contract_counter));
    hash_combine(seed, static_cast<std::size_t>(state.opponent_meld_count));
    hash_combine(seed, static_cast<std::size_t>(state.opponent_discard_count));
    for (int hai = 1; hai < 38; hai++) {
        hash_combine(seed, static_cast<std::size_t>(state.game_state.tehai[hai]));
    }
    for (int hai = 1; hai < 38; hai++) {
        hash_combine(seed, static_cast<std::size_t>(state.remaining_counts[hai]));
    }
    return seed;
}

bool LinhaiSearchEngineV3::load_model_bundle(const std::string& version_dir) {
    model_bundle_.version_dir = version_dir;
    model_bundle_.loaded = false;
    model_bundle_.reason.clear();
    agari_model_ = V3LinearModel();
    tenpai_model_ = V3LinearModel();
    houjuu_model_ = V3LinearModel();
    betaori_model_ = V3LinearModel();
    tsumo_num_model_ = V3LinearModel();
    ryukyoku_model_ = V3LinearModel();

    std::string params_dir = version_dir;
    if (!params_dir.empty() && params_dir.back() != '/') {
        params_dir += "/";
    }

    bool fallback_loaded = false;
    if (file_exists(params_dir + "agari_prob/linhai/agari_para1.txt")) {
        fallback_loaded = fallback_engine_.load_params(params_dir) && fallback_engine_.params_loaded();
    }

    const bool agari_loaded = load_v3_model(params_dir + "v3/agari_prob/model.json", agari_model_);
    const bool tenpai_loaded = load_v3_model(params_dir + "v3/tenpai_prob/model.json", tenpai_model_);
    const bool houjuu_loaded = load_v3_model(params_dir + "v3/houjuu_prob/model.json", houjuu_model_);
    const bool betaori_loaded = load_v3_model(params_dir + "v3/betaori/model.json", betaori_model_);
    const bool tsumo_num_loaded = load_v3_model(params_dir + "v3/tsumo_num/model.json", tsumo_num_model_);
    const bool ryukyoku_loaded = load_v3_model(params_dir + "v3/ryukyoku_prob/model.json", ryukyoku_model_);

    const int loaded_count =
        static_cast<int>(agari_loaded) +
        static_cast<int>(tenpai_loaded) +
        static_cast<int>(houjuu_loaded) +
        static_cast<int>(betaori_loaded) +
        static_cast<int>(tsumo_num_loaded) +
        static_cast<int>(ryukyoku_loaded);

    model_bundle_.loaded = loaded_count > 0;
    if (model_bundle_.loaded) {
        model_bundle_.reason = loaded_count == 6 ? "ok" : "ok_partial";
    } else if (fallback_loaded) {
        model_bundle_.reason = "missing_v3_models";
    } else {
        model_bundle_.reason = "params_not_loaded";
    }
    return model_bundle_.loaded;
}

bool LinhaiSearchEngineV3::load_v3_model(const std::string& path, V3LinearModel& out_model) {
    if (!file_exists(path)) {
        return false;
    }

    const json11::Json json = load_json_from_file(path);
    if (!json.is_object()) {
        return false;
    }

    const auto& object = json.object_items();
    const auto feature_names_it = object.find("feature_names");
    const auto feature_means_it = object.find("feature_means");
    const auto feature_stds_it = object.find("feature_stds");
    const auto weights_it = object.find("weights");
    if (
        feature_names_it == object.end() ||
        feature_means_it == object.end() ||
        feature_stds_it == object.end() ||
        weights_it == object.end() ||
        !feature_names_it->second.is_array() ||
        !weights_it->second.is_array()
    ) {
        return false;
    }

    V3LinearModel model;
    model.loaded = true;
    model.task = object.count("task") ? object.at("task").string_value() : "";
    model.label_key = object.count("label_key") ? object.at("label_key").string_value() : "";
    model.intercept = json_number_or_default(object.count("intercept") ? object.at("intercept") : json11::Json());
    model.logistic = !object.count("model_type") || object.at("model_type").string_value() == "logistic_regression";

    for (const auto& name_json : feature_names_it->second.array_items()) {
        model.feature_names.push_back(name_json.string_value());
    }

    if (feature_means_it->second.is_object()) {
        const auto& means = feature_means_it->second.object_items();
        for (const auto& name : model.feature_names) {
            const auto it = means.find(name);
            model.feature_means.push_back(it == means.end() ? 0.0f : json_number_or_default(it->second));
        }
    }

    if (feature_stds_it->second.is_object()) {
        const auto& stds = feature_stds_it->second.object_items();
        for (const auto& name : model.feature_names) {
            const auto it = stds.find(name);
            float value = it == stds.end() ? 1.0f : json_number_or_default(it->second, 1.0);
            model.feature_stds.push_back(value == 0.0f ? 1.0f : value);
        }
    }

    for (const auto& weight_json : weights_it->second.array_items()) {
        model.weights.push_back(json_number_or_default(weight_json));
    }

    if (
        model.feature_names.empty() ||
        model.feature_names.size() != model.weights.size() ||
        model.feature_names.size() != model.feature_means.size() ||
        model.feature_names.size() != model.feature_stds.size()
    ) {
        return false;
    }

    out_model = model;
    return true;
}

std::unordered_map<std::string, float> LinhaiSearchEngineV3::build_state_features(const CanonicalGameState& state) const {
    std::unordered_map<std::string, float> features;
    int current_discard_count = 0;
    int current_meld_count = static_cast<int>(state.game_state.fuuro.size());
    if (
        state.game_state.current_player >= 0 &&
        state.game_state.current_player < static_cast<int>(state.game_state.all_players.size())
    ) {
        const Player_State& current_player = state.game_state.all_players[state.game_state.current_player];
        current_discard_count = static_cast<int>(current_player.kawa.size());
        current_meld_count = static_cast<int>(current_player.fuuro.size());
    }

    int hand_size = 0;
    int distinct_tile_count = 0;
    int pair_count = 0;
    int triplet_count = 0;
    int quad_count = 0;
    int suited_count = 0;
    int honor_count = 0;
    int terminal_count = 0;
    int simple_count = 0;
    int wind_count = 0;
    int dragon_count = 0;
    int remaining_total = 0;
    for (int hai = 1; hai < 38; hai++) {
        const int count = state.game_state.tehai[hai];
        if (count > 0) {
            hand_size += count;
            distinct_tile_count += 1;
            if (count >= 2) pair_count += 1;
            if (count >= 3) triplet_count += 1;
            if (count >= 4) quad_count += 1;
            if (is_suited_tile(hai)) {
                suited_count += count;
                const int rank = tile_rank(hai);
                if (rank == 1 || rank == 9) terminal_count += count;
                else simple_count += count;
            } else {
                honor_count += count;
                if (is_wind_tile(hai)) wind_count += count;
                if (is_dragon_tile(hai)) dragon_count += count;
            }
        }
        remaining_total += state.remaining_counts[hai];
    }

    const auto has_action = [&](const char* action) {
        return std::find(state.available_actions.begin(), state.available_actions.end(), action) != state.available_actions.end();
    };
    int target_rank = 0;
    bool target_is_honor = false;
    bool target_is_terminal = false;
    bool target_is_white = false;
    int target_in_hand_count = 0;
    if (state.target_hai > 0 && state.target_hai < 38 && is_linhai_valid_tile(state.target_hai)) {
        target_in_hand_count = state.game_state.tehai[state.target_hai];
        target_is_honor = !is_suited_tile(state.target_hai);
        if (is_suited_tile(state.target_hai)) {
            target_rank = tile_rank(state.target_hai);
            target_is_terminal = target_rank == 1 || target_rank == 9;
        }
        target_is_white = state.target_hai == 35;
    }

    features["wall_remaining"] = static_cast<float>(state.game_state.wall_remaining);
    features["from_player"] = static_cast<float>(state.from_player);
    features["has_target_hai"] = state.target_hai > 0 ? 1.0f : 0.0f;
    features["target_in_hand_count"] = static_cast<float>(target_in_hand_count);
    features["target_rank"] = static_cast<float>(target_rank);
    features["target_is_honor"] = target_is_honor ? 1.0f : 0.0f;
    features["target_is_terminal"] = target_is_terminal ? 1.0f : 0.0f;
    features["target_is_white"] = target_is_white ? 1.0f : 0.0f;
    features["hand_size"] = static_cast<float>(hand_size);
    features["meld_count"] = static_cast<float>(current_meld_count);
    features["discard_count"] = static_cast<float>(current_discard_count);
    features["distinct_tile_count"] = static_cast<float>(distinct_tile_count);
    features["pair_count"] = static_cast<float>(pair_count);
    features["triplet_count"] = static_cast<float>(triplet_count);
    features["quad_count"] = static_cast<float>(quad_count);
    features["white_tiles_in_hand"] = static_cast<float>(state.game_state.tehai[35]);
    features["suited_count"] = static_cast<float>(suited_count);
    features["honor_count"] = static_cast<float>(honor_count);
    features["terminal_count"] = static_cast<float>(terminal_count);
    features["simple_count"] = static_cast<float>(simple_count);
    features["wind_count"] = static_cast<float>(wind_count);
    features["dragon_count"] = static_cast<float>(dragon_count);
    features["tree_active"] = state.tree_active ? 1.0f : 0.0f;
    features["grab_charge_active"] = state.grab_charge_active ? 1.0f : 0.0f;
    features["grab_charge_hits"] = static_cast<float>(state.grab_charge_hits);
    features["grab_charge_limit"] = static_cast<float>(state.grab_charge_limit);
    features["contract_target_count"] = static_cast<float>(state.contract_target_count);
    features["contract_counter"] = static_cast<float>(state.contract_counter);
    features["opponent_meld_count"] = static_cast<float>(state.opponent_meld_count);
    features["opponent_discard_count"] = static_cast<float>(state.opponent_discard_count);
    features["remaining_total"] = static_cast<float>(remaining_total);
    features["remaining_white"] = static_cast<float>(state.remaining_counts[35]);
    features["can_win"] = state.can_win ? 1.0f : 0.0f;
    features["available_action_count"] = static_cast<float>(state.available_actions.size());
    features["can_pass"] = has_action("pass") || has_action("guo") ? 1.0f : 0.0f;
    features["can_peng"] = has_action("peng") ? 1.0f : 0.0f;
    features["can_chi"] = has_action("chi") ? 1.0f : 0.0f;
    features["can_gang"] = has_action("gang") ? 1.0f : 0.0f;
    features["can_hu"] = has_action("hu") ? 1.0f : 0.0f;

    // A-2a: per-tile counts (hand / remaining-pool / opponent-discards).
    // These names mirror the Python trainer's PER_TILE_FEATURE_ORDER exactly,
    // so the weights learned offline line up with the features computed here.
    std::array<int, 38> opp_discard_counts = {0};
    for (const int opp_hai : state.opponent_discards) {
        if (opp_hai > 0 && opp_hai < 38) {
            opp_discard_counts[opp_hai] += 1;
        }
    }
    for (int hai = 1; hai < 38; hai++) {
        const char* code = tile_code_for_feature(hai);
        if (code == nullptr) {
            continue;
        }
        features[std::string("hand_t_") + code] = static_cast<float>(state.game_state.tehai[hai]);
        features[std::string("remain_t_") + code] = static_cast<float>(state.remaining_counts[hai]);
        features[std::string("opp_disc_t_") + code] = static_cast<float>(opp_discard_counts[hai]);
    }

    return features;
}

void LinhaiSearchEngineV3::add_candidate_features(
    std::unordered_map<std::string, float>& features,
    const SearchCandidate& candidate,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    features["candidate_shanten"] = static_cast<float>(candidate.shanten);
    features["candidate_ukeire"] = static_cast<float>(candidate.ukeire);
    features["candidate_total_ev"] = candidate.total_ev;
    features["candidate_agari_prob"] = candidate.agari_prob;
    features["candidate_tenpai_prob"] = candidate.tenpai_prob;
    features["candidate_houjuu_prob"] = candidate.houjuu_prob;
    features["candidate_betaori_prob"] = candidate.betaori_prob;
    features["candidate_tsumo_num"] = candidate.tsumo_num;
    features["candidate_ryukyoku_prob"] = candidate.ryukyoku_prob;
    features["candidate_defense_score"] = candidate.defense_score;
    features["candidate_hai"] = static_cast<float>(candidate.hai);
    features["candidate_safe"] = safe ? 1.0f : 0.0f;
    features["candidate_action_pass"] = (action == "pass" || action == "guo") ? 1.0f : 0.0f;
    features["candidate_action_peng"] = action == "peng" ? 1.0f : 0.0f;
    features["candidate_action_chi"] = action == "chi" ? 1.0f : 0.0f;
    features["candidate_action_gang"] = action == "gang" ? 1.0f : 0.0f;
    features["candidate_action_hu"] = action == "hu" ? 1.0f : 0.0f;
    features["candidate_has_target_hai"] = target_hai > 0 ? 1.0f : 0.0f;
    features["candidate_target_hai"] = static_cast<float>(target_hai);
}

float LinhaiSearchEngineV3::predict_model(
    const V3LinearModel& model,
    const std::unordered_map<std::string, float>& features
) const {
    if (!model.loaded) {
        return 0.0f;
    }
    float linear = model.intercept;
    for (std::size_t i = 0; i < model.feature_names.size(); i++) {
        const auto feature_it = features.find(model.feature_names[i]);
        const float raw_value = feature_it == features.end() ? 0.0f : feature_it->second;
        const float std_value = model.feature_stds[i] == 0.0f ? 1.0f : model.feature_stds[i];
        const float normalized = (raw_value - model.feature_means[i]) / std_value;
        linear += model.weights[i] * normalized;
    }
    return model.logistic ? clamp01(1.0f / (1.0f + std::exp(-linear))) : linear;
}

float LinhaiSearchEngineV3::estimate_agari_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_agari_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!agari_model_.loaded) {
        return clamp01(heuristic_agari_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_agari_prob"] = clamp01(heuristic_agari_prob);
    const float model_prob = predict_model(agari_model_, features);
    return clamp01(model_prob);
}

float LinhaiSearchEngineV3::estimate_houjuu_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_houjuu_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!houjuu_model_.loaded) {
        return clamp01(heuristic_houjuu_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_houjuu_prob"] = clamp01(heuristic_houjuu_prob);
    const float model_prob = predict_model(houjuu_model_, features);
    return clamp01(model_prob);
}

float LinhaiSearchEngineV3::estimate_tenpai_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_tenpai_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!tenpai_model_.loaded) {
        return clamp01(heuristic_tenpai_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_tenpai_prob"] = clamp01(heuristic_tenpai_prob);
    const float model_prob = predict_model(tenpai_model_, features);
    return clamp01(model_prob);
}

float LinhaiSearchEngineV3::estimate_betaori_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_betaori_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!betaori_model_.loaded) {
        return clamp01(heuristic_betaori_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_betaori_prob"] = clamp01(heuristic_betaori_prob);
    const float model_prob = predict_model(betaori_model_, features);
    return clamp01(model_prob);
}

float LinhaiSearchEngineV3::estimate_tsumo_num(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_tsumo_num,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!tsumo_num_model_.loaded) {
        return std::max(0.0f, heuristic_tsumo_num);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_tsumo_num"] = std::max(0.0f, heuristic_tsumo_num);
    return std::max(0.0f, predict_model(tsumo_num_model_, features));
}

float LinhaiSearchEngineV3::estimate_ryukyoku_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_ryukyoku_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!ryukyoku_model_.loaded) {
        return clamp01(heuristic_ryukyoku_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_ryukyoku_prob"] = clamp01(heuristic_ryukyoku_prob);
    const float model_prob = predict_model(ryukyoku_model_, features);
    return clamp01(model_prob);
}

SearchResult LinhaiSearchEngineV3::make_fallback_result(const std::string& reason) const {
    SearchResult result;
    result.chosen_by = "fallback_v2";
    result.search_nodes = search_nodes_;
    result.truncated = reason == "time_budget_exceeded" || reason == "node_budget_exceeded";
    result.fallback_reason = reason;
    result.truncate_reason = result.truncated ? reason : "";
    return result;
}

void LinhaiSearchEngineV3::apply_context(CanonicalGameState& state) {
    state.refresh_counts();
    fallback_engine_.set_opponent_discards(state.opponent_discards);
    fallback_engine_.set_pass_hu(!state.can_win);
}

std::vector<SearchCandidate> LinhaiSearchEngineV3::build_discard_candidates(const CanonicalGameState& state) {
    std::vector<SearchCandidate> candidates;
    const GameState& game_state = state.game_state;
    Hai_Array remaining = state.remaining_counts;
    std::set<int> seen;
    for (int hai = 1; hai < 38; hai++) {
        if (game_state.tehai[hai] <= 0 || !is_linhai_valid_tile(hai) || seen.count(hai)) {
            continue;
        }
        seen.insert(hai);
        Hai_Array after = game_state.tehai;
        after[hai] -= 1;
        int shanten_after = estimate_fast_shanten(after);
        int ukeire = estimate_ukeire_after_discard(after, remaining);
        bool safe = is_safe_against_discards(state.opponent_discards, hai);
        int support = tile_connectivity(game_state.tehai, hai);
        float white_synergy_bonus = static_cast<float>(after[35]) * 260.0f;
        float tree_white_bonus = (state.tree_active && after[35] > 0) ? 320.0f : 0.0f;
        float grab_charge_penalty = 0.0f;
        if (state.grab_charge_active) {
            grab_charge_penalty = calc_grab_charge_loss(hai, game_state.tehai, game_state.jikaze, shanten_after == 0);
            if (state.grab_charge_limit > 0) {
                const float ratio =
                    static_cast<float>(std::min(state.grab_charge_hits, state.grab_charge_limit)) /
                    static_cast<float>(state.grab_charge_limit);
                grab_charge_penalty *= (1.0f + ratio);
            }
        }
        float contract_penalty = 0.0f;
        if (!safe && state.contract_target_count > 0) {
            contract_penalty =
                static_cast<float>(state.contract_target_count) * 140.0f +
                static_cast<float>(state.contract_counter) * 60.0f;
        }
        float opponent_pressure_penalty = safe ? 0.0f : static_cast<float>(state.opponent_meld_count) * 70.0f;
        SearchCandidate c;
        c.action = "discard";
        c.hai = hai;
        c.shanten = shanten_after;
        c.ukeire = ukeire;
        const float heuristic_houjuu_prob = clamp01(
            (safe ? 0.01f : 0.05f) +
            static_cast<float>(state.contract_target_count) * 0.012f +
            static_cast<float>(state.opponent_meld_count) * 0.008f
        );
        const float heuristic_tenpai_prob =
            shanten_after <= 0
                ? 0.92f
                : (shanten_after == 1
                       ? clamp01(0.25f + static_cast<float>(ukeire) * 0.01f)
                       : (shanten_after == 2
                              ? clamp01(0.08f + static_cast<float>(ukeire) * 0.004f)
                              : clamp01(0.02f + static_cast<float>(ukeire) * 0.0015f)));
        const float heuristic_betaori_prob = clamp01(
            (safe ? 0.55f : 0.08f) +
            static_cast<float>(state.contract_target_count) * 0.04f +
            static_cast<float>(state.opponent_meld_count) * 0.03f -
            static_cast<float>(ukeire) * 0.0015f
        );
        const float heuristic_tsumo_num = std::max(
            0.0f,
            static_cast<float>((std::max(0, shanten_after) + 1) * 2) +
                18.0f / static_cast<float>(std::max(3, ukeire + 2))
        );
        const float heuristic_ryukyoku_prob = clamp01(
            0.08f +
            static_cast<float>(std::max(0, shanten_after)) * 0.09f +
            (state.game_state.wall_remaining <= 18 ? 0.15f : 0.0f) -
            static_cast<float>(ukeire) * 0.0018f
        );
        c.houjuu_prob = estimate_houjuu_prob(state, c, heuristic_houjuu_prob, c.action, 0, safe);
        c.total_ev = static_cast<float>(-shanten_after * 1200 + ukeire * 12 - support * 45);
        c.total_ev += white_synergy_bonus + tree_white_bonus;
        c.total_ev -= grab_charge_penalty + contract_penalty + opponent_pressure_penalty;
        if (hai == 35) {
            c.total_ev -= 6000.0f;
        }
        if (safe) {
            c.total_ev += 120.0f;
        }
        c.tenpai_prob = estimate_tenpai_prob(state, c, heuristic_tenpai_prob, c.action, 0, safe);
        c.betaori_prob = estimate_betaori_prob(state, c, heuristic_betaori_prob, c.action, 0, safe);
        c.tsumo_num = estimate_tsumo_num(state, c, heuristic_tsumo_num, c.action, 0, safe);
        c.ryukyoku_prob = estimate_ryukyoku_prob(state, c, heuristic_ryukyoku_prob, c.action, 0, safe);
        c.defense_score = -(c.houjuu_prob * 8000.0f + contract_penalty + opponent_pressure_penalty) + c.betaori_prob * 900.0f;
        c.total_ev += c.tenpai_prob * 850.0f;
        c.total_ev += c.betaori_prob * (safe ? 160.0f : 40.0f);
        c.total_ev -= c.ryukyoku_prob * 320.0f;
        c.total_ev -= c.tsumo_num * 18.0f;
        c.explanation = std::to_string(shanten_after) + "向听, 受入" + std::to_string(ukeire);
        if (white_synergy_bonus > 0.0f) {
            c.explanation += ", 保留白板联动";
        }
        if (grab_charge_penalty > 0.0f) {
            c.explanation += ", 抓冲损失";
        }
        if (contract_penalty > 0.0f) {
            c.explanation += ", 承包风险";
        }
        candidates.push_back(c);
    }
    std::sort(candidates.begin(), candidates.end(), [](const SearchCandidate& a, const SearchCandidate& b) {
        return a.total_ev > b.total_ev;
    });
    return candidates;
}

float LinhaiSearchEngineV3::evaluate_future_draws(CanonicalGameState state, int depth, int& nodes_expanded) {
    const std::size_t cache_key = make_state_cache_key(state, depth, 17U);
    auto future_it = future_cache_.find(cache_key);
    if (future_it != future_cache_.end()) {
        cache_hits_ += 1;
        return future_it->second;
    }

    if (is_search_budget_exceeded()) {
        auto recs = build_discard_candidates(state);
        const float timeout_value = recs.empty() ? 0.0f : recs.front().total_ev;
        future_cache_[cache_key] = timeout_value;
        return timeout_value;
    }

    if (depth <= 0 || state.game_state.wall_remaining <= 0) {
        auto recs = build_discard_candidates(state);
        const float terminal_value = recs.empty() ? 0.0f : recs.front().total_ev;
        future_cache_[cache_key] = terminal_value;
        return terminal_value;
    }

    state.refresh_counts();
    int total_remaining = 0;
    for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
        total_remaining += state.remaining_counts[LINHAI_VALID_TILES[i]];
    }
    if (total_remaining <= 0) {
        auto recs = build_discard_candidates(state);
        const float empty_pool_value = recs.empty() ? 0.0f : recs.front().total_ev;
        future_cache_[cache_key] = empty_pool_value;
        return empty_pool_value;
    }

    float weighted = 0.0f;
    int beam = depth > 1 ? config_.beam_width_after_draw : config_.beam_width_far_shanten;
    int used = 0;
    for (int i = 0; i < LINHAI_VALID_TILE_COUNT && used < beam; i++) {
        if (is_search_budget_exceeded()) {
            break;
        }
        int hai = LINHAI_VALID_TILES[i];
        int remain = state.remaining_counts[hai];
        if (remain <= 0) {
            continue;
        }

        CanonicalGameState next = state;
        next.game_state.tehai[hai] += 1;
        next.white_tiles_in_hand = next.game_state.tehai[35];
        next.game_state.wall_remaining = std::max(0, next.game_state.wall_remaining - 1);
        next.remaining_counts[hai] = std::max(0, next.remaining_counts[hai] - 1);

        auto recs = build_discard_candidates(next);
        float future_best = recs.empty() ? 0.0f : recs.front().total_ev;
        weighted += (static_cast<float>(remain) / static_cast<float>(total_remaining)) * future_best;
        add_search_nodes(1);
        nodes_expanded += 1;
        used += 1;
    }
    future_cache_[cache_key] = weighted;
    return weighted;
}

SearchResult LinhaiSearchEngineV3::search_discard_once(CanonicalGameState& state, int depth) {
    const std::size_t cache_key = make_state_cache_key(state, depth, 29U);
    auto discard_it = discard_cache_.find(cache_key);
    if (discard_it != discard_cache_.end()) {
        cache_hits_ += 1;
        return discard_it->second;
    }

    SearchResult result;
    auto candidates = build_discard_candidates(state);
    const int root_candidates_total = static_cast<int>(candidates.size());
    if (candidates.empty()) {
        return make_fallback_result("no_candidates");
    }
    int root_beam = std::max(config_.beam_width_after_draw, config_.beam_width_far_shanten);
    root_beam = std::max(4, std::min(root_beam, 8));
    if (static_cast<int>(candidates.size()) > root_beam) {
        candidates.resize(root_beam);
    }

    int nodes_expanded = 0;
    int root_candidates_evaluated = 0;
    for (auto& candidate : candidates) {
        if (is_search_budget_exceeded()) {
            break;
        }
        CanonicalGameState next = state;
        if (candidate.hai <= 0 || next.game_state.tehai[candidate.hai] <= 0) {
            continue;
        }
        next.game_state.tehai[candidate.hai] -= 1;
        next.white_tiles_in_hand = next.game_state.tehai[35];
        next.game_state.wall_remaining = std::max(0, next.game_state.wall_remaining);
        float future_ev = evaluate_future_draws(next, std::max(0, depth - 1), nodes_expanded);
        candidate.total_ev = candidate.total_ev * kImmediateWeight + future_ev * kFutureWeight;
        candidate.search_depth = depth;
        const float heuristic_agari_prob = clamp01(0.02f * static_cast<float>(candidate.ukeire));
        candidate.agari_prob = estimate_agari_prob(next, candidate, heuristic_agari_prob, candidate.action, 0, false);
        candidate.total_ev += candidate.agari_prob * 2800.0f;
        root_candidates_evaluated += 1;
    }

    std::sort(candidates.begin(), candidates.end(), [](const SearchCandidate& a, const SearchCandidate& b) {
        return a.total_ev > b.total_ev;
    });

    const SearchCandidate& best = candidates.front();
    result.action = best.action;
    result.hai = best.hai;
    result.total_ev = best.total_ev;
    result.agari_prob = best.agari_prob;
    result.tenpai_prob = best.tenpai_prob;
    result.houjuu_prob = best.houjuu_prob;
    result.betaori_prob = best.betaori_prob;
    result.tsumo_num = best.tsumo_num;
    result.ryukyoku_prob = best.ryukyoku_prob;
    result.defense_score = best.defense_score;
    result.shanten = best.shanten;
    result.ukeire = best.ukeire;
    result.search_depth = depth;
    result.nodes_expanded = nodes_expanded;
    result.search_nodes = search_nodes_;
    result.root_candidates_total = root_candidates_total;
    result.root_candidates_evaluated = root_candidates_evaluated;
    result.cache_hits = 0;
    result.truncated = search_truncated_;
    result.chosen_by = "search";
    result.fallback_reason = search_truncated_ ? truncate_reason_ : "";
    result.truncate_reason = search_truncated_ ? truncate_reason_ : "";
    result.candidate_scores = candidates;
    discard_cache_[cache_key] = result;
    return result;
}

std::vector<SearchCandidate> LinhaiSearchEngineV3::build_response_candidates(const CanonicalGameState& state, int& nodes_expanded) {
    std::vector<SearchCandidate> candidates;
    const Hai_Array& tehai = state.game_state.tehai;
    int current_shanten = estimate_fast_shanten(tehai);
    int response_depth = choose_draw_depth(config_, current_shanten);

    auto append_followup_discard = [&](
        const std::string& action,
        CanonicalGameState next_state,
        float action_bonus,
        float action_penalty,
        const std::string& explanation_prefix
    ) {
        next_state.refresh_counts();
        SearchResult follow_up = search_discard_once(next_state, response_depth);
        if (follow_up.candidate_scores.empty() || follow_up.action != "discard") {
            return;
        }
        add_search_nodes(1);
        nodes_expanded += 1 + follow_up.nodes_expanded;

        SearchCandidate c;
        c.action = action;
        c.hai = follow_up.hai;
        c.target_hai = state.target_hai;
        c.total_ev = follow_up.total_ev + action_bonus - action_penalty;
        c.agari_prob = estimate_agari_prob(next_state, c, follow_up.agari_prob, action, state.target_hai, false);
        c.houjuu_prob = estimate_houjuu_prob(next_state, c, follow_up.houjuu_prob, action, state.target_hai, false);
        c.tenpai_prob = estimate_tenpai_prob(next_state, c, follow_up.tenpai_prob, action, state.target_hai, false);
        c.betaori_prob = estimate_betaori_prob(next_state, c, follow_up.betaori_prob, action, state.target_hai, false);
        c.tsumo_num = estimate_tsumo_num(next_state, c, follow_up.tsumo_num, action, state.target_hai, false);
        c.ryukyoku_prob = estimate_ryukyoku_prob(next_state, c, follow_up.ryukyoku_prob, action, state.target_hai, false);
        c.defense_score = follow_up.defense_score - action_penalty + (c.betaori_prob - follow_up.betaori_prob) * 250.0f;
        c.total_ev += (c.agari_prob - follow_up.agari_prob) * 2200.0f;
        c.total_ev -= (c.houjuu_prob - follow_up.houjuu_prob) * 2800.0f;
        c.total_ev += (c.tenpai_prob - follow_up.tenpai_prob) * 600.0f;
        c.total_ev += (c.betaori_prob - follow_up.betaori_prob) * 250.0f;
        c.total_ev -= (c.tsumo_num - follow_up.tsumo_num) * 10.0f;
        c.total_ev -= (c.ryukyoku_prob - follow_up.ryukyoku_prob) * 180.0f;
        c.shanten = follow_up.shanten;
        c.ukeire = follow_up.ukeire;
        c.search_depth = response_depth;
        c.explanation =
            explanation_prefix + "后打" + hai_int_to_str(follow_up.hai) +
            "，" + std::to_string(follow_up.shanten) + "向听";
        if (action_bonus > 0.0f) {
            c.explanation += ", 有动作收益";
        }
        if (action_penalty > 0.0f) {
            c.explanation += ", 有动作代价";
        }
        candidates.push_back(c);
    };

    auto append_draw_chance_followup = [&](
        const std::string& action,
        CanonicalGameState next_state,
        float action_bonus,
        float action_penalty,
        const std::string& explanation_prefix
    ) {
        next_state.refresh_counts();
        std::vector<std::pair<int, int>> draw_candidates;
        int total_remaining = 0;
        for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
            if (is_search_budget_exceeded()) {
                break;
            }
            const int hai = LINHAI_VALID_TILES[i];
            const int remain = next_state.remaining_counts[hai];
            if (remain <= 0) {
                continue;
            }
            total_remaining += remain;
            draw_candidates.push_back({hai, remain});
        }
        if (draw_candidates.empty() || total_remaining <= 0) {
            return;
        }

        std::sort(draw_candidates.begin(), draw_candidates.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            if (a.second != b.second) {
                return a.second > b.second;
            }
            return a.first < b.first;
        });

        const int beam = std::max(1, config_.beam_width_after_draw);
        if (static_cast<int>(draw_candidates.size()) > beam) {
            draw_candidates.resize(beam);
        }

        float weighted_total_ev = 0.0f;
        float weighted_agari_prob = 0.0f;
        float weighted_houjuu_prob = 0.0f;
        float weighted_tenpai_prob = 0.0f;
        float weighted_betaori_prob = 0.0f;
        float weighted_tsumo_num = 0.0f;
        float weighted_ryukyoku_prob = 0.0f;
        float weighted_defense_score = 0.0f;
        float used_probability = 0.0f;
        std::array<float, 38> discard_scores = {0.0f};

        for (const auto& draw_candidate : draw_candidates) {
            if (is_search_budget_exceeded()) {
                break;
            }
            const int draw_hai = draw_candidate.first;
            const int remain = draw_candidate.second;
            const float probability = static_cast<float>(remain) / static_cast<float>(total_remaining);
            add_search_nodes(1);
            nodes_expanded += 1;

            CanonicalGameState drawn_state = next_state;
            drawn_state.game_state.tehai[draw_hai] += 1;
            drawn_state.white_tiles_in_hand = drawn_state.game_state.tehai[35];
            drawn_state.game_state.wall_remaining = std::max(0, drawn_state.game_state.wall_remaining - 1);
            drawn_state.remaining_counts[draw_hai] = std::max(0, drawn_state.remaining_counts[draw_hai] - 1);

            SearchResult follow_up = search_discard_once(drawn_state, response_depth);
            if (follow_up.candidate_scores.empty() || follow_up.action != "discard") {
                continue;
            }
            nodes_expanded += follow_up.nodes_expanded;

            weighted_total_ev += probability * follow_up.total_ev;
            weighted_agari_prob += probability * follow_up.agari_prob;
            weighted_houjuu_prob += probability * follow_up.houjuu_prob;
            weighted_tenpai_prob += probability * follow_up.tenpai_prob;
            weighted_betaori_prob += probability * follow_up.betaori_prob;
            weighted_tsumo_num += probability * follow_up.tsumo_num;
            weighted_ryukyoku_prob += probability * follow_up.ryukyoku_prob;
            weighted_defense_score += probability * follow_up.defense_score;
            used_probability += probability;
            if (follow_up.hai > 0 && follow_up.hai < 38) {
                discard_scores[follow_up.hai] += probability * follow_up.total_ev;
            }
        }

        if (used_probability <= 0.0f) {
            return;
        }

        int best_discard = 0;
        float best_discard_score = -1e30f;
        for (int hai = 1; hai < 38; hai++) {
            if (discard_scores[hai] > best_discard_score) {
                best_discard_score = discard_scores[hai];
                best_discard = hai;
            }
        }

        SearchCandidate c;
        c.action = action;
        c.hai = best_discard;
        c.target_hai = state.target_hai;
        c.total_ev = weighted_total_ev + action_bonus - action_penalty;
        c.agari_prob = estimate_agari_prob(next_state, c, weighted_agari_prob, action, state.target_hai, false);
        c.houjuu_prob = estimate_houjuu_prob(next_state, c, weighted_houjuu_prob, action, state.target_hai, false);
        c.tenpai_prob = estimate_tenpai_prob(next_state, c, weighted_tenpai_prob, action, state.target_hai, false);
        c.betaori_prob = estimate_betaori_prob(next_state, c, weighted_betaori_prob, action, state.target_hai, false);
        c.tsumo_num = estimate_tsumo_num(next_state, c, weighted_tsumo_num, action, state.target_hai, false);
        c.ryukyoku_prob = estimate_ryukyoku_prob(next_state, c, weighted_ryukyoku_prob, action, state.target_hai, false);
        c.defense_score = weighted_defense_score - action_penalty + (c.betaori_prob - weighted_betaori_prob) * 250.0f;
        c.total_ev += (c.agari_prob - weighted_agari_prob) * 2200.0f;
        c.total_ev -= (c.houjuu_prob - weighted_houjuu_prob) * 2800.0f;
        c.total_ev += (c.tenpai_prob - weighted_tenpai_prob) * 600.0f;
        c.total_ev += (c.betaori_prob - weighted_betaori_prob) * 250.0f;
        c.total_ev -= (c.tsumo_num - weighted_tsumo_num) * 10.0f;
        c.total_ev -= (c.ryukyoku_prob - weighted_ryukyoku_prob) * 180.0f;
        c.shanten = estimate_fast_shanten(next_state.game_state.tehai);
        c.search_depth = response_depth + 1;
        c.explanation = explanation_prefix;
        if (best_discard > 0) {
            c.explanation += "，优先打" + hai_int_to_str(best_discard);
        }
        if (action_bonus > 0.0f) {
            c.explanation += ", 有动作收益";
        }
        if (action_penalty > 0.0f) {
            c.explanation += ", 有动作代价";
        }
        candidates.push_back(c);
    };

    for (const auto& action : state.available_actions) {
        if (is_search_budget_exceeded()) {
            break;
        }
        if (action == "hu") {
            add_search_nodes(1);
            nodes_expanded += 1;
            SearchCandidate c;
            c.action = action;
            c.target_hai = state.target_hai;
            c.shanten = current_shanten;
            c.total_ev = state.can_win ? 1000000.0f : -1000000.0f;
            c.agari_prob = estimate_agari_prob(state, c, state.can_win ? 1.0f : 0.0f, action, state.target_hai, true);
            c.houjuu_prob = estimate_houjuu_prob(state, c, 0.0f, action, state.target_hai, true);
            c.tenpai_prob = estimate_tenpai_prob(state, c, state.can_win ? 1.0f : 0.0f, action, state.target_hai, true);
            c.betaori_prob = estimate_betaori_prob(state, c, 1.0f, action, state.target_hai, true);
            c.tsumo_num = estimate_tsumo_num(state, c, 0.0f, action, state.target_hai, true);
            c.ryukyoku_prob = estimate_ryukyoku_prob(state, c, 0.0f, action, state.target_hai, true);
            c.defense_score = c.betaori_prob * 1000.0f;
            c.explanation = state.can_win ? "胡牌优先" : "过胡不胡限制";
            c.search_depth = 0;
            candidates.push_back(c);
            continue;
        }

        if (action == "pass" || action == "guo") {
            append_draw_chance_followup(
                "pass",
                state,
                (state.grab_charge_active ? 120.0f : 0.0f) +
                    static_cast<float>(state.contract_target_count) * 40.0f,
                0.0f,
                "保留门前后按下一摸递推"
            );
            continue;
        }

        const float contract_penalty =
            (action == "peng" || action == "chi" || action == "gang") && state.contract_target_count > 0
                ? static_cast<float>(state.contract_target_count) * 110.0f +
                      static_cast<float>(state.contract_counter) * 45.0f
                : 0.0f;
        const float grab_bonus =
            (action == "peng" || action == "gang") && state.grab_charge_active
                ? static_cast<float>(std::max(0, state.grab_charge_limit - state.grab_charge_hits)) * 18.0f
                : 0.0f;

        if (action == "peng" && tehai[state.target_hai] >= 2) {
            CanonicalGameState next_state = state;
            next_state.game_state.tehai[state.target_hai] = std::max(0, next_state.game_state.tehai[state.target_hai] - 2);
            Fuuro_Elem fuuro;
            fuuro.type = FT_PON;
            fuuro.hai = state.target_hai;
            fuuro.consumed = {state.target_hai, state.target_hai};
            fuuro.target_relative = 1;
            next_state.game_state.fuuro.push_back(fuuro);
            append_followup_discard("peng", next_state, 400.0f + grab_bonus, contract_penalty, "碰");
            continue;
        }

        if (action == "chi") {
            auto patterns = enumerate_chi_patterns(tehai, state.target_hai);
            for (const auto& consumed : patterns) {
                CanonicalGameState next_state = state;
                next_state.game_state.tehai[consumed[0]] -= 1;
                next_state.game_state.tehai[consumed[1]] -= 1;
                Fuuro_Elem fuuro;
                fuuro.type = FT_CHI;
                fuuro.hai = state.target_hai;
                fuuro.consumed = consumed;
                fuuro.target_relative = 1;
                next_state.game_state.fuuro.push_back(fuuro);
                append_followup_discard("chi", next_state, 250.0f, contract_penalty, "吃");
            }
            continue;
        }

        if (action == "gang" && tehai[state.target_hai] >= 3) {
            CanonicalGameState next_state = state;
            next_state.game_state.tehai[state.target_hai] = std::max(0, next_state.game_state.tehai[state.target_hai] - 3);
            next_state.game_state.has_kan = true;
            Fuuro_Elem fuuro;
            fuuro.type = FT_DAIMINKAN;
            fuuro.hai = state.target_hai;
            fuuro.consumed = {state.target_hai, state.target_hai, state.target_hai};
            fuuro.target_relative = 1;
            next_state.game_state.fuuro.push_back(fuuro);
            append_draw_chance_followup("gang", next_state, 300.0f + grab_bonus, contract_penalty, "明杠后按岭上摸牌递推");
            continue;
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const SearchCandidate& a, const SearchCandidate& b) {
        return a.total_ev > b.total_ev;
    });
    return candidates;
}

SearchResult LinhaiSearchEngineV3::recommend_discard_v3(CanonicalGameState state) {
    reset_search_cache();
    begin_search_budget(config_.time_budget_ms_discard, config_.max_nodes_discard);
    apply_context(state);
    int shanten = calc_linhai_shanten(state.game_state.tehai);
    int depth = choose_draw_depth(config_, shanten);

    if (!model_bundle_.loaded) {
        SearchResult fallback = search_discard_once(state, depth);
        fallback.chosen_by = "fallback_v2";
        fallback.search_nodes = search_nodes_;
        fallback.configured_time_budget_ms = config_.time_budget_ms_discard;
        fallback.configured_node_budget = config_.max_nodes_discard;
        fallback.truncated = search_truncated_;
        fallback.fallback_reason = search_truncated_
            ? truncate_reason_
            : (model_bundle_.reason.empty() ? "model_bundle_not_loaded" : model_bundle_.reason);
        fallback.truncate_reason = search_truncated_ ? truncate_reason_ : "";
        fallback.cache_hits = cache_hits_;
        last_search_ = fallback;
        return fallback;
    }

    SearchResult result = search_discard_once(state, depth);
    result.search_nodes = search_nodes_;
    result.configured_time_budget_ms = config_.time_budget_ms_discard;
    result.configured_node_budget = config_.max_nodes_discard;
    result.truncated = search_truncated_;
    if (search_truncated_) {
        result.fallback_reason = truncate_reason_;
    }
    result.truncate_reason = search_truncated_ ? truncate_reason_ : "";
    result.cache_hits = cache_hits_;
    last_search_ = result;
    return result;
}

SearchResult LinhaiSearchEngineV3::recommend_response_v3(CanonicalGameState state) {
    reset_search_cache();
    begin_search_budget(config_.time_budget_ms_response, config_.max_nodes_response);
    apply_context(state);

    SearchResult result;
    int nodes_expanded = 0;
    auto candidates = build_response_candidates(state, nodes_expanded);
    if (candidates.empty()) {
        result = make_fallback_result(search_truncated_ ? truncate_reason_ : "no_response_candidates");
        result.nodes_expanded = nodes_expanded;
        result.search_nodes = search_nodes_;
        result.root_candidates_total = 0;
        result.root_candidates_evaluated = 0;
        result.configured_time_budget_ms = config_.time_budget_ms_response;
        result.configured_node_budget = config_.max_nodes_response;
        result.cache_hits = cache_hits_;
        last_search_ = result;
        return result;
    }

    std::sort(candidates.begin(), candidates.end(), [](const SearchCandidate& a, const SearchCandidate& b) {
        return a.total_ev > b.total_ev;
    });
    const SearchCandidate& best = candidates.front();
    result.action = best.action;
    result.hai = best.hai;
    result.target_hai = best.target_hai;
    result.total_ev = best.total_ev;
    result.agari_prob = best.agari_prob;
    result.tenpai_prob = best.tenpai_prob;
    result.houjuu_prob = best.houjuu_prob;
    result.betaori_prob = best.betaori_prob;
    result.tsumo_num = best.tsumo_num;
    result.ryukyoku_prob = best.ryukyoku_prob;
    result.defense_score = best.defense_score;
    result.shanten = best.shanten;
    result.ukeire = best.ukeire;
    result.search_depth = choose_draw_depth(config_, best.shanten);
    result.nodes_expanded = nodes_expanded;
    result.search_nodes = search_nodes_;
    result.root_candidates_total = static_cast<int>(candidates.size());
    result.root_candidates_evaluated = static_cast<int>(candidates.size());
    result.configured_time_budget_ms = config_.time_budget_ms_response;
    result.configured_node_budget = config_.max_nodes_response;
    result.cache_hits = cache_hits_;
    result.truncated = search_truncated_;
    result.chosen_by = model_bundle_.loaded ? "search" : "fallback_v2";
    result.fallback_reason = search_truncated_
        ? truncate_reason_
        : (model_bundle_.loaded ? "" : (model_bundle_.reason.empty() ? "model_bundle_not_loaded" : model_bundle_.reason));
    result.truncate_reason = search_truncated_ ? truncate_reason_ : "";
    result.candidate_scores = candidates;
    last_search_ = result;
    return result;
}

} // namespace linhai
