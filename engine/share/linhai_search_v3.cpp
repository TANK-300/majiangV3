#include "linhai_search_v3.hpp"
#include "linhai_shanten_v4.hpp"
#include "calc_shanten.hpp"

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

// Deprecated under A-1: kept only as a last-resort fallback wrapper.
// All in-search call sites now use LinhaiSearchEngineV3::cached_shanten(),
// which delegates to the exact calc_linhai_shanten() with memoization.
int estimate_fast_shanten(const Hai_Array& tehai) {
    return calc_linhai_shanten(tehai);
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
    // A-1 hotfix: shanten_cache_ is intentionally NOT cleared between searches.
    // The tehai->shanten map is state-of-the-world-independent (the answer
    // depends only on the 38 tile counts), so cache hits across consecutive
    // search calls are safe and drastically reduce cost for selfplay loops
    // (observed 100x speedup on hands with >= 2 whites). We cap the cache
    // in cached_shanten() to bound memory.
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
    // A-2b: reset GBDT models too. load_v3_head below decides per-head which
    // of the two gets populated based on the JSON's ``model_type`` field.
    agari_gbdt_ = V3GBDTModel();
    tenpai_gbdt_ = V3GBDTModel();
    houjuu_gbdt_ = V3GBDTModel();
    betaori_gbdt_ = V3GBDTModel();
    tsumo_num_gbdt_ = V3GBDTModel();
    ryukyoku_gbdt_ = V3GBDTModel();
    // 验收 score heads
    agari_score_model_ = V3LinearModel();
    houjuu_score_model_ = V3LinearModel();
    agari_score_gbdt_ = V3GBDTModel();
    houjuu_score_gbdt_ = V3GBDTModel();

    std::string params_dir = version_dir;
    if (!params_dir.empty() && params_dir.back() != '/') {
        params_dir += "/";
    }

    bool fallback_loaded = false;
    if (file_exists(params_dir + "agari_prob/linhai/agari_para1.txt")) {
        fallback_loaded = fallback_engine_.load_params(params_dir) && fallback_engine_.params_loaded();
    }

    const bool agari_loaded = load_v3_head(params_dir + "v3/agari_prob/model.json", agari_model_, agari_gbdt_);
    const bool tenpai_loaded = load_v3_head(params_dir + "v3/tenpai_prob/model.json", tenpai_model_, tenpai_gbdt_);
    const bool houjuu_loaded = load_v3_head(params_dir + "v3/houjuu_prob/model.json", houjuu_model_, houjuu_gbdt_);
    const bool betaori_loaded = load_v3_head(params_dir + "v3/betaori/model.json", betaori_model_, betaori_gbdt_);
    const bool tsumo_num_loaded = load_v3_head(params_dir + "v3/tsumo_num/model.json", tsumo_num_model_, tsumo_num_gbdt_);
    const bool ryukyoku_loaded = load_v3_head(params_dir + "v3/ryukyoku_prob/model.json", ryukyoku_model_, ryukyoku_gbdt_);
    // 验收 score heads（可选）：缺失不算错误，回退到 6-prob 路径
    (void)load_v3_head(params_dir + "v3/agari_score/model.json", agari_score_model_, agari_score_gbdt_);
    (void)load_v3_head(params_dir + "v3/houjuu_score/model.json", houjuu_score_model_, houjuu_score_gbdt_);

    // Phase A spec §3.5: 加载 v3/score_table.json（含 ev_risk_weights /
    // betaori_thresholds / feature_weights）。文件缺失 → 默认表（同 spec 默认）。
    score_table_ = linhai_score::load_score_table(params_dir + "v3/score_table.json");

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

// A-2b: dispatch on the JSON ``model_type`` field. The linear loader in
// ``load_v3_model`` is deliberately strict about its required fields
// (feature_stds, weights, ...). A GBDT-shaped JSON (emitted by
// tools/train_model_stub.py when LightGBM is used) lacks those fields on
// purpose, so stale deployments without PR-2 will safely fall back to
// heuristic. Here we route GBDT JSONs to the new loader instead.
bool LinhaiSearchEngineV3::load_v3_head(
    const std::string& path,
    V3LinearModel& linear_out,
    V3GBDTModel& gbdt_out
) {
    if (!file_exists(path)) {
        return false;
    }
    const json11::Json json = load_json_from_file(path);
    if (!json.is_object()) {
        return false;
    }
    const auto& object = json.object_items();
    std::string model_type;
    const auto mt_it = object.find("model_type");
    if (mt_it != object.end() && mt_it->second.is_string()) {
        model_type = mt_it->second.string_value();
    }
    if (model_type == "lightgbm_gbdt") {
        return load_v3_gbdt(path, gbdt_out);
    }
    if (model_type == "gbdt_unavailable" || model_type == "unavailable") {
        // train_model_stub.py emits these when the operator asked for GBDT
        // but the fitter couldn't produce one. Treat as "no model", which
        // cascades to the heuristic fallback at inference time.
        return false;
    }
    return load_v3_model(path, linear_out);
}

bool LinhaiSearchEngineV3::load_v3_gbdt(const std::string& path, V3GBDTModel& out_model) {
    if (!file_exists(path)) {
        return false;
    }
    const json11::Json json = load_json_from_file(path);
    if (!json.is_object()) {
        return false;
    }

    const auto& object = json.object_items();
    const auto feature_names_it = object.find("feature_names");
    const auto trees_it = object.find("trees");
    const auto objective_it = object.find("objective");
    if (
        feature_names_it == object.end() ||
        trees_it == object.end() ||
        !feature_names_it->second.is_array() ||
        !trees_it->second.is_array()
    ) {
        return false;
    }

    V3GBDTModel model;
    model.task = object.count("task") ? object.at("task").string_value() : "";
    model.label_key = object.count("label_key") ? object.at("label_key").string_value() : "";
    model.init_score = json_number_or_default(object.count("init_score") ? object.at("init_score") : json11::Json());
    if (objective_it != object.end() && objective_it->second.is_string()) {
        model.logistic = objective_it->second.string_value() == "binary";
    } else {
        // Default to logistic so binary heads keep the sigmoid squash even
        // when the trainer forgot to serialize the objective.
        model.logistic = true;
    }

    for (const auto& name_json : feature_names_it->second.array_items()) {
        model.feature_names.push_back(name_json.string_value());
    }
    if (model.feature_names.empty()) {
        return false;
    }
    const int feat_count = static_cast<int>(model.feature_names.size());

    for (const auto& tree_json : trees_it->second.array_items()) {
        if (!tree_json.is_object()) {
            return false;
        }
        const auto& tree_obj = tree_json.object_items();
        const auto nodes_it = tree_obj.find("nodes");
        if (nodes_it == tree_obj.end() || !nodes_it->second.is_array()) {
            return false;
        }
        std::vector<V3GBDTNode> nodes;
        nodes.reserve(nodes_it->second.array_items().size());
        for (const auto& node_json : nodes_it->second.array_items()) {
            if (!node_json.is_object()) {
                return false;
            }
            const auto& node_obj = node_json.object_items();
            V3GBDTNode node;
            node.feat = node_obj.count("feat") ? static_cast<int>(node_obj.at("feat").int_value()) : -1;
            node.thr = node_obj.count("thr") ? json_number_or_default(node_obj.at("thr")) : 0.0f;
            node.left = node_obj.count("left") ? static_cast<int>(node_obj.at("left").int_value()) : -1;
            node.right = node_obj.count("right") ? static_cast<int>(node_obj.at("right").int_value()) : -1;
            node.leaf_value = node_obj.count("leaf_value") ? json_number_or_default(node_obj.at("leaf_value")) : 0.0f;
            // Structural validation: if the JSON is malformed the predictor
            // would otherwise segfault or spin forever. Fail fast instead.
            if (node.feat >= feat_count) {
                return false;
            }
            if (node.feat >= 0 && (node.left < 0 || node.right < 0)) {
                return false;
            }
            nodes.push_back(node);
        }
        if (nodes.empty()) {
            return false;
        }
        // Cross-reference child indices after the whole list is built.
        const int node_count = static_cast<int>(nodes.size());
        for (const auto& node : nodes) {
            if (node.feat >= 0) {
                if (node.left >= node_count || node.right >= node_count) {
                    return false;
                }
            }
        }
        model.trees.push_back(std::move(nodes));
    }

    model.loaded = !model.trees.empty();
    if (!model.loaded) {
        return false;
    }
    out_model = std::move(model);
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
    // A-2c-3: also emits safety_t_<tile> flag (genbutsu) and aggregate
    // defensive signals for houjuu/betaori heads.
    std::array<int, 38> opp_discard_counts = {0};
    for (const int opp_hai : state.opponent_discards) {
        if (opp_hai > 0 && opp_hai < 38) {
            opp_discard_counts[opp_hai] += 1;
        }
    }
    float safe_in_hand_count = 0.0f;
    float raw_in_hand_count = 0.0f;
    int opp_max_tile_disc = 0;
    int opp_honor_disc_count = 0;
    int opp_terminal_disc_count = 0;
    int opp_middle_disc_count = 0;
    int opp_disc_total = 0;
    for (int hai = 1; hai < 38; hai++) {
        const char* code = tile_code_for_feature(hai);
        if (code == nullptr) {
            continue;
        }
        const float hand_c = static_cast<float>(state.game_state.tehai[hai]);
        const float remain_c = static_cast<float>(state.remaining_counts[hai]);
        const int opp_disc_c_i = opp_discard_counts[hai];
        const float opp_disc_c = static_cast<float>(opp_disc_c_i);
        features[std::string("hand_t_") + code] = hand_c;
        features[std::string("remain_t_") + code] = remain_c;
        features[std::string("opp_disc_t_") + code] = opp_disc_c;
        const float safety_flag = opp_disc_c > 0.0f ? 1.0f : 0.0f;
        features[std::string("safety_t_") + code] = safety_flag;
        if (hand_c > 0.0f) {
            if (safety_flag > 0.0f) {
                safe_in_hand_count += hand_c;
            } else if (remain_c > 0.0f) {
                raw_in_hand_count += hand_c;
            }
        }
        if (opp_disc_c_i > opp_max_tile_disc) {
            opp_max_tile_disc = opp_disc_c_i;
        }
        if (opp_disc_c_i > 0) {
            opp_disc_total += opp_disc_c_i;
            if (!is_suited_tile(hai)) {
                opp_honor_disc_count += opp_disc_c_i;
            } else {
                const int rank = tile_rank(hai);
                if (rank == 1 || rank == 9) {
                    opp_terminal_disc_count += opp_disc_c_i;
                } else if (rank == 4 || rank == 5 || rank == 6) {
                    opp_middle_disc_count += opp_disc_c_i;
                }
            }
        }
    }

    features["safe_in_hand_count"] = safe_in_hand_count;
    features["raw_in_hand_count"] = raw_in_hand_count;
    features["opp_honor_disc_count"] = static_cast<float>(opp_honor_disc_count);
    features["opp_terminal_disc_count"] = static_cast<float>(opp_terminal_disc_count);
    features["opp_middle_disc_count"] = static_cast<float>(opp_middle_disc_count);
    features["opp_max_tile_disc"] = static_cast<float>(opp_max_tile_disc);
    const int opp_meld_denom = state.opponent_meld_count > 0 ? state.opponent_meld_count : 1;
    features["opp_disc_per_meld"] = static_cast<float>(opp_disc_total) / static_cast<float>(opp_meld_denom);

    // 验收 (spec §4.4): per-tile suji (筋) + kabe (壁) safety signals.
    // 必须与 tools/extract_canonical_states.py::build_model_features 同步加。
    // tests/python/test_feature_parity.py 锁名字一致性。
    //
    // 筋 (suji): 对手已弃 4 → 1/7 安全；5 → 2/8；6 → 3/9。两个数色独立计算。
    // 壁 (kabe): 一张牌可见 ≥ 3 → 相邻数牌安全（不会被对手听）。
    std::array<bool, 38> suji_safe = {false};
    std::array<bool, 38> kabe_safe = {false};

    // suji
    auto check_suji = [&](int suit_base) {
        // suit_base: 0 for 1w-9w (hai 1..9), 20 for 1t-9t (hai 21..29).
        // 4 → 1/7
        if (opp_discard_counts[suit_base + 4] > 0) {
            suji_safe[suit_base + 1] = true;
            suji_safe[suit_base + 7] = true;
        }
        if (opp_discard_counts[suit_base + 5] > 0) {
            suji_safe[suit_base + 2] = true;
            suji_safe[suit_base + 8] = true;
        }
        if (opp_discard_counts[suit_base + 6] > 0) {
            suji_safe[suit_base + 3] = true;
            suji_safe[suit_base + 9] = true;
        }
    };
    check_suji(0);   // 1m..9m
    check_suji(20);  // 1t..9t (hai 21..29 means base = 20, 21=base+1)

    // kabe: visible >= 3 → adjacent tiles are safer
    // visible = total copies (4) - remaining_counts
    auto check_kabe = [&](int suit_base) {
        for (int r = 1; r <= 9; ++r) {
            int hai = suit_base + r;
            int visible = 4 - state.remaining_counts[hai];
            if (visible >= 3) {
                if (r > 1) kabe_safe[suit_base + (r - 1)] = true;
                if (r < 9) kabe_safe[suit_base + (r + 1)] = true;
            }
        }
    };
    check_kabe(0);   // 1m..9m
    check_kabe(20);  // 1t..9t

    for (int hai = 1; hai < 38; ++hai) {
        const char* code = tile_code_for_feature(hai);
        if (code == nullptr) continue;
        features[std::string("safety_suji_t_") + code] =
            suji_safe[hai] ? 1.0f : 0.0f;
        features[std::string("safety_kabe_t_") + code] =
            kabe_safe[hai] ? 1.0f : 0.0f;
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

float LinhaiSearchEngineV3::predict_gbdt(
    const V3GBDTModel& model,
    const std::unordered_map<std::string, float>& features
) const {
    if (!model.loaded) {
        return 0.0f;
    }
    // A-2b: materialize the feature vector once using the name->value map,
    // so subsequent tree walks are O(depth) pointer chases instead of O(n)
    // hashmap lookups. Missing features default to 0.0f, which matches the
    // training-time behavior (LightGBM sees 0 where features weren't set).
    const std::size_t feat_count = model.feature_names.size();
    std::vector<float> x(feat_count, 0.0f);
    for (std::size_t i = 0; i < feat_count; i++) {
        const auto it = features.find(model.feature_names[i]);
        if (it != features.end()) {
            x[i] = it->second;
        }
    }

    float raw = model.init_score;
    for (const auto& tree : model.trees) {
        if (tree.empty()) {
            continue;
        }
        int idx = 0;
        // Bound the walk by tree.size() as an extra safety net; the JSON
        // validator already rejected invalid indices at load time.
        int guard = static_cast<int>(tree.size()) + 2;
        while (guard-- > 0 && tree[idx].feat >= 0) {
            const V3GBDTNode& node = tree[idx];
            const int feat = node.feat;
            const float value = feat < static_cast<int>(feat_count) ? x[feat] : 0.0f;
            idx = value <= node.thr ? node.left : node.right;
            if (idx < 0 || idx >= static_cast<int>(tree.size())) {
                break;
            }
        }
        if (idx >= 0 && idx < static_cast<int>(tree.size())) {
            raw += tree[idx].leaf_value;
        }
    }
    return model.logistic ? clamp01(1.0f / (1.0f + std::exp(-raw))) : raw;
}

float LinhaiSearchEngineV3::predict_head(
    const V3LinearModel& linear,
    const V3GBDTModel& gbdt,
    const std::unordered_map<std::string, float>& features
) const {
    // A-2b: GBDT wins when both are loaded; this is the same precedence
    // tools/train_model_stub.py uses offline, so train/infer stay aligned.
    if (gbdt.loaded) {
        return predict_gbdt(gbdt, features);
    }
    if (linear.loaded) {
        return predict_model(linear, features);
    }
    return 0.0f;
}

// A-2b: all six estimate_* heads now delegate to predict_head(), which
// prefers the GBDT model when available and falls through to the linear
// model otherwise. When neither is loaded we return the heuristic directly
// (this is the path stale params directories take).

float LinhaiSearchEngineV3::estimate_agari_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_agari_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!agari_model_.loaded && !agari_gbdt_.loaded) {
        return clamp01(heuristic_agari_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_agari_prob"] = clamp01(heuristic_agari_prob);
    return clamp01(predict_head(agari_model_, agari_gbdt_, features));
}

float LinhaiSearchEngineV3::estimate_houjuu_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_houjuu_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!houjuu_model_.loaded && !houjuu_gbdt_.loaded) {
        return clamp01(heuristic_houjuu_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_houjuu_prob"] = clamp01(heuristic_houjuu_prob);
    return clamp01(predict_head(houjuu_model_, houjuu_gbdt_, features));
}

float LinhaiSearchEngineV3::estimate_tenpai_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_tenpai_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!tenpai_model_.loaded && !tenpai_gbdt_.loaded) {
        return clamp01(heuristic_tenpai_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_tenpai_prob"] = clamp01(heuristic_tenpai_prob);
    return clamp01(predict_head(tenpai_model_, tenpai_gbdt_, features));
}

float LinhaiSearchEngineV3::estimate_betaori_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_betaori_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!betaori_model_.loaded && !betaori_gbdt_.loaded) {
        return clamp01(heuristic_betaori_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_betaori_prob"] = clamp01(heuristic_betaori_prob);
    return clamp01(predict_head(betaori_model_, betaori_gbdt_, features));
}

float LinhaiSearchEngineV3::estimate_tsumo_num(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_tsumo_num,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!tsumo_num_model_.loaded && !tsumo_num_gbdt_.loaded) {
        return std::max(0.0f, heuristic_tsumo_num);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_tsumo_num"] = std::max(0.0f, heuristic_tsumo_num);
    return std::max(0.0f, predict_head(tsumo_num_model_, tsumo_num_gbdt_, features));
}

// 验收 (spec §3.3): score-aware EV head. 返回模型预测的"自家本局期望冲数"
// 或"自家本局期望失分（负值）"。score head 已加载时调用，否则返回 0。
float LinhaiSearchEngineV3::estimate_agari_score(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!agari_score_model_.loaded && !agari_score_gbdt_.loaded) {
        return 0.0f;
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    return predict_head(agari_score_model_, agari_score_gbdt_, features);
}

float LinhaiSearchEngineV3::estimate_houjuu_score(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!houjuu_score_model_.loaded && !houjuu_score_gbdt_.loaded) {
        return 0.0f;
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    return predict_head(houjuu_score_model_, houjuu_score_gbdt_, features);
}

float LinhaiSearchEngineV3::estimate_ryukyoku_prob(
    const CanonicalGameState& state,
    const SearchCandidate& candidate,
    float heuristic_ryukyoku_prob,
    const std::string& action,
    int target_hai,
    bool safe
) const {
    if (!ryukyoku_model_.loaded && !ryukyoku_gbdt_.loaded) {
        return clamp01(heuristic_ryukyoku_prob);
    }
    auto features = build_state_features(state);
    add_candidate_features(features, candidate, action, target_hai, safe);
    features["heuristic_ryukyoku_prob"] = clamp01(heuristic_ryukyoku_prob);
    return clamp01(predict_head(ryukyoku_model_, ryukyoku_gbdt_, features));
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

int LinhaiSearchEngineV3::cached_shanten(const Hai_Array& tehai) {
    std::size_t seed = 0xcbf29ce484222325ULL;
    for (int hai = 1; hai < 38; hai++) {
        hash_combine(seed, static_cast<std::size_t>(tehai[hai]));
    }
    auto it = shanten_cache_.find(seed);
    if (it != shanten_cache_.end()) {
        return it->second;
    }
    // Bound memory: in pathological long-running processes the cache could
    // grow unbounded. 200k entries is comfortable for a selfplay loop on a
    // production box and keeps the map well under 10 MB.
    if (shanten_cache_.size() >= 200000) {
        shanten_cache_.clear();
    }
    const int shanten = calc_linhai_shanten(tehai);
    shanten_cache_[seed] = shanten;
    return shanten;
}

int LinhaiSearchEngineV3::cached_shanten(const Hai_Array& tehai, const Fuuro_Vector& fuuro) {
    if (fuuro.empty()) {
        return cached_shanten(tehai);
    }
    // Merge fuuro tiles back into a virtual 13/14-tile hand, then reuse the
    // hashed cache. We can't naively subtract `2 * fuuro_count` from the raw
    // shanten because standard decomposition on e.g. a 10-tile post-chi hand
    // can still produce pairless configurations (2 melds + 2 tatsu + no pair)
    // that the `-2*fuuro` adjustment would falsely rank as tenpai — observed
    // in real chi→discard evaluations where check_win returned 0 waits even
    // though the adjusted shanten was 0.
    const Hai_Array merged = using_hai_array(tehai, fuuro);
    return cached_shanten(merged);
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
    // === Phase A spec §3.5.1: 风险厌恶 EV 重加权 ===
    // 关键：opp_pressure 只与"对手公开信息"有关，与具体打哪张牌无关，所以
    // 提到候选循环外只算一次。circa 27 候选 × 1 µs = 27 µs 的固定开销，与
    // perf_regression P95 ≤ 800ms 预算相比可忽略。
    const float opp_pressure = compute_opp_pressure_score(state);
    const auto& evr = score_table_.ev_risk_weights;
    const float dynamic_lambda = std::min(
        evr.lambda_max,
        evr.lambda_base + evr.lambda_pressure_step * opp_pressure
    );
    const float dynamic_mu = std::min(
        evr.mu_max,
        evr.mu_base + evr.mu_pressure_step * opp_pressure
    );
    // Phase A spec §3.5.3: 防守特征对 EV 的线性修正系数（对所有候选共用）
    const float opp_pressure_feature_penalty =
        score_table_.feature_weights.opp_meld_pressure_alpha * opp_pressure * 200.0f;
    for (int hai = 1; hai < 38; hai++) {
        if (game_state.tehai[hai] <= 0 || !is_linhai_valid_tile(hai) || seen.count(hai)) {
            continue;
        }
        seen.insert(hai);
        Hai_Array after = game_state.tehai;
        after[hai] -= 1;
        // A-1: use exact linhai shanten (cached) instead of the fast-but-wrong
        // estimate. The fast estimator could be off by multiple shanten in
        // real hands, which systematically biases every downstream EV term
        // (total_ev, tenpai_prob, tsumo_num, ...).
        // Bug fix: include fuuro so post-chi/post-pon states evaluate shanten
        // on the full 13-tile representation. Without this, the engine
        // systematically underestimated chi tenpai paths.
        int shanten_after = cached_shanten(after, game_state.fuuro);
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
        const float heuristic_agari_prob = clamp01(
            shanten_after <= 0
                ? 0.55f + static_cast<float>(ukeire) * 0.005f
                : (shanten_after == 1
                       ? 0.18f + static_cast<float>(ukeire) * 0.004f
                       : (shanten_after == 2
                              ? 0.05f + static_cast<float>(ukeire) * 0.0015f
                              : 0.015f + static_cast<float>(ukeire) * 0.0005f))
        );
        c.houjuu_prob = estimate_houjuu_prob(state, c, heuristic_houjuu_prob, c.action, 0, safe);
        c.agari_prob = estimate_agari_prob(state, c, heuristic_agari_prob, c.action, 0, safe);
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
        // C-3: betaori_unsafe 40→150 (safe 160→350). 旧配比 40 在 houjuu
        // 高的局面几乎无效（betaori=0.3 时只贡献 12），提升后能让搜索在
        // 多面听威胁下更主动选取 betaori 高的弃牌。
        c.total_ev += c.betaori_prob * (safe ? 350.0f : 150.0f);
        c.total_ev -= c.ryukyoku_prob * 320.0f;
        c.total_ev -= c.tsumo_num * 18.0f;
        // A-2c-final: previously houjuu_prob was computed but never affected
        // total_ev (only the unused defense_score field). This explained why
        // ROI #3's improved houjuu predictions never moved win rate -- the
        // search literally ignored them.
        // C-3: 4000→5500，进一步强化防御。baseline 0.693 winrate 下
        // self_houjuu=0.135 vs opp=0.287，还有下压空间。
        // === Phase A spec §3.5.1: 动态 λ + μ 风险加权 ===
        // λ 项：5500 系数读自 score_table.ev_risk_weights.houjuu_base_coeff，
        // 基础值 1.5（spec 默认），随 opp_pressure_score 线性放大到 lambda_max。
        const float houjuu_loss_base = c.houjuu_prob * evr.houjuu_base_coeff;
        c.total_ev -= dynamic_lambda * houjuu_loss_base;
        // μ 项：当对手压力高（opp_pressure > 0.3）且模型 score head 已加载时，
        // 对"对手 ≥4 冲"长尾胡牌路径额外惩罚。high_chong_loss 是预测放炮失分
        // 超过 high_chong_threshold（默认 4 冲）的部分。
        if (opp_pressure > 0.3f && has_score_heads_loaded()) {
            const float houjuu_score_ev = estimate_houjuu_score(state, c, c.action, 0, safe);
            const float high_chong_loss = std::max(0.0f, std::abs(houjuu_score_ev) - evr.high_chong_threshold);
            c.total_ev -= dynamic_mu * high_chong_loss * 200.0f;
        }
        // C-2: agari_prob was only weighted at root, so inner look-ahead
        // candidates ignored winning-chance entirely. Pull it into
        // build_discard_candidates so recursive draw simulation sees the
        // attack signal. Root retains its own +agari*2800 against the
        // post-discard state, so inner uses a damped 1800 to avoid
        // over-stacking (root effective weight ~ 1800*0.45 + 2800).
        c.total_ev += c.agari_prob * 1800.0f;
        // 验收 (spec §3.3): score-aware EV. 当 agari_score / houjuu_score
        // head 加载成功时，叠加"模型预测的期望冲数"。1 冲 = 1 分（spec T6
        // 默认），冲数尺度与 prob×经验冲数 相比小一个量级，所以这里乘 800
        // 把 score 信号拉到与 agari_prob*1800 / houjuu_prob*5500 同一量级。
        // 训练后 R3 模型会把 hunyise/qingyise/抢杠胡 等真实番数信号反映出来。
        if (has_score_heads_loaded()) {
            const float agari_score_ev = estimate_agari_score(state, c, c.action, 0, safe);
            const float houjuu_score_ev = estimate_houjuu_score(state, c, c.action, 0, safe);
            // agari_score 是带号期望（胡牌时正、不胡时小）；houjuu_score 是
            // 带号期望（放炮时负、不放炮时小）。score 表示"一冲"，乘 800
            // 把单位拉到与原 EV 一致。
            c.total_ev += agari_score_ev * 800.0f;
            c.total_ev -= houjuu_score_ev * 800.0f;
        }
        // === Phase A spec §3.5.3: 防守特征对 EV 的线性修正 ===
        // 对所有候选共用同一个惩罚（opp_pressure 与具体打哪张牌无关），
        // 因此本质上是对全候选 total_ev 的统一偏移，不改变排序；
        // 但保留以便 acceptance_25_8 度量与 score_aware EV 累加一致。
        c.total_ev -= opp_pressure_feature_penalty;
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
    // === Phase A spec §3.5.2: 硬 betaori 门 ===
    // 危险局面（所有候选 houjuu_prob 都超过动态阈值）下，强制把 EV 重排为
    // "houjuu_prob 越低 EV 越高"，即"宁可输 shanten 也保命"。这与 A4 的
    // 软性 λ/μ 加权互补——A4 调小排序差异，A5 在极端情况硬翻盘排序。
    // opp_pressure 已在循环外计算（line 1067 附近 hoisted），此处复用。
    if (!candidates.empty()) {
        const auto& bt = score_table_.betaori_thresholds;
        // 实时阈值：基础值 -（副露压力下调）-（清一色压力下调）
        float betaori_threshold = bt.base;
        const bool has_meld_pressure = state.opponent_meld_count >= 1;
        const bool has_qingyise_signal = opp_pressure > 0.5f;
        if (has_meld_pressure) betaori_threshold -= bt.meld_drop;
        if (has_qingyise_signal) betaori_threshold -= bt.qingyise_drop;

        // 最低 houjuu_prob：若全候选都不安全，门触发
        float min_houjuu = 1.0f;
        for (const auto& c : candidates) {
            if (c.houjuu_prob < min_houjuu) min_houjuu = c.houjuu_prob;
        }
        const bool force_betaori_mode =
            (min_houjuu > betaori_threshold) ||
            (min_houjuu > betaori_threshold * 0.5f && has_qingyise_signal);

        if (force_betaori_mode) {
            // 用 ev_loss_multiplier 放大原 houjuu 惩罚（基础 8000，是常规 5500
            // 的 1.45×；再乘 1.5 默认 multiplier，总效果 ~12000，足以压倒
            // shanten*1200 等正向 EV 项），把 ranking 强制翻为低 houjuu 优先。
            for (auto& c : candidates) {
                c.total_ev -= c.houjuu_prob * 8000.0f * bt.ev_loss_multiplier;
                c.explanation += "[强制 betaori]";
            }
        }
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

    // A-1: probability-mass-preserving chance integration.
    //
    // Before: we walked the tile list in index order, stopped after `beam`
    // matches, and divided partial contributions by the *full* remaining count.
    // That made `evaluate_future_draws` systematically under-estimate EV, and
    // different subtrees would truncate at different points -> non-comparable.
    //
    // Now: we rank candidate draw tiles by remaining count (largest first),
    // process as many as budget allows, and extrapolate the untaken
    // probability mass using the weighted average of the observed future_best
    // values. This keeps the integral unbiased (E[future_best] matches the
    // sampled tiles' average) while still respecting the search budget.
    struct DrawChoice { int hai; int remain; };
    std::vector<DrawChoice> draws;
    draws.reserve(LINHAI_VALID_TILE_COUNT);
    for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
        const int hai = LINHAI_VALID_TILES[i];
        const int remain = state.remaining_counts[hai];
        if (remain <= 0) {
            continue;
        }
        draws.push_back({hai, remain});
    }
    std::sort(draws.begin(), draws.end(), [](const DrawChoice& a, const DrawChoice& b) {
        if (a.remain != b.remain) {
            return a.remain > b.remain;
        }
        return a.hai < b.hai;
    });

    const int beam = depth > 1 ? config_.beam_width_after_draw : config_.beam_width_far_shanten;
    const int limit = std::max(1, beam);
    float weighted = 0.0f;
    float covered_probability = 0.0f;
    int used = 0;
    for (const auto& draw : draws) {
        if (used >= limit) {
            break;
        }
        if (is_search_budget_exceeded()) {
            break;
        }

        CanonicalGameState next = state;
        next.game_state.tehai[draw.hai] += 1;
        next.white_tiles_in_hand = next.game_state.tehai[35];
        next.game_state.wall_remaining = std::max(0, next.game_state.wall_remaining - 1);
        next.remaining_counts[draw.hai] = std::max(0, next.remaining_counts[draw.hai] - 1);

        auto recs = build_discard_candidates(next);
        const float future_best = recs.empty() ? 0.0f : recs.front().total_ev;
        const float probability = static_cast<float>(draw.remain) / static_cast<float>(total_remaining);
        weighted += probability * future_best;
        covered_probability += probability;
        add_search_nodes(1);
        nodes_expanded += 1;
        used += 1;
    }

    if (covered_probability > 0.0f && covered_probability < 1.0f) {
        // Extrapolate the uncovered probability mass using the sampled mean.
        // This is the unbiased estimator under the assumption that the
        // remaining draws look "similar on average" to the sampled ones,
        // which is a much better approximation than treating them as zero.
        const float mean_future = weighted / covered_probability;
        weighted += (1.0f - covered_probability) * mean_future;
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
        // C-4 (lone non-yakuhai honor preference): the engine's pure EV math
        // treats a lone non-yakuhai honor (e.g. South for an East player) as
        // roughly equivalent to other terminal floats like 9m. In real play
        // — and in any seasoned engine (Mortal/Suphx/Tenhou) — a lone honor
        // that doesn't combine with anything and gives no yaku is the
        // textbook first discard: zero structural cost, near-zero deal-in
        // risk early, minimum information leak. Without this nudge the
        // engine has been systematically picking number-tile floats over
        // lone non-grab-charge honors when EVs are within ~30. We apply the
        // bonus ONLY at the root (not in build_discard_candidates) — adding
        // it inside the recursion would let "discard 9m now → discard south
        // with +60 next turn" outscore "discard south now with +60", which
        // backfires on the original case. Root-only application correctly
        // breaks ties between equivalent floats without leaking into
        // future_ev rollouts.
        const int pre_count = state.game_state.tehai[candidate.hai];
        const bool is_honor = (candidate.hai >= 31 && candidate.hai <= 37);
        const bool is_wildcard = (candidate.hai == 35);
        if (pre_count == 1 && is_honor && !is_wildcard
                && !is_grab_charge_tile(candidate.hai, state.game_state.jikaze)) {
            candidate.total_ev += 60.0f;
            candidate.explanation += ", 弃孤张非役字牌";
        }
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
    // A-1: use exact shanten so response-mode depth picking is correct.
    // Include fuuro so depth choice is correct when the hand already has melds.
    int current_shanten = cached_shanten(tehai, state.game_state.fuuro);
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

        // A-1: preserve probability mass on the beam-truncated tail.
        // Originally we dropped uncovered mass silently, which biased `weighted_*`
        // against long-tail draws AND broke comparability across actions
        // whose chance trees happen to have different uncovered mass.
        // Here we extrapolate the uncovered mass using the sampled average
        // (unbiased estimator given no further information).
        if (used_probability < 1.0f) {
            const float uncovered = 1.0f - used_probability;
            const float mean_total_ev = weighted_total_ev / used_probability;
            const float mean_agari = weighted_agari_prob / used_probability;
            const float mean_houjuu = weighted_houjuu_prob / used_probability;
            const float mean_tenpai = weighted_tenpai_prob / used_probability;
            const float mean_betaori = weighted_betaori_prob / used_probability;
            const float mean_tsumo = weighted_tsumo_num / used_probability;
            const float mean_ryukyoku = weighted_ryukyoku_prob / used_probability;
            const float mean_defense = weighted_defense_score / used_probability;
            weighted_total_ev += uncovered * mean_total_ev;
            weighted_agari_prob += uncovered * mean_agari;
            weighted_houjuu_prob += uncovered * mean_houjuu;
            weighted_tenpai_prob += uncovered * mean_tenpai;
            weighted_betaori_prob += uncovered * mean_betaori;
            weighted_tsumo_num += uncovered * mean_tsumo;
            weighted_ryukyoku_prob += uncovered * mean_ryukyoku;
            weighted_defense_score += uncovered * mean_defense;
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
        c.shanten = cached_shanten(next_state.game_state.tehai, next_state.game_state.fuuro);
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
    int shanten = cached_shanten(state.game_state.tehai, state.game_state.fuuro);
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

// Phase A spec §3.5.3: 副露压力综合标量。
// 取值范围 [0, ~1.5]：0=对手无副露/无威胁；~1.5=对手副露 ≥3 + 同色集中 ≥9 张
// + 字牌刻子可见 + 抓冲两次 + 副露含白板，全部叠满（极端罕见）。
//
// Phase A 仅暴露此 helper，A4 任务会在 EV 公式里通过它动态加权 λ/μ；本任务无外部
// 行为变化，验证仅靠 compile-clean + 现有 pytest 不回归。
//
// 字段来源：
//   - state.opponent_meld_count        : Python _build_canonical_state 已填
//   - state.opponent_discards          : 对手实际弃牌（38-array hai code 列表）
//   - state.opponent_honor_triplets    : Phase A 新增，Python 据 meld.tiles 填
//   - state.grab_charge_hits           : 已存在；spec 中 grab_charge_caught_count
//                                         在 CanonicalGameState 上对应字段就是它
//   - state.opp_white_meld_count       : Phase A 新增，Python 据 meld.tiles 填
float LinhaiSearchEngineV3::compute_opp_pressure_score(const CanonicalGameState& state) const {
    auto local_clamp01 = [](float x) { return std::max(0.0f, std::min(1.0f, x)); };

    // 1) 副露姿态：副露数 / 3 → [0, 1]
    const float meld_term =
        0.30f * local_clamp01(static_cast<float>(state.opponent_meld_count) / 3.0f);

    // 2) 清一色警报：在对手已经"暴露"的牌里（弃牌 + 副露代表牌）按花色聚合，
    //    取最大数。≥6 张时开始累加，9 张时拉满。
    //    临海花色：万 (1..9) / 索 (21..29) / 字 (31..37)。无筒子。
    int suit_counts[3] = {0, 0, 0};
    for (int hai : state.opponent_discards) {
        if (hai >= 1 && hai <= 9) {
            suit_counts[0] += 1;
        } else if (hai >= 21 && hai <= 29) {
            suit_counts[1] += 1;
        } else if (hai >= 31 && hai <= 37) {
            suit_counts[2] += 1;
        }
    }
    int max_suit = 0;
    if (suit_counts[0] > max_suit) max_suit = suit_counts[0];
    if (suit_counts[1] > max_suit) max_suit = suit_counts[1];
    // 字牌不计入清一色（它由 ziyise_alarm 单独覆盖）
    const float qingyise_alarm =
        local_clamp01((static_cast<float>(max_suit) - 5.0f) / 4.0f);
    const float qingyise_term = 0.25f * qingyise_alarm;

    // 3) 字一色警报：对手字牌刻子数 / 2 → [0, 1]
    const float ziyise_alarm =
        local_clamp01(static_cast<float>(state.opponent_honor_triplets) / 2.0f);
    const float ziyise_term = 0.20f * ziyise_alarm;

    // 4) 抓冲红头威胁：对手已抓冲数 / 2 → [0, 1]
    //    （CanonicalGameState 字段名为 grab_charge_hits，语义等同于
    //    spec 中的 grab_charge_caught_count。）
    const float redhead_term =
        0.15f * local_clamp01(static_cast<float>(state.grab_charge_hits) / 2.0f);

    // 5) 白板威胁：对手副露含白板（白板=35）→ 1
    const float white_term =
        0.10f * local_clamp01(static_cast<float>(state.opp_white_meld_count));

    return meld_term + qingyise_term + ziyise_term + redhead_term + white_term;
}

} // namespace linhai
