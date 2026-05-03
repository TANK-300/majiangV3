#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include "share/linhai_ai_engine.hpp"
#include "share/linhai_ev_engine.hpp"
#include "share/linhai_search_v3.hpp"
#include "share/linhai_shanten_v4.hpp"
#include "share/types.hpp"

namespace py = pybind11;

// 辅助函数：牌码字符串转整数
int tile_str_to_int(const std::string& tile_str) {
    // 万子 (m)
    if (tile_str == "1m") return 1;
    if (tile_str == "2m") return 2;
    if (tile_str == "3m") return 3;
    if (tile_str == "4m") return 4;
    if (tile_str == "5m") return 5;
    if (tile_str == "6m") return 6;
    if (tile_str == "7m") return 7;
    if (tile_str == "8m") return 8;
    if (tile_str == "9m") return 9;

    // 筒子 (p)
    if (tile_str == "1p") return 11;
    if (tile_str == "2p") return 12;
    if (tile_str == "3p") return 13;
    if (tile_str == "4p") return 14;
    if (tile_str == "5p") return 15;
    if (tile_str == "6p") return 16;
    if (tile_str == "7p") return 17;
    if (tile_str == "8p") return 18;
    if (tile_str == "9p") return 19;

    // 条子 (s)
    if (tile_str == "1s" || tile_str == "1t") return 21;
    if (tile_str == "2s" || tile_str == "2t") return 22;
    if (tile_str == "3s" || tile_str == "3t") return 23;
    if (tile_str == "4s" || tile_str == "4t") return 24;
    if (tile_str == "5s" || tile_str == "5t") return 25;
    if (tile_str == "6s" || tile_str == "6t") return 26;
    if (tile_str == "7s" || tile_str == "7t") return 27;
    if (tile_str == "8s" || tile_str == "8t") return 28;
    if (tile_str == "9s" || tile_str == "9t") return 29;

    // 字牌
    if (tile_str == "east" || tile_str == "E") return 31;
    if (tile_str == "south" || tile_str == "S") return 32;
    if (tile_str == "west" || tile_str == "W") return 33;
    if (tile_str == "north" || tile_str == "N") return 34;
    if (tile_str == "white" || tile_str == "P") return 35;
    if (tile_str == "green" || tile_str == "F") return 36;
    if (tile_str == "red" || tile_str == "C") return 37;

    return 0;
}

// 辅助函数：手牌列表转数组
Hai_Array hand_list_to_array(const std::vector<std::string>& hand) {
    Hai_Array tehai = {0};
    for (const auto& tile : hand) {
        int tile_int = tile_str_to_int(tile);
        if (tile_int > 0 && tile_int < 38) {
            tehai[tile_int]++;
        }
    }
    return tehai;
}

PYBIND11_MODULE(linhai_v3, m) {
    m.doc() = "临海麻将V3搜索引擎Python绑定";

    // 绑定DiscardRecommendation
    py::class_<linhai::DiscardRecommendation>(m, "DiscardRecommendation")
        .def(py::init<>())
        .def_readonly("hai", &linhai::DiscardRecommendation::hai)
        .def_readonly("score", &linhai::DiscardRecommendation::score)
        .def_readonly("shanten", &linhai::DiscardRecommendation::shanten)
        .def_readonly("bonus_value", &linhai::DiscardRecommendation::bonus_value)
        .def_readonly("risk_value", &linhai::DiscardRecommendation::risk_value)
        .def_readonly("explanation", &linhai::DiscardRecommendation::explanation);

    py::class_<linhai::ResponseRecommendation>(m, "ResponseRecommendation")
        .def(py::init<>())
        .def_readonly("action", &linhai::ResponseRecommendation::action)
        .def_readonly("target_hai", &linhai::ResponseRecommendation::target_hai)
        .def_readonly("consumed", &linhai::ResponseRecommendation::consumed)
        .def_readonly("discard_hai", &linhai::ResponseRecommendation::discard_hai)
        .def_readonly("score", &linhai::ResponseRecommendation::score)
        .def_readonly("shanten", &linhai::ResponseRecommendation::shanten)
        .def_readonly("bonus_value", &linhai::ResponseRecommendation::bonus_value)
        .def_readonly("risk_value", &linhai::ResponseRecommendation::risk_value)
        .def_readonly("explanation", &linhai::ResponseRecommendation::explanation);

    // 绑定Player_State
    py::class_<Player_State>(m, "PlayerState")
        .def(py::init<>())
        .def_readwrite("jikaze", &Player_State::jikaze)
        .def_readwrite("reach_declared", &Player_State::reach_declared);

    // 绑定GameState
    py::class_<linhai::GameState>(m, "GameState")
        .def(py::init<>())
        .def_readwrite("current_player", &linhai::GameState::current_player)
        .def_readwrite("jikaze", &linhai::GameState::jikaze)
        .def_readwrite("has_kan", &linhai::GameState::has_kan)
        .def_readwrite("wall_remaining", &linhai::GameState::wall_remaining)
        .def_readwrite("round", &linhai::GameState::round)
        .def_readwrite("all_players", &linhai::GameState::all_players)
        .def("set_hand", [](linhai::GameState& self, const std::vector<std::string>& hand) {
            self.tehai = hand_list_to_array(hand);
        })
        .def("set_current_melds", [](
            linhai::GameState& self,
            const std::vector<std::string>& meld_types,
            const std::vector<std::vector<std::string>>& meld_tiles
        ) {
            self.fuuro.clear();
            const size_t count = std::min(meld_types.size(), meld_tiles.size());
            for (size_t i = 0; i < count; ++i) {
                Fuuro_Elem fuuro;
                const std::string type = meld_types[i];
                if (type == "chi") {
                    fuuro.type = FT_CHI;
                } else if (type == "peng") {
                    fuuro.type = FT_PON;
                } else if (type == "ming_gang") {
                    fuuro.type = FT_DAIMINKAN;
                } else if (type == "an_gang") {
                    fuuro.type = FT_ANKAN;
                } else if (type == "bu_gang") {
                    fuuro.type = FT_KAKAN;
                } else {
                    fuuro.type = FT_PON;
                }

                int representative = 0;
                for (const auto& tile : meld_tiles[i]) {
                    const int tile_int = tile_str_to_int(tile);
                    if (tile_int > 0) {
                        if (representative == 0) {
                            representative = tile_int;
                        }
                        fuuro.consumed.push_back(tile_int);
                    }
                }
                fuuro.hai = representative;
                fuuro.target_relative = 1;
                self.fuuro.push_back(fuuro);
            }
        })
        .def("set_player_snapshots", [](
            linhai::GameState& self,
            const std::vector<int>& jikazes,
            const std::vector<int>& discard_counts,
            const std::vector<int>& meld_counts,
            const std::vector<bool>& reach_flags
        ) {
            size_t player_count = 4;
            player_count = std::max(player_count, jikazes.size());
            player_count = std::max(player_count, discard_counts.size());
            player_count = std::max(player_count, meld_counts.size());
            player_count = std::max(player_count, reach_flags.size());

            self.all_players.assign(player_count, Player_State());
            for (size_t i = 0; i < player_count; ++i) {
                Player_State& player = self.all_players[i];
                player.jikaze = i < jikazes.size() ? jikazes[i] : static_cast<int>(i);
                player.reach_declared = i < reach_flags.size() ? reach_flags[i] : false;
                player.reach_accepted = player.reach_declared;

                const int discard_count = i < discard_counts.size() ? std::max(0, discard_counts[i]) : 0;
                player.kawa.clear();
                for (int j = 0; j < discard_count; ++j) {
                    Sutehai sutehai;
                    player.kawa.push_back(sutehai);
                }

                const int meld_count = i < meld_counts.size() ? std::max(0, meld_counts[i]) : 0;
                player.fuuro.clear();
                for (int j = 0; j < meld_count; ++j) {
                    Fuuro_Elem fuuro;
                    fuuro.type = FT_PON;
                    player.fuuro.push_back(fuuro);
                }
            }
        });

    // 绑定AIConfig
    py::class_<linhai::AIConfig>(m, "AIConfig")
        .def(py::init<>())
        .def_readwrite("shanten_weight", &linhai::AIConfig::shanten_weight)
        .def_readwrite("bonus_weight", &linhai::AIConfig::bonus_weight)
        .def_readwrite("risk_weight", &linhai::AIConfig::risk_weight)
        .def_readwrite("aggressive_mode", &linhai::AIConfig::aggressive_mode)
        .def_readwrite("defensive_mode", &linhai::AIConfig::defensive_mode);

    // 绑定LinHaiAIEngine
    py::class_<linhai::LinHaiAIEngine>(m, "LinHaiAIEngine")
        .def(py::init<>())
        .def(py::init<const linhai::AIConfig&>())
        .def("recommend_best_discard", &linhai::LinHaiAIEngine::recommend_best_discard)
        .def("recommend_discard", &linhai::LinHaiAIEngine::recommend_discard)
        .def("recommend_best_response", &linhai::LinHaiAIEngine::recommend_best_response)
        .def("recommend_response", &linhai::LinHaiAIEngine::recommend_response)
        .def("record_chi", &linhai::LinHaiAIEngine::record_chi)
        .def("record_pon", &linhai::LinHaiAIEngine::record_pon)
        .def("reset", &linhai::LinHaiAIEngine::reset)
        .def("set_aggressive_mode", &linhai::LinHaiAIEngine::set_aggressive_mode)
        .def("set_defensive_mode", &linhai::LinHaiAIEngine::set_defensive_mode)
        .def("get_config", &linhai::LinHaiAIEngine::get_config)
        .def("set_config", &linhai::LinHaiAIEngine::set_config);

    // ========================================
    // V2: LinHaiEVEngine (期望值引擎)
    // ========================================

    // TileEV 结构
    py::class_<linhai::TileEV>(m, "TileEV")
        .def(py::init<>())
        .def_readonly("hai", &linhai::TileEV::hai)
        .def_readonly("ev_total", &linhai::TileEV::ev_total)
        .def_readonly("ev_offense", &linhai::TileEV::ev_offense)
        .def_readonly("ev_defense", &linhai::TileEV::ev_defense)
        .def_readonly("ev_linhai_bonus", &linhai::TileEV::ev_linhai_bonus)
        .def_readonly("risk_houjuu", &linhai::TileEV::risk_houjuu)
        .def_readonly("shanten_after", &linhai::TileEV::shanten_after)
        .def_readonly("ukeire_after", &linhai::TileEV::ukeire_after)
        .def_readonly("explanation", &linhai::TileEV::explanation);

    // LinHaiEVEngine
    py::class_<linhai::LinHaiEVEngine>(m, "LinHaiEVEngine")
        .def(py::init<>())
        .def(py::init<const linhai::AIConfig&>())
        .def("load_params", &linhai::LinHaiEVEngine::load_params,
            py::arg("dir") = "params/linhai/")
        .def("params_loaded", &linhai::LinHaiEVEngine::params_loaded)
        .def("recommend_discard_ev", &linhai::LinHaiEVEngine::recommend_discard_ev)
        .def("recommend_best_discard_v2", &linhai::LinHaiEVEngine::recommend_best_discard_v2)
        .def("recommend_discard_v2", &linhai::LinHaiEVEngine::recommend_discard_v2)
        .def("recommend_response_v2", &linhai::LinHaiEVEngine::recommend_response_v2)
        .def("set_opponent_discards", &linhai::LinHaiEVEngine::set_opponent_discards)
        .def("set_pass_hu", &linhai::LinHaiEVEngine::set_pass_hu)
        .def("get_can_win", &linhai::LinHaiEVEngine::get_can_win)
        .def("set_aggressive_mode", &linhai::LinHaiEVEngine::set_aggressive_mode)
        .def("set_defensive_mode", &linhai::LinHaiEVEngine::set_defensive_mode)
        .def("record_chi", &linhai::LinHaiEVEngine::record_chi)
        .def("record_pon", &linhai::LinHaiEVEngine::record_pon)
        .def("reset", &linhai::LinHaiEVEngine::reset)
        .def("get_config", &linhai::LinHaiEVEngine::get_config)
        .def("set_config", &linhai::LinHaiEVEngine::set_config);

    py::class_<linhai::SearchConfig>(m, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("max_self_draw_depth", &linhai::SearchConfig::max_self_draw_depth)
        .def_readwrite("deep_depth_for_near_ready", &linhai::SearchConfig::deep_depth_for_near_ready)
        .def_readwrite("near_ready_shanten_threshold", &linhai::SearchConfig::near_ready_shanten_threshold)
        .def_readwrite("beam_width_after_draw", &linhai::SearchConfig::beam_width_after_draw)
        .def_readwrite("beam_width_far_shanten", &linhai::SearchConfig::beam_width_far_shanten)
        .def_readwrite("enable_rollout", &linhai::SearchConfig::enable_rollout)
        .def_readwrite("discard_rollout_simulations", &linhai::SearchConfig::discard_rollout_simulations)
        .def_readwrite("response_rollout_simulations", &linhai::SearchConfig::response_rollout_simulations)
        .def_readwrite("discard_rollout_gap_ratio", &linhai::SearchConfig::discard_rollout_gap_ratio)
        .def_readwrite("response_rollout_gap_ratio", &linhai::SearchConfig::response_rollout_gap_ratio)
        .def_readwrite("discard_rollout_override_delta", &linhai::SearchConfig::discard_rollout_override_delta)
        .def_readwrite("response_rollout_override_delta", &linhai::SearchConfig::response_rollout_override_delta)
        .def_readwrite("time_budget_ms_discard", &linhai::SearchConfig::time_budget_ms_discard)
        .def_readwrite("time_budget_ms_response", &linhai::SearchConfig::time_budget_ms_response)
        .def_readwrite("max_nodes_discard", &linhai::SearchConfig::max_nodes_discard)
        .def_readwrite("max_nodes_response", &linhai::SearchConfig::max_nodes_response);

    py::class_<linhai::ModelBundle>(m, "ModelBundle")
        .def(py::init<>())
        .def_readonly("version_dir", &linhai::ModelBundle::version_dir)
        .def_readonly("loaded", &linhai::ModelBundle::loaded)
        .def_readonly("reason", &linhai::ModelBundle::reason);

    py::class_<linhai::SearchCandidate>(m, "SearchCandidate")
        .def(py::init<>())
        .def_readonly("action", &linhai::SearchCandidate::action)
        .def_readonly("hai", &linhai::SearchCandidate::hai)
        .def_readonly("target_hai", &linhai::SearchCandidate::target_hai)
        .def_readonly("total_ev", &linhai::SearchCandidate::total_ev)
        .def_readonly("agari_prob", &linhai::SearchCandidate::agari_prob)
        .def_readonly("tenpai_prob", &linhai::SearchCandidate::tenpai_prob)
        .def_readonly("houjuu_prob", &linhai::SearchCandidate::houjuu_prob)
        .def_readonly("betaori_prob", &linhai::SearchCandidate::betaori_prob)
        .def_readonly("tsumo_num", &linhai::SearchCandidate::tsumo_num)
        .def_readonly("ryukyoku_prob", &linhai::SearchCandidate::ryukyoku_prob)
        .def_readonly("defense_score", &linhai::SearchCandidate::defense_score)
        .def_readonly("shanten", &linhai::SearchCandidate::shanten)
        .def_readonly("ukeire", &linhai::SearchCandidate::ukeire)
        .def_readonly("search_depth", &linhai::SearchCandidate::search_depth)
        .def_readonly("explanation", &linhai::SearchCandidate::explanation);

    py::class_<linhai::SearchResult>(m, "SearchResult")
        .def(py::init<>())
        .def_readonly("action", &linhai::SearchResult::action)
        .def_readonly("hai", &linhai::SearchResult::hai)
        .def_readonly("target_hai", &linhai::SearchResult::target_hai)
        .def_readonly("total_ev", &linhai::SearchResult::total_ev)
        .def_readonly("agari_prob", &linhai::SearchResult::agari_prob)
        .def_readonly("tenpai_prob", &linhai::SearchResult::tenpai_prob)
        .def_readonly("houjuu_prob", &linhai::SearchResult::houjuu_prob)
        .def_readonly("betaori_prob", &linhai::SearchResult::betaori_prob)
        .def_readonly("tsumo_num", &linhai::SearchResult::tsumo_num)
        .def_readonly("ryukyoku_prob", &linhai::SearchResult::ryukyoku_prob)
        .def_readonly("defense_score", &linhai::SearchResult::defense_score)
        .def_readonly("shanten", &linhai::SearchResult::shanten)
        .def_readonly("ukeire", &linhai::SearchResult::ukeire)
        .def_readonly("search_depth", &linhai::SearchResult::search_depth)
        .def_readonly("nodes_expanded", &linhai::SearchResult::nodes_expanded)
        .def_readonly("search_nodes", &linhai::SearchResult::search_nodes)
        .def_readonly("root_candidates_total", &linhai::SearchResult::root_candidates_total)
        .def_readonly("root_candidates_evaluated", &linhai::SearchResult::root_candidates_evaluated)
        .def_readonly("configured_time_budget_ms", &linhai::SearchResult::configured_time_budget_ms)
        .def_readonly("configured_node_budget", &linhai::SearchResult::configured_node_budget)
        .def_readonly("cache_hits", &linhai::SearchResult::cache_hits)
        .def_readonly("truncated", &linhai::SearchResult::truncated)
        .def_readonly("chosen_by", &linhai::SearchResult::chosen_by)
        .def_readonly("fallback_reason", &linhai::SearchResult::fallback_reason)
        .def_readonly("truncate_reason", &linhai::SearchResult::truncate_reason)
        .def_readonly("candidate_scores", &linhai::SearchResult::candidate_scores);

    py::class_<linhai::CanonicalGameState>(m, "CanonicalGameState")
        .def(py::init<>())
        .def_readwrite("game_state", &linhai::CanonicalGameState::game_state)
        .def_readwrite("opponent_discards", &linhai::CanonicalGameState::opponent_discards)
        .def_readwrite("can_win", &linhai::CanonicalGameState::can_win)
        .def_readwrite("target_hai", &linhai::CanonicalGameState::target_hai)
        .def_readwrite("from_player", &linhai::CanonicalGameState::from_player)
        .def_readwrite("available_actions", &linhai::CanonicalGameState::available_actions)
        .def_readwrite("white_tiles_in_hand", &linhai::CanonicalGameState::white_tiles_in_hand)
        .def_readwrite("tree_active", &linhai::CanonicalGameState::tree_active)
        .def_readwrite("grab_charge_active", &linhai::CanonicalGameState::grab_charge_active)
        .def_readwrite("grab_charge_hits", &linhai::CanonicalGameState::grab_charge_hits)
        .def_readwrite("grab_charge_limit", &linhai::CanonicalGameState::grab_charge_limit)
        .def_readwrite("contract_target_count", &linhai::CanonicalGameState::contract_target_count)
        .def_readwrite("contract_counter", &linhai::CanonicalGameState::contract_counter)
        .def_readwrite("opponent_meld_count", &linhai::CanonicalGameState::opponent_meld_count)
        .def_readwrite("opponent_discard_count", &linhai::CanonicalGameState::opponent_discard_count)
        // Phase A §3.5.3: 副露压力辅助字段（compute_opp_pressure_score 内部使用）
        .def_readwrite("opponent_honor_triplets", &linhai::CanonicalGameState::opponent_honor_triplets)
        .def_readwrite("opp_white_meld_count", &linhai::CanonicalGameState::opp_white_meld_count)
        .def("refresh_counts", &linhai::CanonicalGameState::refresh_counts)
        .def("set_hand", [](linhai::CanonicalGameState& self, const std::vector<std::string>& hand) {
            self.game_state.tehai = hand_list_to_array(hand);
            self.white_tiles_in_hand = self.game_state.tehai[35];
            self.refresh_counts();
        });

    py::class_<linhai::LinhaiSearchEngineV3>(m, "LinhaiSearchEngineV3")
        .def(py::init<>())
        .def("load_model_bundle", &linhai::LinhaiSearchEngineV3::load_model_bundle)
        .def("set_search_config", &linhai::LinhaiSearchEngineV3::set_search_config)
        .def("get_search_config", &linhai::LinhaiSearchEngineV3::get_search_config)
        .def("get_model_bundle", &linhai::LinhaiSearchEngineV3::get_model_bundle)
        .def("get_last_search_debug", &linhai::LinhaiSearchEngineV3::get_last_search_debug)
        .def("recommend_discard_v3", &linhai::LinhaiSearchEngineV3::recommend_discard_v3)
        .def("recommend_response_v3", &linhai::LinhaiSearchEngineV3::recommend_response_v3)
        // Phase A §3.5.3: 副露压力综合标量（A4 将动态加权 EV 风险项）
        .def("compute_opp_pressure_score", &linhai::LinhaiSearchEngineV3::compute_opp_pressure_score);

    // ========================================
    // 辅助函数
    // ========================================

    m.def("tile_str_to_int", &tile_str_to_int, "将牌码字符串转换为整数");
    m.def("tile_int_to_str", &hai_int_to_str, "将整数转换为牌码字符串");
    m.def("hand_list_to_array", &hand_list_to_array, "将手牌列表转换为数组");

    // 快速胡牌检查 (用于自对弈模拟器)
    m.def("check_win", [](const std::vector<std::string>& hand) -> bool {
        Hai_Array tehai = hand_list_to_array(hand);
        // 向听数 = -1 表示已胡牌
        int shanten = calc_linhai_shanten(tehai);
        return shanten == -1;
    }, "检查手牌是否胡牌 (向听数=-1)");

    // 快速向听数计算
    m.def("calc_shanten", [](const std::vector<std::string>& hand) -> int {
        Hai_Array tehai = hand_list_to_array(hand);
        return calc_linhai_shanten(tehai);
    }, "计算手牌向听数");
}
