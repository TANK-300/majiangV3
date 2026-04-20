#include "share/linhai_search_v3.hpp"
#include "share/linhai_rules.hpp"
#include "share/linhai_shanten_v4.hpp"
#include <cassert>

int main() {
    assert(WHITE_TILE == 35);

    Hai_Array white_pair_hand = {0};
    white_pair_hand[1] = 1;
    white_pair_hand[2] = 1;
    white_pair_hand[3] = 1;
    white_pair_hand[4] = 1;
    white_pair_hand[5] = 1;
    white_pair_hand[6] = 1;
    white_pair_hand[7] = 1;
    white_pair_hand[8] = 1;
    white_pair_hand[9] = 1;
    white_pair_hand[21] = 1;
    white_pair_hand[22] = 1;
    white_pair_hand[23] = 1;
    white_pair_hand[31] = 1;
    white_pair_hand[35] = 1;
    assert(calc_linhai_shanten(white_pair_hand) < calc_standard_shanten(white_pair_hand));

    Hai_Array triple_white_hand = {0};
    triple_white_hand[1] = 1;
    triple_white_hand[2] = 1;
    triple_white_hand[3] = 1;
    triple_white_hand[4] = 1;
    triple_white_hand[5] = 1;
    triple_white_hand[6] = 1;
    triple_white_hand[7] = 1;
    triple_white_hand[8] = 1;
    triple_white_hand[9] = 1;
    triple_white_hand[21] = 1;
    triple_white_hand[31] = 1;
    triple_white_hand[35] = 3;
    assert(calc_linhai_shanten(triple_white_hand) < calc_standard_shanten(triple_white_hand));

    linhai::CanonicalGameState state;
    state.refresh_counts();
    linhai::LinhaiSearchEngineV3 engine;
    linhai::SearchConfig cfg;
    cfg.time_budget_ms_discard = -1;
    cfg.time_budget_ms_response = -1;
    engine.set_search_config(cfg);

    state.game_state.tehai[1] = 1;
    state.game_state.tehai[2] = 1;
    state.game_state.tehai[3] = 2;
    state.game_state.tehai[4] = 1;
    state.game_state.tehai[5] = 1;
    state.game_state.tehai[6] = 1;
    state.game_state.tehai[21] = 1;
    state.game_state.tehai[22] = 1;
    state.game_state.tehai[23] = 1;
    state.game_state.tehai[31] = 1;
    state.game_state.tehai[35] = 1;
    state.target_hai = 3;
    state.available_actions = {"pass", "peng", "chi"};
    state.can_win = true;
    state.refresh_counts();

    auto response = engine.recommend_response_v3(state);
    assert(!response.candidate_scores.empty());
    assert(response.action == "pass" || response.action == "peng" || response.action == "chi" || response.action == "hu");
    assert(response.nodes_expanded > 0);
    assert(response.search_nodes > 0);
    assert(response.search_nodes == response.nodes_expanded);
    assert(response.root_candidates_total == static_cast<int>(response.candidate_scores.size()));
    assert(response.root_candidates_evaluated == response.root_candidates_total);
    assert(response.configured_time_budget_ms == cfg.time_budget_ms_response);
    assert(response.configured_node_budget == cfg.max_nodes_response);
    assert(!response.truncated);
    assert(response.truncate_reason.empty());

    auto discard = engine.recommend_discard_v3(state);
    assert(discard.action == "discard");
    assert(discard.hai > 0);
    assert(discard.nodes_expanded > 0);
    assert(discard.search_nodes > 0);
    assert(discard.search_nodes == discard.nodes_expanded);
    assert(discard.root_candidates_total >= static_cast<int>(discard.candidate_scores.size()));
    assert(discard.root_candidates_evaluated <= static_cast<int>(discard.candidate_scores.size()));
    assert(discard.configured_time_budget_ms == cfg.time_budget_ms_discard);
    assert(discard.configured_node_budget == cfg.max_nodes_discard);
    assert(!discard.truncated);
    assert(discard.truncate_reason.empty());

    linhai::CanonicalGameState gang_state = state;
    gang_state.game_state.tehai[3] = 3;
    gang_state.available_actions = {"pass", "gang"};
    gang_state.refresh_counts();
    auto gang_response = engine.recommend_response_v3(gang_state);
    assert(!gang_response.candidate_scores.empty());
    assert(gang_response.action == "pass" || gang_response.action == "gang");
    assert(gang_response.nodes_expanded > 0);
    assert(gang_response.search_nodes > 0);
    assert(gang_response.search_nodes == gang_response.nodes_expanded);
    assert(gang_response.root_candidates_total == static_cast<int>(gang_response.candidate_scores.size()));
    assert(gang_response.root_candidates_evaluated == gang_response.root_candidates_total);
    assert(!gang_response.truncated);
    assert(gang_response.truncate_reason.empty());

    linhai::LinhaiSearchEngineV3 limited_engine;
    linhai::SearchConfig limited_cfg;
    limited_cfg.time_budget_ms_response = 0;
    limited_engine.set_search_config(limited_cfg);
    auto limited_response = limited_engine.recommend_response_v3(state);
    assert(limited_response.search_nodes >= 0);
    assert(limited_response.root_candidates_evaluated <= limited_response.root_candidates_total);
    assert(limited_response.configured_time_budget_ms == limited_cfg.time_budget_ms_response);
    assert(limited_response.configured_node_budget == limited_cfg.max_nodes_response);
    assert(limited_response.truncated);
    assert(limited_response.fallback_reason == "time_budget_exceeded");
    assert(limited_response.truncate_reason == "time_budget_exceeded");
    assert(limited_response.action == "pass");

    linhai::LinhaiSearchEngineV3 limited_discard_engine;
    linhai::SearchConfig limited_discard_cfg;
    limited_discard_cfg.time_budget_ms_discard = 0;
    limited_discard_engine.set_search_config(limited_discard_cfg);
    auto limited_discard = limited_discard_engine.recommend_discard_v3(state);
    assert(limited_discard.search_nodes >= 0);
    assert(limited_discard.root_candidates_evaluated <= limited_discard.root_candidates_total);
    assert(limited_discard.configured_time_budget_ms == limited_discard_cfg.time_budget_ms_discard);
    assert(limited_discard.configured_node_budget == limited_discard_cfg.max_nodes_discard);
    assert(limited_discard.truncated);
    assert(limited_discard.fallback_reason == "time_budget_exceeded");
    assert(limited_discard.truncate_reason == "time_budget_exceeded");
    assert(limited_discard.action == "discard");
    assert(limited_discard.hai > 0);

    linhai::LinhaiSearchEngineV3 node_limited_response_engine;
    linhai::SearchConfig node_limited_response_cfg;
    node_limited_response_cfg.time_budget_ms_response = -1;
    node_limited_response_cfg.max_nodes_response = 0;
    node_limited_response_engine.set_search_config(node_limited_response_cfg);
    auto node_limited_response = node_limited_response_engine.recommend_response_v3(state);
    assert(node_limited_response.search_nodes >= 0);
    assert(node_limited_response.root_candidates_evaluated <= node_limited_response.root_candidates_total);
    assert(node_limited_response.configured_time_budget_ms == node_limited_response_cfg.time_budget_ms_response);
    assert(node_limited_response.configured_node_budget == node_limited_response_cfg.max_nodes_response);
    assert(node_limited_response.truncated);
    assert(node_limited_response.fallback_reason == "node_budget_exceeded");
    assert(node_limited_response.truncate_reason == "node_budget_exceeded");

    linhai::LinhaiSearchEngineV3 node_limited_discard_engine;
    linhai::SearchConfig node_limited_discard_cfg;
    node_limited_discard_cfg.time_budget_ms_discard = -1;
    node_limited_discard_cfg.max_nodes_discard = 0;
    node_limited_discard_engine.set_search_config(node_limited_discard_cfg);
    auto node_limited_discard = node_limited_discard_engine.recommend_discard_v3(state);
    assert(node_limited_discard.search_nodes >= 0);
    assert(node_limited_discard.root_candidates_evaluated <= node_limited_discard.root_candidates_total);
    assert(node_limited_discard.configured_time_budget_ms == node_limited_discard_cfg.time_budget_ms_discard);
    assert(node_limited_discard.configured_node_budget == node_limited_discard_cfg.max_nodes_discard);
    assert(node_limited_discard.truncated);
    assert(node_limited_discard.fallback_reason == "node_budget_exceeded");
    assert(node_limited_discard.truncate_reason == "node_budget_exceeded");
    assert(node_limited_discard.action == "discard");
    return 0;
}
