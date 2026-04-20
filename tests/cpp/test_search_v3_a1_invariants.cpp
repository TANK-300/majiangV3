// A-1 regression test: lock in the two invariants we fixed in the
// "精确向听 + chance node 概率守恒" PR.
//
// 1. The shanten number the search reports on the best candidate must equal
//    the exact calc_linhai_shanten() value. Before A-1 the search used a
//    coarse estimator that could be off by multiple shanten.
// 2. When the chance tree is beam-truncated, the returned EV must not drop
//    proportionally to uncovered probability mass. We exercise this by
//    running the same state with a tiny beam and a large beam and confirming
//    the reported total_ev is within the same order of magnitude (before
//    A-1, a tiny beam would produce an EV roughly `beam/25` of the large
//    beam's, which is an obvious bug).

#include "share/linhai_search_v3.hpp"
#include "share/linhai_shanten_v4.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using linhai::CanonicalGameState;
    using linhai::LinhaiSearchEngineV3;
    using linhai::SearchConfig;
    using linhai::SearchResult;

    // Hand where the fast estimator is known to be off:
    // 1w 2w 3w 4w 5w 6w 7w 8w 9w 1t 2t 3t white = 2-shanten-ish,
    // but fast estimator returns a larger number because it counts naive
    // meld_like pairs rather than overlapping runs.
    CanonicalGameState state;
    state.game_state.tehai[1] = 1;
    state.game_state.tehai[2] = 1;
    state.game_state.tehai[3] = 1;
    state.game_state.tehai[4] = 1;
    state.game_state.tehai[5] = 1;
    state.game_state.tehai[6] = 1;
    state.game_state.tehai[7] = 1;
    state.game_state.tehai[8] = 1;
    state.game_state.tehai[9] = 1;
    state.game_state.tehai[21] = 1;
    state.game_state.tehai[22] = 1;
    state.game_state.tehai[23] = 1;
    state.game_state.tehai[24] = 1;
    state.game_state.tehai[35] = 1;  // white acts as wildcard
    state.game_state.wall_remaining = 30;
    state.can_win = true;
    state.refresh_counts();

    LinhaiSearchEngineV3 engine;
    SearchConfig cfg;
    cfg.time_budget_ms_discard = -1;
    cfg.time_budget_ms_response = -1;
    engine.set_search_config(cfg);

    SearchResult discard = engine.recommend_discard_v3(state);
    assert(discard.action == "discard");

    // Invariant 1: reported shanten matches exact calc_linhai_shanten applied
    // to the post-discard hand.
    Hai_Array after = state.game_state.tehai;
    if (discard.hai > 0 && discard.hai < 38 && after[discard.hai] > 0) {
        after[discard.hai] -= 1;
    }
    const int exact_shanten = calc_linhai_shanten(after);
    if (discard.shanten != exact_shanten) {
        std::cerr << "INVARIANT 1 FAILED: reported shanten=" << discard.shanten
                  << " exact=" << exact_shanten << "\n";
        return 1;
    }

    // Invariant 2: chance-tree beam width should not linearly scale EV.
    LinhaiSearchEngineV3 wide_engine;
    SearchConfig wide_cfg = cfg;
    wide_cfg.beam_width_after_draw = 25;    // cover every valid tile
    wide_cfg.beam_width_far_shanten = 25;
    wide_engine.set_search_config(wide_cfg);
    SearchResult wide = wide_engine.recommend_discard_v3(state);

    LinhaiSearchEngineV3 narrow_engine;
    SearchConfig narrow_cfg = cfg;
    narrow_cfg.beam_width_after_draw = 1;
    narrow_cfg.beam_width_far_shanten = 1;
    narrow_engine.set_search_config(narrow_cfg);
    SearchResult narrow = narrow_engine.recommend_discard_v3(state);

    // Even if the search truncates, the EV must be on the same order of
    // magnitude -- unbiased probability-mass extrapolation prevents the
    // previous "divide by full pool but sum beam only" bug.
    const float ratio = std::fabs(wide.total_ev) > 1e-3f
                            ? std::fabs(narrow.total_ev) / std::fabs(wide.total_ev)
                            : 1.0f;
    if (ratio < 0.2f) {
        std::cerr << "INVARIANT 2 FAILED: narrow_ev=" << narrow.total_ev
                  << " wide_ev=" << wide.total_ev
                  << " ratio=" << ratio << "\n";
        return 1;
    }

    std::cout << "A1_INVARIANTS_PASS (shanten=" << discard.shanten
              << " narrow_ev=" << narrow.total_ev
              << " wide_ev=" << wide.total_ev
              << " ratio=" << ratio << ")\n";
    return 0;
}
