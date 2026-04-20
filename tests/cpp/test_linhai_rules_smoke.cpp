#include "share/linhai_rules.hpp"
#include <algorithm>
#include <cassert>

int main() {
    assert(WHITE_TILE == 35);

    Hai_Array all_wait_hand = {0};
    all_wait_hand[1] = 1;
    all_wait_hand[2] = 1;
    all_wait_hand[3] = 1;
    all_wait_hand[4] = 1;
    all_wait_hand[5] = 1;
    all_wait_hand[6] = 1;
    all_wait_hand[7] = 1;
    all_wait_hand[8] = 1;
    all_wait_hand[9] = 1;
    all_wait_hand[21] = 1;
    all_wait_hand[22] = 1;
    all_wait_hand[23] = 1;
    all_wait_hand[35] = 1;

    Tenpai_Info white_single_waits = linhai_cal_tenpai_info(31, 31, all_wait_hand, {});
    assert(white_single_waits.mentu_shanten_num == 0);
    assert(!white_single_waits.agari_vec.empty());
    assert(std::any_of(
        white_single_waits.agari_vec.begin(),
        white_single_waits.agari_vec.end(),
        [](const Agari_Info& info) { return info.hai == 31; }
    ));
    assert(std::any_of(
        white_single_waits.agari_vec.begin(),
        white_single_waits.agari_vec.end(),
        [](const Agari_Info& info) { return info.hai == 35; }
    ));

    std::vector<int> north_tiles = get_grab_charge_tiles(3);
    assert(std::find(north_tiles.begin(), north_tiles.end(), 35) != north_tiles.end());
    assert(std::find(north_tiles.begin(), north_tiles.end(), 37) == north_tiles.end());

    std::vector<int> south_tiles = get_grab_charge_tiles(1);
    assert(std::find(south_tiles.begin(), south_tiles.end(), 37) != south_tiles.end());
    assert(std::find(south_tiles.begin(), south_tiles.end(), 35) == south_tiles.end());

    return 0;
}
