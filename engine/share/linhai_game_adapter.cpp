#include "linhai_game_adapter.hpp"
#include <algorithm>
#include <cassert>

namespace linhai {

bool is_linhai_valid_tile(int hai) {
    if (hai <= 0 || hai >= 38) return false;
    if (hai % 10 == 0) return false;  // 赤牌编码(10,20,30)跳过
    // 万子: 1-9
    if (hai >= 1 && hai <= 9) return true;
    // 筒子: 11-19 不存在
    if (hai >= 11 && hai <= 19) return false;
    // 条子: 21-29
    if (hai >= 21 && hai <= 29) return true;
    // 字牌: 31-37
    if (hai >= 31 && hai <= 37) return true;
    return false;
}

Game_State to_akochan_game_state(const GameState& ls) {
    Game_State gs;

    // 基本信息
    gs.bakaze = 0;  // 东风
    gs.kyoku = 1;   // 第1局
    gs.honba = 0;
    gs.kyotaku = 0;
    gs.dora_marker.clear();

    // 设置4个玩家
    // Player 0: 当前玩家 (ls.current_player映射到0)
    // Player 1: 对手
    // Player 2,3: 虚拟玩家

    const int my_pid = 0;  // 在Akochan中固定为player 0

    // Player 0: 当前玩家
    gs.player_state[0].score = 25000;
    gs.player_state[0].jikaze = ls.jikaze;
    gs.player_state[0].tehai = ls.tehai;
    gs.player_state[0].fuuro = ls.fuuro;
    gs.player_state[0].reach_declared = false;
    gs.player_state[0].reach_accepted = false;

    // 从 all_players 中获取对手信息
    // 构建对手牌河 (用空Sutehai填充，因为我们只知道数量)
    gs.player_state[1].score = 25000;
    gs.player_state[1].jikaze = (ls.jikaze + 1) % 4;
    // 对手手牌我们不知道，清零
    for (int i = 0; i < 38; i++) {
        gs.player_state[1].tehai[i] = 0;
    }
    gs.player_state[1].reach_declared = false;
    gs.player_state[1].reach_accepted = false;

    if (ls.all_players.size() > 1) {
        const Player_State& opp = ls.all_players[1];
        gs.player_state[1].reach_declared = opp.reach_declared;
        gs.player_state[1].reach_accepted = opp.reach_accepted;
        gs.player_state[1].fuuro = opp.fuuro;
        gs.player_state[1].kawa = opp.kawa;
    }

    // Player 0 的牌河
    if (ls.all_players.size() > 0) {
        gs.player_state[0].kawa = ls.all_players[0].kawa;
    }

    // Player 2,3: 虚拟空玩家
    for (int pid = 2; pid < 4; pid++) {
        gs.player_state[pid].score = 25000;
        gs.player_state[pid].jikaze = (ls.jikaze + pid) % 4;
        for (int i = 0; i < 38; i++) {
            gs.player_state[pid].tehai[i] = 0;
        }
        gs.player_state[pid].reach_declared = false;
        gs.player_state[pid].reach_accepted = false;
    }

    // 设置自风
    gs.set_all_jikaze(ls.jikaze);  // 庄家 = 东位

    return gs;
}

Moves build_minimal_moves(const GameState& ls) {
    Moves moves;

    // 1. start_kyoku 事件
    // 这是 Akochan 最基本的需求：确定局的基本信息
    json11::Json::object start_kyoku;
    start_kyoku["type"] = json11::Json("start_kyoku");
    start_kyoku["bakaze"] = json11::Json("E");
    start_kyoku["dora_marker"] = json11::Json(json11::Json::array{json11::Json("5m")});
    start_kyoku["kyoku"] = json11::Json(1);
    start_kyoku["honba"] = json11::Json(0);
    start_kyoku["kyotaku"] = json11::Json(0);
    start_kyoku["oya"] = json11::Json(0);

    // tehais: 4个玩家的手牌
    json11::Json::array all_tehais;
    for (int pid = 0; pid < 4; pid++) {
        json11::Json::array tehai_arr;
        if (pid == 0) {
            // 当前玩家手牌
            for (int hai = 1; hai < 38; hai++) {
                if (!is_linhai_valid_tile(hai)) continue;
                for (int c = 0; c < ls.tehai[hai]; c++) {
                    tehai_arr.push_back(json11::Json(hai_int_to_str(hai)));
                }
            }
            // 补齐到13张 (如果副露了会少于13)
        } else {
            // 对手和虚拟玩家: 不知道手牌
            for (int i = 0; i < 13; i++) {
                tehai_arr.push_back(json11::Json("?"));
            }
        }
        all_tehais.push_back(json11::Json(tehai_arr));
    }
    start_kyoku["tehais"] = json11::Json(all_tehais);

    // scores
    json11::Json::array scores;
    for (int i = 0; i < 4; i++) {
        scores.push_back(json11::Json(25000));
    }
    start_kyoku["scores"] = json11::Json(scores);

    moves.push_back(json11::Json(start_kyoku));

    // 2. 添加对手的打牌记录 (如果有)
    if (ls.all_players.size() > 0) {
        // 自己的牌河
        for (const auto& sutehai : ls.all_players[0].kawa) {
            json11::Json::object dahai;
            dahai["type"] = json11::Json("dahai");
            dahai["actor"] = json11::Json(0);
            dahai["pai"] = json11::Json(hai_int_to_str(sutehai.hai));
            dahai["tsumogiri"] = json11::Json(sutehai.tsumogiri);
            moves.push_back(json11::Json(dahai));

            // 在每次打牌后，下家需要摸牌
            json11::Json::object tsumo_next;
            tsumo_next["type"] = json11::Json("tsumo");
            tsumo_next["actor"] = json11::Json(1);
            tsumo_next["pai"] = json11::Json("?");
            moves.push_back(json11::Json(tsumo_next));
        }
    }

    if (ls.all_players.size() > 1) {
        // 对手的牌河
        for (const auto& sutehai : ls.all_players[1].kawa) {
            json11::Json::object dahai;
            dahai["type"] = json11::Json("dahai");
            dahai["actor"] = json11::Json(1);
            dahai["pai"] = json11::Json(hai_int_to_str(sutehai.hai));
            dahai["tsumogiri"] = json11::Json(sutehai.tsumogiri);
            moves.push_back(json11::Json(dahai));

            // 打牌后自己摸牌
            json11::Json::object tsumo_next;
            tsumo_next["type"] = json11::Json("tsumo");
            tsumo_next["actor"] = json11::Json(0);
            tsumo_next["pai"] = json11::Json("?");
            moves.push_back(json11::Json(tsumo_next));
        }
    }

    // 3. 最后：当前玩家摸牌 (表示轮到自己出牌)
    json11::Json::object tsumo;
    tsumo["type"] = json11::Json("tsumo");
    tsumo["actor"] = json11::Json(0);
    tsumo["pai"] = json11::Json("?");
    moves.push_back(json11::Json(tsumo));

    return moves;
}

Moves build_moves_with_discards(
    const GameState& ls,
    const std::vector<int>& my_discards,
    const std::vector<int>& opponent_discards
) {
    Moves moves;

    // start_kyoku
    json11::Json::object start_kyoku;
    start_kyoku["type"] = json11::Json("start_kyoku");
    start_kyoku["bakaze"] = json11::Json("E");
    start_kyoku["dora_marker"] = json11::Json(json11::Json::array{json11::Json("5m")});
    start_kyoku["kyoku"] = json11::Json(1);
    start_kyoku["honba"] = json11::Json(0);
    start_kyoku["kyotaku"] = json11::Json(0);
    start_kyoku["oya"] = json11::Json(0);

    json11::Json::array all_tehais;
    for (int pid = 0; pid < 4; pid++) {
        json11::Json::array tehai_arr;
        if (pid == 0) {
            for (int hai = 1; hai < 38; hai++) {
                if (!is_linhai_valid_tile(hai)) continue;
                for (int c = 0; c < ls.tehai[hai]; c++) {
                    tehai_arr.push_back(json11::Json(hai_int_to_str(hai)));
                }
            }
        } else {
            for (int i = 0; i < 13; i++) {
                tehai_arr.push_back(json11::Json("?"));
            }
        }
        all_tehais.push_back(json11::Json(tehai_arr));
    }
    start_kyoku["tehais"] = json11::Json(all_tehais);

    json11::Json::array scores;
    for (int i = 0; i < 4; i++) scores.push_back(json11::Json(25000));
    start_kyoku["scores"] = json11::Json(scores);
    moves.push_back(json11::Json(start_kyoku));

    // 交替添加双方打牌记录 (模拟真实对局流程)
    const size_t max_turns = std::max(my_discards.size(), opponent_discards.size());
    for (size_t turn = 0; turn < max_turns; turn++) {
        // 自己打牌
        if (turn < my_discards.size()) {
            json11::Json::object dahai;
            dahai["type"] = json11::Json("dahai");
            dahai["actor"] = json11::Json(0);
            dahai["pai"] = json11::Json(hai_int_to_str(my_discards[turn]));
            dahai["tsumogiri"] = json11::Json(false);
            moves.push_back(json11::Json(dahai));

            // 对手摸牌
            json11::Json::object tsumo_opp;
            tsumo_opp["type"] = json11::Json("tsumo");
            tsumo_opp["actor"] = json11::Json(1);
            tsumo_opp["pai"] = json11::Json("?");
            moves.push_back(json11::Json(tsumo_opp));
        }

        // 对手打牌
        if (turn < opponent_discards.size()) {
            json11::Json::object dahai;
            dahai["type"] = json11::Json("dahai");
            dahai["actor"] = json11::Json(1);
            dahai["pai"] = json11::Json(hai_int_to_str(opponent_discards[turn]));
            dahai["tsumogiri"] = json11::Json(false);
            moves.push_back(json11::Json(dahai));

            // 自己摸牌
            json11::Json::object tsumo_me;
            tsumo_me["type"] = json11::Json("tsumo");
            tsumo_me["actor"] = json11::Json(0);
            tsumo_me["pai"] = json11::Json("?");
            moves.push_back(json11::Json(tsumo_me));
        }
    }

    // 最终: 当前玩家摸牌
    json11::Json::object tsumo;
    tsumo["type"] = json11::Json("tsumo");
    tsumo["actor"] = json11::Json(0);
    tsumo["pai"] = json11::Json("?");
    moves.push_back(json11::Json(tsumo));

    return moves;
}

Hai_Array get_linhai_visible_tiles(const GameState& ls) {
    Hai_Array visible = {0};

    // 自己手牌
    for (int hai = 1; hai < 38; hai++) {
        visible[hai] += ls.tehai[hai];
    }

    // 自己副露
    for (const auto& f : ls.fuuro) {
        for (int c : f.consumed) {
            if (c > 0 && c < 38) visible[c]++;
        }
        if (f.hai > 0 && f.hai < 38) visible[f.hai]++;
    }

    // 所有玩家的牌河和副露
    for (const auto& player : ls.all_players) {
        for (const auto& sutehai : player.kawa) {
            if (sutehai.hai > 0 && sutehai.hai < 38) {
                visible[sutehai.hai]++;
            }
        }
        for (const auto& f : player.fuuro) {
            for (int c : f.consumed) {
                if (c > 0 && c < 38) visible[c]++;
            }
            if (f.hai > 0 && f.hai < 38) visible[f.hai]++;
        }
    }

    return visible;
}

Hai_Array get_linhai_remaining_pool(const GameState& ls) {
    Hai_Array visible = get_linhai_visible_tiles(ls);
    Hai_Array remaining = {0};

    for (int i = 0; i < LINHAI_VALID_TILE_COUNT; i++) {
        int hai = LINHAI_VALID_TILES[i];
        remaining[hai] = std::max(0, 4 - visible[hai]);
    }

    return remaining;
}

} // namespace linhai
