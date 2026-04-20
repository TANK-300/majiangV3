# Canonical State

`CanonicalGameState` 统一描述：

- 当前玩家手牌、副露、自风、墙剩余
- 所有已知弃牌与副露
- 精确剩余牌池
- `can_win / pass_hu`
- 响应上下文：`target_hai / from_player / available_actions`
- 临海扩展上下文：`white_tiles_in_hand / tree_active / grab_charge_active`
- 风险扩展上下文：`contract_target_count / contract_counter / opponent_meld_count / opponent_discard_count`

离线抽取工具 `tools/extract_canonical_states.py` 当前会产出：

- `current_player.hand / hand_counts / melds`
- `player_snapshots`
- `visible_counts / remaining_counts`
- `opponent_discards`
- `model_features`
- `label / source_meta`
