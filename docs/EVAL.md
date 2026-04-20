# Evaluation

当前验收关注：

- 关键牌例决策是否稳定
- V3 对比 V2 的候选排序差异
- 回退链是否正常
- 响应和弃牌接口是否统一输出调试信息

当前离线评估脚本 `tools/eval_search.py` 支持：

- 按 `record_id` 对齐预测与标签
- `best_discard_tile` 的 top-1 命中率
- `best_response_action` 的 top-1 命中率
- `best_response_discard_tile` 的 top-1 命中率
- `can_win_label` 的准确率
- `agari_prob / tenpai_prob / houjuu_prob / betaori_prob / ryukyoku_prob` 的：
  - 样本数
  - `log_loss`
  - `brier`
- `tsumo_num` 的：
  - 样本数
  - `mae`
  - `rmse`
- 搜索行为指标：
  - `fallback_rate`
  - `truncate_rate`
  - `avg_search_nodes`
