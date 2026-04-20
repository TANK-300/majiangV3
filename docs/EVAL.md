# Evaluation

当前验收关注：

- 关键牌例决策是否稳定
- V3 对比 V2 的候选排序差异
- 回退链是否正常
- 响应和弃牌接口是否统一输出调试信息
- 自对弈胜率/放炮率/自摸率（定期跑 `tools/selfplay_eval.py`）

## 自对弈胜率回归（tools/selfplay_eval.py）

该脚本在一个简化但**对称**的双人临海规则下跑两个 policy 对打 N 局，并输出：

- `winrate_a / winrate_b / draw_rate`
- `tsumo_a / tsumo_b`（自摸数）
- `ron_a / ron_b`（荣胡数）
- `houjuu_rate_a / houjuu_rate_b`（放炮率）
- `avg_turns / median_turns`
- 如果任一 policy 使用 `orchestrator`，还会附带 `orchestrator_status`，报告当前生效的引擎（`v3` / `v2` / `heuristic`）

Policy 支持三种：

- `heuristic`：纯 Python 启发式，等价于 `StrategyFallbackService`；
- `random`：随机打牌，用来做最底线参考；
- `orchestrator` / `orchestrator:标签`：走完整的 `LinhaiV3Orchestrator`，自动选择 V3/V2/heuristic。

典型用法：

```
# 基线：heuristic 对 heuristic，应接近 50%
python3 tools/selfplay_eval.py --policy-a heuristic --policy-b heuristic --games 200 --seed 1

# 新版 V3 对 baseline：
python3 tools/selfplay_eval.py --policy-a orchestrator:v3 --policy-b heuristic --games 500 --seed 1
```

**注意**：这里的"胜率"是简化规则下的相对度量，不是生产环境的绝对胜率；它的价值在于同一脚本在两个 policy 上给出的差值。

## /debug/engine_status 端点

生产环境诊断"胜率不对劲"时，先 GET `/debug/engine_status`。如果返回的 `active_engine == "heuristic"`，说明 C++ 扩展没加载成功，整个服务其实在跑启发式，胜率差异就不奇怪。`v3.reason` 字段会告诉你具体失败原因（例如 `import_failed:...` 或 `params_not_found`）。

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
