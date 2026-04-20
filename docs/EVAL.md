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

## 自对弈样本生成（tools/selfplay_sample.py, B-1）

`tools/selfplay_eval.py` 只给胜率汇总，`tools/selfplay_sample.py` 在同样的对局模拟之上，把**每一个打牌决策**的 `(state, action, outcome)` 三元组写到 JSONL，供离线训练器（`tools/train_model_stub.py`）和未来的 B-2 神经网络直接消费。

```
python3 tools/selfplay_sample.py --policy-a heuristic --policy-b heuristic \
    --games 1000 --seed 1 --output /tmp/samples.jsonl

python3 tools/train_model_stub.py --task agari_prob \
    --version-dir /tmp/params --dataset /tmp/samples.jsonl
```

单条样本字段（关键项）：

- `record_id` / `game_index` / `step_index` / `seat_wind`
- `model_features`：与 `extract_canonical_states` 一致的特征字典（可直接喂训练器）
- `outcome`：`win / loss / draw`、`tsumo / ron / draw`、距离终局步数
- `task_labels.can_win_label`、`source_meta.houjuu_label`、`source_meta.tsumo_num_label`、`source_meta.ryukyoku_label`、`source_meta.betaori_label`

端到端 smoke：200 局自对弈 → 训练 `agari_prob`，accuracy 达到 67%（positive rate 49% 基线），说明数据链有效。

## Python 神经网络 policy（B-2a）

`tools/train_neural.py` 读 B-1 产出的 JSONL，训练一个多头 MLP（`agari / houjuu / tsumo_num`），导出 ONNX + `model_meta.json`。然后可以用 `neural:/path/to/bundle` 作为 selfplay 的 policy：

```
# 1. 生成大规模自对弈样本
python3 tools/selfplay_sample.py --policy-a heuristic --policy-b heuristic \
    --games 5000 --seed 1 --output samples.jsonl

# 2. 训练 + 导出 ONNX
python3 tools/train_neural.py --dataset samples.jsonl --output-dir params/neural_v1 --epochs 30 --hidden 128

# 3. 对战 heuristic
python3 tools/selfplay_eval.py --policy-a neural:params/neural_v1 --policy-b heuristic --games 1000 --seed 1
```

样本 1000 局、hidden=128、30 epoch 时，`val_agari_acc ≈ 0.85`、`val_houjuu_acc ≈ 0.76`。

**已知 caveat**：直接用 heuristic-vs-heuristic 产生的样本训出来的 NN 在对打 heuristic 时胜率可能低于 50%。原因是训练样本里 heuristic 放炮率本来就很低，NN 学到的 houjuu 先验偏弱 → 打牌时不够防守。正确做法（留给 B-2b / Plan B 迭代）：用 NN 自对弈样本做第 2 轮训练，重新校准 houjuu 分布；或者把训练样本改成 heuristic vs random（放炮率更均衡）。

## C++ 侧集成（B-2b，未完成）

`tools/train_neural.py` 产出的 ONNX 会在 B-2b 被 C++ `LinhaiSearchEngineV3` 通过 onnxruntime 加载，替换 `V3LinearModel` 的 `predict_model`。届时生产 V3 就能用神经网络估值。

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
