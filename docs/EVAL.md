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

## A-2b：GBDT 模型的训练→上线流水线

A-2b 把六个 head 切换到 LightGBM GBDT，可以学到"丢单张字牌最优"这类非
线性规则。线性模型做不到这一点——参见下方的退化样例。

### 退化样例（PR-1 之前）

手牌：`2w 3w 6w 6w 7w 7w 8w 4t 4t 5t 5t 9t 9t north`（剩余一张孤张字牌
`north`，对手弃牌里已经出过 7 张筒）。

- 老线性 V3：推荐打 `2w`，`agari_prob=26%`、`houjuu_prob=26.7%`。
  从 Mahjong 打法上看，应该先扔 `north`——它没有任何搭子潜力，且牌河里
  没有对应字牌，放炮风险极低。线性模型只能输出"特征的加权和"，学不到
  "孤张字牌优先丢弃"这种条件规则。
- 新 GBDT V3：推荐打 `north`（hai=34），`chosen_by=search`。
  `agari_gbdt_` 在 `wall_remaining` 高且 `hand_t_north=1` 时命中一个叶子，
  把 agari 贡献压得比 `2w` 高，于是搜索就正确地选择了 `north`。

### 一键训练流水线

```bash
# 1. 生成样本（2-3 万行即可触发 GBDT 路径）
python3 tools/selfplay_sample.py \
    --policy-a heuristic --policy-b random \
    --games 200 --seed 7 \
    --output /tmp/a2b_samples.jsonl

# 2. 训练六个 head
for task in agari_prob tenpai_prob houjuu_prob betaori tsumo_num ryukyoku_prob; do
    python3 tools/train_model_stub.py \
        --task $task \
        --version-dir /tmp/a2b_bundle \
        --dataset /tmp/a2b_samples.jsonl \
        --validation-split 0.2 \
        --model-kind auto \
        --n-estimators 200 --num-leaves 31 --gbdt-min-rows 200
done

# 3. 让 backend 加载新 bundle
export LINHAI_V3_PARAMS_DIR=/tmp/a2b_bundle
# （重启 uvicorn，或在跑 selfplay_eval 时设该环境变量）

# 4. 对局评估
python3 tools/selfplay_eval.py \
    --policy-a orchestrator --policy-b heuristic \
    --games 200 --seed 42
```

### 判断 GBDT 是否真的上线

- `GET /debug/engine_status` 的 `v3.params_dir` 应指向新 bundle 目录。
- `v3.reason == "ok"` 或 `"ok_partial"`（混合 bundle 时）。
- 打开 `$LINHAI_V3_PARAMS_DIR/v3/agari_prob/model.json`：
  `model_type` 必须是 `"lightgbm_gbdt"`；若仍是 `logistic_regression`，说明
  LightGBM 没装（或数据 < `--gbdt-min-rows`）自动回退了。
- 跑 `python3 -m pytest tests/python/test_gbdt_loader.py` 确认 C++ 端到端
  加载路径没坏。

### 常见坑

1. **老部署的 `.so` 覆盖了新编译产物**：如果 `backend/linhai_v3.*.so`
   比 `engine/linhai_v3.*.so` 旧，Python 会优先加载旧的，GBDT loader
   不存在，bundle 会被判"无效"全部走 heuristic。删掉 `backend/*.so`
   或重新 `engine/setup.py build_ext --inplace` 即可。
2. **GBDT 数据量不足时沉默回退到 linear**：`--model-kind=auto` 且行数
   < `--gbdt-min-rows`（默认 200）时，`train_model_stub.py` 会用 sklearn
   linear 兜底。想强制拒绝，用 `--model-kind=gbdt`（会写
   `model_type=gbdt_unavailable`）。
3. **训练/推理特征不一致静默失效**：`V3GBDTModel` 按 `feature_names`
   下标读 `x[feat]`，顺序固定在训练时写入 `model.json`；C++ 推理按名字
   查表再对齐，名字对不上等于拿 0。任何新特征必须同时更新训练侧
   `build_model_features()` 和 C++ 侧 `build_state_features()`。
