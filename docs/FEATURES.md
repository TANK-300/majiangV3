# Features

## 特征族

规划中的临海特征包括：

- 双人局巡目与剩余牌池
- 白板万能牌
- 抓冲与承包
- 过胡不胡约束
- 对手弃牌、现物、壁、筋、防守压力
- 手牌结构、向听、受入与待牌质量

## 训练 / 推理一致性

V3 的模型走"Python 训练 + C++ 推理"两端。特征名**必须完全一致**，否则 `V3LinearModel::predict` 会按名字查不到，权重就等于 0（静默失效）。

两侧实现：

- 训练侧 `tools/extract_canonical_states.py` → `build_model_features()` + `PER_TILE_FEATURE_ORDER`
- 推理侧 `engine/share/linhai_search_v3.cpp` → `LinhaiSearchEngineV3::build_state_features()` + `tile_code_for_feature()`

**任何新特征必须同时加到这两处**，否则就会有隐形 bug。

## A-2a 新增的 per-tile 特征

对每张临海有效牌（25 种，顺序见 `PER_TILE_FEATURE_ORDER`）生成：

| 前缀 | 含义 |
|---|---|
| `hand_t_{tile}` | 该牌在手牌中的数量 |
| `remain_t_{tile}` | 该牌在剩余牌池中的数量 |
| `opp_disc_t_{tile}` | 对手已弃该牌的数量 |

这一层提供了"具体到哪张牌"的信息，原先只有聚合量（`pair_count`、`honor_count` 等）。在不修 C++ 预测器的前提下，这些特征直接进入 `V3LinearModel` 的线性打分里。

## 已知仍欠缺（留给后续期）

- 精确向听数和受入分布 per-tile（目前 C++ 侧已算 cached_shanten，但没作为特征暴露给模型）
- 副露类型（peng/chi/gang/anka）one-hot

## A-3 / 验收：safety per-tile 防守特征族

按 `docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md` §4.4 实现。

| 前缀 | 含义 | 启发 |
|---|---|---|
| `safety_t_X` | X 是否在对手河（现物） | A-2c-3 已有，spec §4.4 沿用 |
| `safety_suji_t_X` | X 被筋（4 出过 → 1/7 安全；5 → 2/8；6 → 3/9） | spec §4.4 新增 |
| `safety_kabe_t_X` | X 邻牌可见 ≥3（壁） | spec §4.4 新增 |

**双侧实现**（必须严格同步，否则 head 静默失效）：
- 训练侧：`tools/extract_canonical_states.py::build_model_features`
- 推理侧：`engine/share/linhai_search_v3.cpp::build_state_features`
- 一致性测试：`tests/python/test_feature_parity.py`（13 用例，CI 必须过）

**特征行为**：
- `safety_t_X`：X 在对手河里出现至少 1 次 → 1.0；否则 0.0
- `safety_suji_t_X`：仅对万 / 条数牌生效，字牌恒 0
- `safety_kabe_t_X`：仅对万 / 条数牌生效，字牌恒 0；可见量 = 4 - remaining_count

## A-2b：GBDT (LightGBM) 模型 schema

A-2b 把六个 head（`agari_prob` / `tenpai_prob` / `houjuu_prob` / `betaori` /
`tsumo_num` / `ryukyoku_prob`）从线性模型升级为 LightGBM GBDT。线性模型
无法表达"丢一张单张字牌最优"这类非线性规则（见 `docs/EVAL.md` 里的
`north` 退化样例）——GBDT 的树结构可以学到这类条件逻辑。

**训练入口**：`tools/train_model_stub.py --model-kind=auto|gbdt|linear`

- `auto`（默认）：优先用 LightGBM GBDT；LightGBM 不可用或行数不足时回退到
  sklearn linear / 手写 SGD，C++ 侧零兼容问题。
- `gbdt`：强制 GBDT；LightGBM 不在时写出 `model_type=gbdt_unavailable`，
  发布脚本能据此拒绝推上线。
- `linear`：完全跳过 GBDT，沿用 A-2a 的 `logistic_regression` / `linear_regression`
  schema；pin 在 PR-2 前的 C++ 部署可以继续用这一条。

**model.json schema**（`model_type == "lightgbm_gbdt"`）：

```json
{
  "task": "agari_prob",
  "label_key": "task_labels.can_win_label",
  "model_type": "lightgbm_gbdt",
  "objective": "binary",            // 或 "regression"（仅 tsumo_num 用）
  "init_score": -0.1234,            // LightGBM 隐式先验；C++ 必须加回去
  "feature_names": ["wall_remaining", ...],
  "trees": [
    {
      "nodes": [
        {"feat": 0, "thr": 20.5, "left": 1, "right": 2, "leaf_value": 0.0},
        {"feat": -1, "thr": 0.0, "left": -1, "right": -1, "leaf_value": -1.2},
        {"feat": -1, "thr": 0.0, "left": -1, "right": -1, "leaf_value":  1.3}
      ]
    },
    ...
  ],
  "n_trees": 50,
  "metrics": {"train_accuracy": 0.87, "val_accuracy": 0.83},
  "fitter": "lightgbm"
}
```

**关键契约**：

- 每棵树的 `nodes[0]` 必定是根；叶子节点用 `feat = -1` + `left = right = -1` 标识。
- `feat` 字段是 `feature_names` 数组的索引（不是名字），C++ 侧
  `predict_gbdt` 会一次性把 `feature_names[i]` 对应的值从 features
  map 查出来填到 `x[i]`，之后的树游走只用 `x[feat]`。
- GBDT bundle **故意不写** `feature_means` / `feature_stds` / `weights`。
  这是"老 linear C++ 加载器拒绝它 → 回退 heuristic"的安全降级阀门。
- 分裂语义固定为 `x[feat] <= thr` 走 `left`，其余走 `right`——和
  `train_model_stub._flatten_lightgbm_tree` 对 LightGBM `decision_type="<="`
  的假定一一对应。

**C++ 推理路径**：`engine/share/linhai_search_v3.cpp`

- `load_v3_head(path, linear_out, gbdt_out)` 根据 JSON `model_type` 字段
  分发到 `load_v3_model`（线性）或 `load_v3_gbdt`（GBDT）。
- `predict_head(linear, gbdt, features)` 的优先级：GBDT 已加载则走 GBDT，
  否则走 linear，否则返回 0（`estimate_*_prob` 已在上层短路到 heuristic）。
- `LinhaiSearchEngineV3` 同时持有六对 `V3LinearModel` 和 `V3GBDTModel`，
  允许混合 bundle（一部分 head GBDT、一部分 head linear）——回归测试
  `tests/python/test_gbdt_loader.py::test_gbdt_loader_mixed_with_linear`
  专门锁这个行为。

