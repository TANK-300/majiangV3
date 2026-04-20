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

## 已知仍欠缺（留给 A-2b / B 期）

- 精确向听数和受入分布 per-tile（目前 C++ 侧已算 cached_shanten，但没作为特征暴露给模型）
- 筋/壁/现物（需要对每张牌做规则判定）
- 副露类型（peng/chi/gang/anka）one-hot

