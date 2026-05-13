# 临海 V3 算法胜率优化方向

> 日期：2026-05-13
> 当前分支：`feature/algorithm-power`
> 验收基线（4 人临海原始 V3，无 adapter，来源 `docs/power/release/acceptance_report_5seed.json`）：
> 1000 局 / 125 个 8-局窗口，**正积分窗口 106 / 125 = 84.8 %**
> 当前 (5-12，5 seed × 200 局)：vs `reference_human`，无缺一门 →
> 窗口正率 **76.8 %**、winrate **64.4 %**、houjuu **15.0 %**、avg_chong **3.17**

## 一、验收指标（参考图 / 原始 V3 baseline）

| 规模 | 结算次数 | 正积分次数 | 正积分概率 |
|---|---|---|---|
| 200 局 / seed 1 | 25 | 21 | 84.0 % |
| 200 局 / seed 2 | 25 | 19 | 76.0 % |
| 200 局 / seed 3 | 25 | 24 | 96.0 % |
| 200 局 / seed 4 | 25 | 19 | 76.0 % |
| 200 局 / seed 5 | 25 | 23 | 92.0 % |
| **1000 局合计** | **125** | **106** | **84.8 %** |

> 目标：每 8 局结算正积分概率 ≥ 84 %，向 1000 局合计 90 %+ 推进。

## 二、当前两个事实

1. **harness 是 2 人对战**（`selfplay_eval.py` EAST vs SOUTH），但送进 V3 引擎的 canonical state 仍然是 4 座位（WEST/NORTH 用空 snapshot 占位）。V3 引擎本身是 4 人临海训练的，**从未见过真正的 2 人玩法**；`two_player_adapter` 是事后补丁。
2. **现行 baseline 比原始 V3 baseline 低 8 个点**（76.8 % vs 84.8 %）。变化来源是 commit `98b6d75 算法优化`（2026-05-09），把 `adjust_for_2p_linhai` 接到 orchestrator 上。原始 V3 baseline `acceptance_report_5seed.json`（2026-04-28）在那之前生成，没经过 adapter。

## 三、八个优化方向（按性价比从高到低）

### 🔴 立刻可测试（5–15 分钟）

#### 1. 关掉 adapter 跑 V3 原始输出 ⭐ 当前优先

- 假设：adapter 的"孤张字牌 +800" / betaori 等 always-on 规则在 V3 已有 EV 上是双重计分，污染了 V3 训练好的字牌策略。
- 改法：`orchestrator.recommend` 在 `mode is None` 且 `missing_suit_self/opp` 都为 None 时**跳过 adapter**，直接返回 v3_result。
- 预期：回到 ~84 % 窗口正率（恢复原始 V3 baseline）。
- 风险：低。变更仅 orchestrator 一个分支判断，selfplay 默认 `missing_suit_enabled=False` 时触发新路径；缺一门场景仍走 adapter。
- 验证文件：`docs/power/release/acceptance_no_adapter_5seed.json`

#### 2. 调 `_honor_bonus_for_count` 字牌奖励

- 假设：现在 1/2/3/4 张字牌的奖励 `+800 / -200 / -800 / -1500` 是 2P 启发式拍的，可能与 V3 内部字牌权重冲突。
- 改法：先关 adapter（#1 完成），如果 #1 提升不够，再尝试单独把字牌奖励重新拍：
  - 孤字牌：+400（弱化一半）
  - 字牌对子：0（让 V3 EV 决定）
  - 字牌刻子/杠：保持
- 预期：在 #1 之后再加 1–2 个点。
- 验证文件：`docs/power/release/acceptance_honor_tuned_5seed.json`

#### 3. V3 切 `deep` profile

- 假设：现在 `prod` profile 是 `depth=1、beam=2`；`deep` 是 `depth=2、beam=4、near_ready_depth=3`，搜索更深。
- 改法：环境变量 `LINHAI_V3_PROFILE=deep` 跑同样回归。
- 预期：+1–3 个点；代价 ~5x 延迟（200 局约 1 小时）。
- 风险：低，只是搜索深度增加。
- 验证文件：`docs/power/release/acceptance_deep_5seed.json`

### 🟡 中等改造（数小时—1 天）

#### 4. 调 V3 内部 EV 系数（2P 平衡）

- V3 的 `houjuu_penalty` 是按 3 对手训练的；2P 实际放炮风险约 1/3。
- 改法：在候选打分上加 `+ houjuu_prob * α`（α ≈ 系数 × (1 − 1/3)），把 V3 估出来的"3 对手放炮代价"反向折回 2P 实际水平。
- 比 `TWO_PLAYER_HOUJUU_FACTOR=0.4`（只改返回值不改排序）有效得多——后者只是装饰。

#### 5. 跑 V3 vs heuristic 测真实上限

- 现在 vs `reference_human`（= V2 + betaori，已经很强），winrate 64.4 %。
- 换成 vs `heuristic`（简单弃牌规则）：
  - 若 winrate > 85 % → V3 没问题，差距来自对手强度，前面优化方向有效。
  - 若 winrate ≤ 70 % → V3 本身就弱，必须重训。
- 用途：诊断瓶颈在 "V3 本身" 还是 "对手太强"。

#### 6. 启用 NeuralPolicy (B-2a)

- `selfplay_eval` 已实现 ONNX-backed NeuralPolicy，需要训练好的 bundle。
- 跑 `tools/train_neural.py` 生成 bundle → V3 vs neural 看相对强度。

### 🟢 大改（数天）

#### 7. V3 canonical state 真正改成 2 人

- `v3_service.py` 的 `wind_order` 从 4 风改为 2 风（EAST/SOUTH），重新生成训练数据，蒸馏 2P 专用参数包。
- 这是 adapter 文档里写的 P1 阶段。

#### 8. R3+ 迭代自对弈训练

- R0-R2 已完成（见 `data/iter_log_R1R2R3.txt`、`docs/power/release/r1_eval_5seed.json`），可接着 R3 用更强 V3 自对弈蒸馏。

## 四、执行顺序

按用户要求：

1. **先做 #1**（关 adapter），跑结果 → 对照 84.8 % 基线。
2. **再做 #2**（调字牌奖励），跑结果。
3. **最后做 #3**（deep profile），跑结果。
4. 三个都做完再决定是否进入 #4-#8。

## 五、验收命令模板

```bash
python3 tools/acceptance_25_8.py \
  --games-per-seed 200 --seeds 1 2 3 4 5 \
  --policy-a orchestrator:v3 --policy-b reference_human \
  --min-window-chong 2 --min-avg-chong 0.5 --max-houjuu-rate 0.12 \
  --output docs/power/release/acceptance_<label>_5seed.json
```

每次跑完对比四个指标：**窗口正率 / winrate / houjuu / avg_chong**。

## 六、执行记录（边做边填）

| 步骤 | 配置 | 窗口正率 | winrate | houjuu | avg_chong | 结论 |
|---|---|---|---|---|---|---|
| 0 基线 | 原始 V3，无 adapter（4-28 报告） | 84.8 % | 68.4 % | 14.8 % | 3.97 | 目标 |
| 0 当前 | V3 + adapter（5-12 跑） | 76.8 % | 64.4 % | 15.0 % | 3.17 | 退步 -8pp |
| **#1 关 adapter** | orchestrator bypass adapter when no 2P signal | **84.0 %** | **67.1 %** | **13.5 %** | **3.62** | ✅ +7.2pp，回到基线水平 |
| #2 调字牌奖励 | _跳过_（#1 已经关 adapter） | – | – | – | – | – |
| **#3 deep profile** | `LINHAI_V3_PROFILE=deep`（depth=2、beam=4） | **88.0 %** | **66.9 %** | **12.7 %** | **3.66** | ✅ +4pp，超过历史基线 +3.2pp，主要靠降放炮 |
| #3 独立 500 局确认 | 同上，5 seed × 100 局 | 85.0 % | 67.2 % | 14.2 % | 3.58 | ✓ 与 1000 局 88% 在采样方差内 |
| #5 V3 vs heuristic | deep + 换对手 heuristic | 88.0 % | 71.9 % | 10.9 % | 4.70 | ⚠️ 窗口正率持平 88 %、winrate +5 pp、avg_chong +1.0 — 诊断：88 % 是 V3 内在极限 |
| #4 EV 重平衡 (rebal=1000) | `LINHAI_V3_REBALANCE_FACTOR=1000` | 87.2 % | 67.0 % | 12.7 % | 3.70 | ❌ −0.8pp，因素太弱推不动排序 |
| #4 EV 重平衡 (rebal=3000) | `LINHAI_V3_REBALANCE_FACTOR=3000` | 88.0 % | 67.1 % | 13.3 % | 3.71 | ❌ 与 #3 持平、houjuu 略升 → 重平衡无法挪动天花板 |

## 七、关键结论 (2026-05-13)

经过 #1 / #3 / #5 / #4 四步实验：

1. **adapter 是污染源** —— always-on 的孤张字牌 +800 / betaori 等规则与 V3 内部 EV 冲突，关掉直接 +7.2 pp（76.8 → 84.0）。生产应保持 orchestrator 的 bypass。
2. **deep profile 再 +4 pp** —— `LINHAI_V3_PROFILE=deep` 把搜索深度从 1 提到 2、beam 从 2 提到 4，主要靠降放炮（15 → 12.7 %）拿到 88 %。代价是 ~5x 延迟。
3. **88 % 是 V3 内在极限** —— vs heuristic（弱对手）窗口正率仍是 88 %；EV 重平衡（factor=1000/3000）也推不动。winrate 67 % 是当前参数包对 reference_human 的真实战力。
4. **EV 重平衡作为 adapter 层补丁无效** —— V3 内部 EV 平衡在当前 canonical state（4 座 2 活 2 空）下已接近 2P 最优，加权 houjuu_prob 仅产生噪声级抖动。

要继续往上推必须**改 V3 参数包**：
- **#8 R3+ 迭代自对弈**（基于现有 R0-R2 训练管线，~1 天）
- **#7 V3 canonical 真改 2 座 + 重训**（adapter 文档 P1 阶段，~数天）
- **#6 NeuralPolicy** 替代搜索引擎（ONNX bundle，~1 天，收益未知）
