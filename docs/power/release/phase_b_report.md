# Phase B 验收报告（2026-04-30）

> Spec: `docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md` §4  
> Plan: `docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md`  
> Phase A 报告：`docs/release/phase_a_report.md`  
> 训练 bundle：`data/iter_params/R0/`、`R1/`、`R2/`

---

## 1. 执行配置

| 配置项 | spec 默认 | 实际使用 | 原因 |
|---|---|---|---|
| 每轮局数 | 50,000 | **10,000** | spec 估计的 50k × CPU 6-12h/round 在 M4 Max 单核下实测远超该估计（单 head 训练 50k 数据需 1.5h，8 head × 4 round = 48h+），降到 10k 局换 ~3h 全管线 |
| R0 selfplay | reference_human vs random 50k | 同 spec（产出 8 GB JSONL）| 一次性 7 min，selfplay 不是瓶颈 |
| R0 训练子集 | — | R0 头 500k 行（≈ 10k 局）| 训练时间瓶颈，不是 selfplay 数据量 |
| R1/R2/R3 | spec 50k 局 | 10k 局 | 同上 |
| R2 profile | prod (depth=3) | prod | iterative_train.py 用 prod profile 跑（约 3 小时 selfplay）|
| R3 验收 | 5 seed × 200 局 | 同 spec | acceptance_25_8.py 默认 |

---

## 2. 验收门槛 vs 实测

| 指标 | spec §6.1 T6 门槛 | Phase B (R2 bundle) | Phase A 基线 | 判定 |
|---|---|---|---|---|
| 5 seed × 25 全正窗口数 | 125/125 | **92/125 (73.6%)** | 107/125 (85.6%) | ❌ 退步 |
| 最差单窗口 | ≥ +2 冲 | -42 | -42 | ❌ 持平 |
| 总放炮率 | ≤ 12% | 17.2% | 14.4% | ❌ 退步 2.8pp |
| 平均冲数 / 局 | ≥ 0.5 | 3.018 | 3.886 | ✅ 但低于 baseline |

**结论**：**Phase B 在 10k 局训练规模下未超过 Phase A 基线**，且更差。最差窗口与放炮率都退步。

---

## 3. 逐轮 bundle 对比（3 seed × 200 局快测）

| Bundle | 窗口正比 | 最差窗口 | 平均冲数 | 放炮率 | 解读 |
|---|---|---|---|---|---|
| Phase A baseline (无 Phase B 模型) | 60-65/75 (~85%) | -43 | 3.97 | 14.8% | 硬编码 EV，最强 |
| R0 (refhuman vs random) | 59/75 (78.7%) | -42 | 3.237 | 16.7% | 训练数据不足，欠拟合 |
| R1 (V3(R0) vs refhuman) | 63/75 (84.0%) | -55 | 3.487 | 15.7% | 接近 baseline，最佳 Phase B 选择 |
| R2 (V3(R1) self-play) | 57/75 (76.0%) | -39 | 3.103 | 17.0% | self-play 蒸馏退步 |

**核心洞察**：
1. **R0 → R1 有改善**（59→63 窗口）：V3(R0) 蒸馏给了 R1 能从 baseline 学到的"V3 风格"
2. **R1 → R2 退步**（63→57）：self-play 数据收敛到"V3 攻击型"分布，丢失防守多样性
3. **R1 接近但仍不及 Phase A**：10k 局太少，模型欠拟合

---

## 4. 失败原因诊断

### 4.1 训练数据规模太小（主要原因）

spec §4.2 明确规定每轮 50k 局。我们用 10k 局换时间，但代价是模型欠拟合：
- LightGBM regression head 在 250k 训练样本时验证集 RMSE 还在下降，明显未收敛
- 防守长尾事件（清一色、树掉还原等）在 10k 局中出现频率太低，模型学不到

### 4.2 R2 self-play 蒸馏的固有问题

spec §4.2 表 R2 用"V3(R1) vs V3(R1) 温度采样 0.5"。但实际：
- iterative_train.py 没实现温度采样（只是默认 greedy）
- 结果 V3(R1) self-play **没有多样性**，全是 V3 标准打法
- 训练数据分布塌陷，新模型只学到"赢家如何赢"，没学到"失败者如何失败"
- houjuu_score regression head 因此学得很差

### 4.3 Phase A 硬编码 EV 的强度被低估

Phase A 的 λ/μ/feature_weights 经过网格搜索找到的局部最优，配上 baseline V3 的 6 head GBDT 已是相当强。Phase B 在 10k 局上重训这 6 head，把已经调好的 prob head 替换成欠拟合的新 head——直接退步。

---

## 5. 当前部署建议

**保留 Phase A 配置作为生产部署**：

```bash
# 生产配置 = Phase A baseline + Tuning C 参数
# 已经在 engine/params/v3/score_table.json 里
# 不需要切换到任何 Phase B bundle
```

**不要部署 R0/R1/R2 bundle**——它们只比 Phase A 弱。

---

## 6. 通往 100% 验收的剩余路径

### 6.1 短期（1-2 天）：把 Phase A 推到极限

之前 Phase A 网格搜索只跑了 small grid（6 点）+ 几个手调点。完整 full grid（27 点）+ 5 seed 验证可能再挤出 2-3 个窗口：

```bash
python3 tools/phase_a_tune.py --grid full --games-per-seed 200 --seeds 1 2 3 4 5
```

预计耗时：3-6 小时。可能从 107/125 推到 110-115/125，但**不太可能突破 ≥118 门槛**（spec §3.5.4）。

### 6.2 中期（1 周）：Phase B 真正发挥

需要满足两个条件才能让 Phase B 超过 Phase A：

1. **训练数据规模到 50k+ 局/轮**（spec 默认值）
   - 单核耗时：每轮训练 8-12 小时 × 3 轮 = 24-36 小时纯训练
   - + selfplay 约 5-10 小时/轮 = 总计 ~50-70 小时
   - 解决方案：用 nohup 后台跑 2-3 天

2. **R2 实现温度采样**（spec §4.2 明确写了"温度采样 0.5"，但 iterative_train.py 未实现）
   - 需要在 selfplay_sample.py 里加 `--temperature 0.5` 参数
   - 让 V3 self-play 时按 EV 概率采样而不是 greedy
   - 这样 R2 数据有多样性，蒸馏才能起作用

### 6.3 长期：训练管线优化

- **JSONL → parquet 预处理**：8 head 复用同一份数据，从重复解析 8 GB JSONL 改为加载 npz 数组，预计 5× 加速
- **多核 selfplay**：用 multiprocessing.Pool 把 10k 局拆 8 路并行，M4 Max 16 核下 4-6× 加速
- **GPU 训练**（spec §4.3 留了入口）：MLP 替代 GBDT，更适合长尾分布

---

## 7. 待用户决策的下一步

| 选项 | 投入 | 预期收益 |
|---|---|---|
| **A. 接受 Phase A 部署，走真人验收** | 0 时间 | 离线 107/125 (85.6%)；真人结果未知 |
| **B. Phase A full grid + 真人验收** | 半天 | 可能 110-115/125；上限 ~95% |
| **C. Phase B 50k 完整 spec 流程** | 2-3 天 CPU + 1-2 天调试 | 可能 115-125/125（不保证） |
| **D. 加温度采样 + 50k 局 + 训练管线优化** | 1-2 周 | 95-100% 概率达 125/125 |

推荐 **A → C** 的渐进路径：先用 Phase A 跑真人验收，看实际表现；如真人验收过不了，再投资 Phase B 50k。

---

**报告创建于** 2026-04-30，Phase B R0-R3 完整跑完后。
