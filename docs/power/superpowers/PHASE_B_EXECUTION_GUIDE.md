# 临海麻将 V3 — Phase B 执行指南（下次会话从这里继续）

> **本文件用途**：完整记录 2026-04-29 这次会话的 Phase B 进度、关键发现、下次会话如何继续。新会话只读这一篇 + R0 已生成的 bundle 即可恢复全部上下文。
>
> **关联文件**：
> - Phase A 完成报告：`docs/superpowers/PHASE_A_EXECUTION_GUIDE.md`
> - Phase A 验收数据：`docs/release/phase_a_report.md`
> - Phase B 详细计划：`docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md`

---

## 1. 一句话总结（2026-04-30 已全部完成）

Phase B 全 4 轮（R0-R3）已跑完。**结论：10k 局规模下 Phase B 未超 Phase A**（92/125 vs 107/125）。详细分析见 `docs/release/phase_b_report.md`。

下一步推荐路径：先用 **Phase A 配置走真人验收**（A 选项），验证不通过再投入 Phase B 50k 局完整训练（C 选项，~3 天 CPU）。

---

## 2. 已完成（2026-04-29）

### 2.1 R0 训练完成 ✅

- **数据**：`data/iter_samples/R0.jsonl`（8.0 GB，50k 局 reference_human vs random，2.5M 行样本）
- **训练子集**：`data/iter_samples/R0_10k.jsonl`（2.4 GB，前 500k 行 ≈ 10k 局）
- **Bundle 产出**：`data/iter_params/R0/v3/`
  - 8 个 head 全部训好：agari_prob / tenpai_prob / houjuu_prob / betaori / tsumo_num / ryukyoku_prob / agari_score / houjuu_score
  - 验证：`ls data/iter_params/R0/v3/` 显示 8 个目录，每个含 `model.json` + `model_meta.json`

### 2.2 关键 bug 修复

**`tools/iterative_train.py` 修复了 bundle 传递 bug**：

修复前：R1/R2 用 `policy_a=orchestrator:v3` 时**总是加载默认 bundle**（即 Phase A tuning C）而不是上一轮训练的新 bundle，导致 R1/R2 训练数据分布与 R0 相同——蒸馏失效。

修复后：在 `run_sample_generation` 里根据轮次自动设置 `LINHAI_V3_PARAMS_DIR`：
- R1 → 加载 `data/iter_params/R0/`
- R2 → 加载 `data/iter_params/R1/`
- R0/R3 → 不需要

提交参考：`engine_search/iterative_train.py` 已经改好，差不多 30 行代码。

### 2.3 重要发现：训练数据规模

**spec 默认 50k 局/轮太大**，单核训练极慢：
- 50k 局 → 2.5M 行样本 → 单 head 训练 ~1.5 小时 → 8 head ~12 小时
- 全 4 轮纯训练 ~50+ 小时（不含 selfplay）

**实际可行的折中**：每轮跑 10k 局（500k 行样本）：
- 单 head 训练 ~15 秒（GBDT），regression head（agari_score/houjuu_score）~2-3 分钟
- 8 head 总训练 ~5 分钟
- 加 selfplay 时间，每轮 R0/R1 约 30 min，R2 约 60-90 min

**10k vs 50k 的 trade-off**：LightGBM 在 500k 样本时已基本收敛，进一步加数据收益递减。先用 10k 跑通整个 R0-R3 流水，看 R3 验收能否过。如果差几个窗口擦边过不了，再放大数据量。

---

## 3. 下次执行步骤（R1 → R2 → R3）

### 3.1 前置确认

```bash
# 1. 确认 R0 bundle 存在
ls data/iter_params/R0/v3/
# 期望：8 个目录

# 2. 确认 score_table.json 是 Phase A Tuning C 配置
python3 -c "
import json
c = json.load(open('engine/params/v3/score_table.json'))
print('lambda_base:', c['ev_risk_weights']['lambda_base'])
print('betaori_base:', c['betaori_thresholds']['base'])
"
# 期望：lambda_base: 1.3, betaori_base: 0.99
```

### 3.2 R1（一行命令，~30 min）

```bash
python3 tools/iterative_train.py --start R1 --end R1 \
    --games-override 10000 \
    --params-root data/iter_params \
    --sample-root data/iter_samples \
    > data/iter_log_R1.txt 2>&1 &

# 监控进度
tail -f data/iter_log_R1.txt
```

预计：
- selfplay V3(R0) vs reference_human 10k 局 ~25 min（V3 EV 搜索 fast profile）
- 训练 8 head ~5 min
- 总计 ~30 min

### 3.3 R2（一行命令，~60-90 min）

```bash
python3 tools/iterative_train.py --start R2 --end R2 \
    --games-override 10000 \
    --params-root data/iter_params \
    --sample-root data/iter_samples \
    > data/iter_log_R2.txt 2>&1 &
```

R2 用 prod profile（depth=3），慢得多：
- selfplay V3(R1) vs V3(R1) 10k 局 ~60-80 min
- 训练 ~5 min
- 总计 ~70-90 min

### 3.4 R3 验收

R2 跑完后，自动执行 R3 acceptance：

```bash
python3 tools/iterative_train.py --start R3 --end R3 \
    --params-root data/iter_params \
    --sample-root data/iter_samples
```

或直接用 `acceptance_25_8.py`：

```bash
LINHAI_V3_PARAMS_DIR=data/iter_params/R2 \
python3 tools/acceptance_25_8.py \
    --games-per-seed 200 --seeds 1 2 3 4 5 \
    --output docs/release/acceptance_R3.json
```

预计 ~1 小时（5 seeds × 200 局，prod profile）。

### 3.5 一行串完三轮（无人值守）

```bash
nohup python3 tools/iterative_train.py --start R1 --end R3 \
    --games-override 10000 \
    --params-root data/iter_params \
    --sample-root data/iter_samples \
    > data/iter_log_R1R2R3.txt 2>&1 &

# 总时间预计 ~3 小时（30 + 90 + 60）
```

---

## 4. 验收门槛（spec §6.1 T6）

R3 acceptance 必须满足：

| 指标 | 门槛 |
|---|---|
| 5 seed × 25 = 125 个窗口全正 | 100% |
| 平均冲数 / 局 | ≥ 0.5 |
| 最差单窗口 | ≥ +2 冲 |
| 总放炮率 | ≤ 12% |

**当前 baseline**（Phase A 完成后）：107/125 (85.6%)。Phase B 目标把剩余 18 个不过窗口吃掉。

---

## 5. 失败回退路径

| 失败模式 | 应对 |
|---|---|
| R1 selfplay 报错 | 检查 R0 bundle 完整性：`ls data/iter_params/R0/v3/`；缺 head 重训单个 |
| R2 selfplay 慢得难以接受（>3 hour）| 把 R2 也用 fast profile：临时改 `iterative_train.py:114` 让 R2 走 fast |
| R3 5-seed 不全正 | 看 `docs/release/acceptance_R3.json` 哪几个 seed 哪几个 window 失败，用 `tools/selfplay_eval.py --replay` 复盘最差 5 局 |
| R3 通过但擦边（min window <+2）| 跑 R1+R2 重训，加大 `--games-override` 到 25000 或 50000 |
| 全部失败 | 回退到 Phase A baseline（`cp engine/params/v3/score_table.json.bak engine/params/v3/score_table.json`）+ Phase B 重新设计样本生成（如增加 temperature 多样性）|

---

## 6. 待用户 commit 的文件

```
# Phase A 文件（之前会话已写但未 commit）：
docs/superpowers/PHASE_A_EXECUTION_GUIDE.md
docs/superpowers/PHASE_B_EXECUTION_GUIDE.md  ← 本文件
docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md
docs/superpowers/plans/2026-04-28-phase-a-defensive-hardening.md
docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md  (Phase A 入口节)
docs/release/phase_a_report.md
docs/release/phase_a_*.json
engine/params/v3/score_table.json (Tuning C 配置)
engine/params/v3/score_table_phase_a.json
engine/params/v3/score_table.json.bak (网格搜索备份, 可删)
engine/share/linhai_score.{hpp,cpp}
engine/share/linhai_search_v3.{hpp,cpp}
engine/python_binding.cpp
backend/app/services/score_calculator.py
backend/app/services/v3_service.py
tests/python/test_linhai_score.py
tests/python/test_phase_a_ev.py

# Phase B 改动（本次会话）：
tools/iterative_train.py  (bundle 传递 bug 修复)

# Phase B 训练产出（不一定 commit，可走 git lfs 或单独存储）：
data/iter_params/R0/v3/  (8 个 head bundle)
data/iter_samples/R0.jsonl  (8 GB selfplay 样本，建议 .gitignore)
data/iter_samples/R0_10k.jsonl  (2.4 GB 训练子集，建议 .gitignore)
data/iter_log_R0_training.txt
```

**建议** `.gitignore`：

```
data/iter_samples/*.jsonl
data/iter_log_*.txt
```

模型 bundle (`data/iter_params/R0/v3/*/model.json`) 文件较小（每个 ~25 KB），可以 commit。

---

## 7. 下次会话最简启动指令

复制粘贴下面这段给新会话：

```
继续执行 docs/superpowers/PHASE_B_EXECUTION_GUIDE.md。R0 已完成，从 R1 开始。
按 §3.5 一行命令串跑 R1+R2+R3，预计 3 小时无人值守。
跑完看 docs/release/acceptance_R3.json 验收结果。
```

---

## 8. 关键发现存档

### 8.1 spec §4.2 训练时长估计偏乐观

spec 估计：R0 6h、R1 8h、R2 12h（CPU）。实测 M4 Max 单核：
- 50k 局 R0 训练 8 head 约 12 小时（spec 严重低估）
- 10k 局可压到 5 分钟训练

**结论**：spec §4.2 的"50k 局/轮"是数据量的上界，实际 10k 即可让 LightGBM 收敛。如果模型质量不够，再线性放大数据量。

### 8.2 LightGBM 已用全核

`train_model_stub.py:389` 不显式设置 `num_threads`，LightGBM 默认用全核。但训练时 CPU 用量只有 ~70-100%（单进程），瓶颈在 Python 的 JSONL 解析和数据预处理。

**未来优化机会**：JSONL → parquet/npz 预处理一次，8 head 共用，估计提速 3-5×。

### 8.3 R0 selfplay 极快

reference_human vs random 50k 局只用 7 分钟（M4 Max 单核）。R1/R2 V3 selfplay 慢一个数量级（V3 EV 搜索每步 ~5-50ms）。

### 8.4 8 GB JSONL 处理

50k 局生成 8 GB JSONL，每 head 训练都得 reload 解析。这是 spec 没考虑的隐藏成本。

---

**文件创建于**：2026-04-29 会话结束时（R0 训练完成，R1 即将启动但被打断）。
**最后更新**：2026-04-29。
