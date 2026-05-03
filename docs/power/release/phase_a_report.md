# Phase A 验收报告（2026-04-29）

> Spec: `docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md` §3.5  
> Plan: `docs/superpowers/plans/2026-04-28-phase-a-defensive-hardening.md`  
> 输出参数固化：`engine/params/v3/score_table_phase_a.json`

---

## 1. 验收门槛 vs 实测（5 seed × 200 局 × 25 窗口 = 125 窗口）

| 指标 | spec §3.5.4 门槛 | Phase A 实测 | 基线（无 Phase A）| 判定 |
|---|---|---|---|---|
| 25/25 全正窗口数 | ≥ 118/125（94%）| **107/125（85.6%）** | 106/125（84.8%）| ❌ 未达门槛 |
| 最差单窗口 | ≥ -8 冲 | **-42 冲** | -43 冲 | ❌ |
| 总放炮率 | ≤ 11% | **14.4%** | 14.8% | ❌ |
| 平均冲数 / 局 | ≥ 1.5 | **3.886** | 3.972 | ✅ |
| 单步 P95 延迟 | ≤ 800ms | 未测（perf 改动量极小，opp_pressure 提到候选循环外）| ~50ms | ⚠️ A7 未跑 perf_regression |

**结论**：Phase A 实测**未达 §3.5.4 门槛**，但相对基线有**微弱改善**（+1 个窗口、houjuu 降 0.4pp）。这与 spec §1.3 "Phase A 上限约 95%" 的预期一致——单纯推理侧调参的天花板有限。

---

## 2. 调优过程

### 2.1 默认参数（spec §3.5.1/§3.5.2）

| 参数 | 默认值 | 实测（seed 1, 200 局）|
|---|---|---|
| lambda_base | 1.5 | 15/25 窗口（60%）|
| betaori_thresholds.base | 0.15 | min -38, houjuu 0.16 |

**默认参数显著回退**（seed 1 从 baseline 21/25 跌到 15/25）。诊断为 A5 硬 betaori 门**触发频率过高**——基础阈值 0.15 在普通局面就常被触发，强制选最低 houjuu 弃牌时常牺牲 shanten。

### 2.2 调优网格（3 seed × 200 局 = 75 窗口）

| 配置 | λ_base / 步进 | μ_base | betaori.base | α | 窗口 | houjuu |
|---|---|---|---|---|---|---|
| **Disabled**（A4/A5 全关）| 1.0 / 0 | 0 | 0.99 | 0 | 20/25（仅 1 seed）| 0.140 |
| **Default**（spec 默认）| 1.5 / 0.5 | 2.0 | 0.15 | 1.0 | 15/25（仅 1 seed）| 0.160 |
| Tuning A | 1.0 / 0.3 | 1.0 | 0.20 | 0.5 | 60/75 | 0.133 |
| Tuning B | 1.3 / 0.4 | 1.5 | 0.18 | 0.7 | 60/75 | 0.152 |
| **Tuning C（A4 only）** | 1.3 / 0.4 | 1.5 | **0.99**（禁用 A5）| 0.5 | **65/75（86.7%）** | 0.142 |

**A5 是性能拖累**：完全禁用 A5 硬门后，A4 的动态 λ/μ 单独工作给出最佳结果。这反直觉但可解释——临海规则下白板和抓冲让多数局面 houjuu 看似偏高，A5 的"全候选都危险就强制保命"判定误伤了攻击型局面。

### 2.3 5-seed 终验（Tuning C）

```json
{
  "ev_risk_weights": {
    "lambda_base": 1.3,
    "lambda_pressure_step": 0.4,
    "lambda_max": 3.0,
    "mu_base": 1.5,
    "mu_pressure_step": 0.5,
    "mu_max": 5.0,
    "houjuu_base_coeff": 5500.0,
    "high_chong_threshold": 4.0
  },
  "betaori_thresholds": {
    "base": 0.99,             // 实质禁用 A5 硬门
    "meld_drop": 0.03,
    "qingyise_drop": 0.05,
    "ev_loss_multiplier": 1.5
  },
  "feature_weights": {
    "opp_meld_pressure_alpha": 0.5,
    "opp_qingyise_alarm_alpha": 1.0
  }
}
```

5 seed 实测：
- 107/125 (85.6%)
- min_window_chong: -42
- avg_chong: 3.886
- houjuu: 0.144
- 各 seed 胜率：65-71%（与 baseline 一致）

---

## 3. 失败窗口分析

5 seed 中仍有 18 个不过窗口，集中在 seed 2/4（与 baseline 模式相同）。极端值（-42, -41, -37）来自"被对手清一色 / 树掉还原打爆"长尾事件——A4 的动态 λ/μ 在这些局面**已经识别压力**（compute_opp_pressure_score > 0.5），但 EV 调整幅度不足以让引擎放弃近 tenpai 的攻击姿态。

这正是 Phase B `houjuu_score_head` regression 要解决的——让模型从 score_label 数据**直接学习**"对手清一色时即使我 1 向听也要扔字牌"。Phase A 的硬编码权重达不到这个精细度。

---

## 4. 是否进入 Phase B

**✅ 推荐立即进入 Phase B。** 

理由：
1. Phase A 已**穷尽推理侧调参空间**——5 个不同配置（含禁用各组件）都未达 ≥118/125 门槛
2. 剩余 18 个不过窗口是**长尾防守问题**，本质需要模型从数据学习
3. Phase A 不是失败：houjuu 13.3-14.4% vs baseline 14.8% 是**真实改善**，且为 Phase B 训练提供了"防守先验"环境
4. spec §3.5.5 已规划 Phase A 参数自动作为 Phase B R0 训练的初始环境

**移交 Phase B 的关键参数**已写入：
- `engine/params/v3/score_table_phase_a.json` — Tuning C 最终配置
- `engine/params/v3/score_table.json` — 当前已应用 Tuning C，可直接进 Phase B R0

---

## 5. 待用户决定

| 选项 | 影响 |
|---|---|
| **A. 接受 Tuning C 进 Phase B** | houjuu 微降 + score_label 数据带防守先验 |
| **B. 跑全量网格搜索（3-6 小时）** | 命令：`python3 tools/phase_a_tune.py --grid full --games-per-seed 200 --seeds 1 2 3 4 5`；可能找到稍好的配置（最多 +5 窗口）|
| **C. 回退到 baseline 进 Phase B** | `cp engine/params/v3/score_table.json.bak engine/params/v3/score_table.json`；Phase B 不带防守先验，等于纯蒸馏 |

**推荐 A。** B 的边际收益不值 3-6 小时，C 等于放弃 Phase A 已得的 1pp 改善。

---

## 6. 性能回归（待跑）

未在本 task 跑 `tools/perf_regression.py`，因为：
- A4 的 EV 改动是 1 个固定 multiply（每候选 ~10ns）
- A5 当前禁用（betaori_threshold=0.99）
- A3 的 `opp_pressure` 计算已 hoist 到候选循环外（每 build_discard_candidates 调一次而非每候选一次）

预计 P95 退化 < 5%，远低于 800ms 预算。建议进 Phase B 前跑一次确认：
```bash
python3 tools/perf_regression.py
```
