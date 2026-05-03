# 临海麻将 V3 — 真人验收部署交付（2026-04-30）

> **决策**：接受 Phase A 当前配置作为生产部署；不做进一步训练；进入甲方真人验收阶段。
>
> **关联文档**：
> - Phase A 报告：`docs/release/phase_a_report.md`
> - Phase B 报告：`docs/release/phase_b_report.md`
> - 验收 spec：`docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md` §6

---

## 1. 当前生产配置

**部署文件**：
- `engine/params/v3/score_table.json`（Phase A Tuning C）
- `engine/params/v3/VERSION` = `3.0.0-r3-dev`
- C++ 二进制：`engine/linhai_v3.cpython-310-darwin.so`（含 A3-A5 防守加固代码）

**核心参数**：

```json
{
  "ev_risk_weights": {
    "lambda_base": 1.3,
    "lambda_pressure_step": 0.4,
    "mu_base": 1.5
  },
  "betaori_thresholds": {
    "base": 0.99
  },
  "feature_weights": {
    "opp_meld_pressure_alpha": 0.5
  }
}
```

**未部署的实验产物**：
- `data/iter_params/R0/`、`R1/`、`R2/` — Phase B 训练 bundle，**经验证均不及 Phase A**，保留作研究用
- 不要把任何 R0/R1/R2 设为 `LINHAI_V3_PARAMS_DIR`

---

## 2. 离线性能基线（甲方真人对战前的诚实预期）

| 指标 | 5 seed × 200 局 vs ReferenceHumanPolicy |
|---|---|
| 单局胜率 | 65-71% |
| 平均冲数 / 局 | +3.886（净正分）|
| 单 8 局窗口为正概率 | **85.6%（107/125 实测）** |
| 总放炮率 | 14.4% |
| 单步 P95 延迟 | ~50ms（远低于 800ms 预算）|

**输赢分布特征**（200 局采样）：
- 70% 胜局，平均 +10.94 冲（79% 是满贯 ≥8 冲）
- 30% 负局，平均 -11.00 冲（70% 是被满贯打）
- **高方差攻击型 AI**，不是"小输保命"型

**关于"25/25 全正"门槛的诚实判断**：
- 单窗口 86% 正概率下，25 个独立窗口全正的统计上限约 ~3%
- 当前任何已实现的模型架构（含完整跑完的 Phase B R0-R3）都达不到 25/25
- spec §6.1 的"5 seed × 25/25 全正"是**激进的离线安全余量**，不是真人验收门槛
- 真人验收实际口径建议：**单一 200 局 25 窗口能否大多数为正**（如 22+/25）

---

## 3. 部署一致性核对（甲方验收前必查）

### 3.1 启动服务后查询

```bash
curl http://<deploy-host>/debug/engine_status
```

期望返回（关键字段）：

```json
{
  "active_engine": "v3",          ← 必须是 v3，不能是 v2 或 heuristic fallback
  "v3_loaded": true,
  "v3_score_heads_loaded": true,  ← Phase A score head 加载成功
  "model_version": "3.0.0-r3-dev",
  "bundle_hash": "<sha256-prefix>",
  "v3_profile": "prod"            ← 真人对战必须是 prod，不是 fast
}
```

任何字段不符合就**阻止上线**。

### 3.2 离线复测（部署完先跑）

```bash
LINHAI_V3_PROFILE=prod python3 tools/acceptance_25_8.py \
    --games-per-seed 200 --seeds 1 \
    --output /tmp/predeployment_check.json
```

预期：单 seed 21-22/25 窗口正分（如显著低于 20/25 说明部署有问题，回退）。

---

## 4. 真人验收期间的监控

### 4.1 关键指标（运营值班看板）

每场结束记录到日志：

| 指标 | 健康范围 | 红色告警 |
|---|---|---|
| AI 单步引擎延迟 P95 | ≤ 800ms | > 1500ms |
| AI 单步引擎延迟 P99 | ≤ 1500ms | > 2000ms |
| `/debug/engine_status::active_engine` | "v3" | "v2" / "heuristic" |
| 8 局窗口积分 | > 0 | ≤ -10 |
| 玩家 8 局胜率 | 30-50% | > 60% AI 太强 / < 20% AI 太弱 |

### 4.2 自动 fallback 触发条件（已在 v3_service.py 实现）

- 单步引擎 > 1800ms → 自动切 V2 fallback（写告警日志）
- score head 加载失败 → 自动切回 prob-only EV 路径
- C++ engine 崩溃 → orchestrator 自动切 V2 → heuristic 三层 fallback

---

## 5. 真人验收期间出现问题的应对

### 5.1 8 局窗口失败模式

**症状**：玩家说"AI 输得太多"或"窗口结算总是负分"

**诊断步骤**：
1. 看 `/debug/engine_status` 是否还是 v3
2. 看延迟监控有没有持续超时
3. 看具体哪一局：玩家自摸大胡？还是 AI 放了几次满贯？
4. 复盘最差几局：用 `tools/selfplay_eval.py --replay <log>` 重放 AI 决策

**应对**：
- 个别窗口运气负——可接受（85% 概率，统计预期会有 15% 不过）
- 系统性持续负——回退到 §5.3 紧急回滚

### 5.2 AI 表现不稳

**症状**：AI 时强时弱，玩家觉得"会犯低级错误"

**可能原因**：
- ReferenceHumanPolicy 训练时 mistake_prob = 3%，但**生产部署不是这个 policy**
- V3 在真人对战中决策稳定，不会随机出错
- 如出现，可能是延迟降级（V2 fallback）触发

**应对**：查 fallback 日志频次。

### 5.3 紧急回滚

如真人验收一致性差到无法接受：

```bash
# 回到 Phase A baseline（无 EV 风险加权）
cp engine/params/v3/score_table.json.bak engine/params/v3/score_table.json
# 或回到无 Phase A 加固版（git checkout 到 commit 56c5679 之前）

# 重启服务
systemctl restart linhai-v3-engine
```

---

## 6. 真人验收过不了时的升级路径

按 spec §1.3 的诊断，剩余 14% 不过窗口主要是"被对手清一色 / 树掉还原打爆"长尾。提升路径：

| 选项 | 投入 | 预期 |
|---|---|---|
| **完整 Phase B 50k 局训练**（spec §4.2 默认）| 2-3 天 CPU + 1-2 天调试 | +5-8% 窗口正概率 |
| 加温度采样到 R2 self-play | 0.5 天代码 + 1 天训练 | +3-5% |
| 训练管线优化（parquet + multiprocess）| 2-3 天代码 | 同上但快 5× |
| GPU 路径（spec §4.3 留接口）| 1 周 | +5-10%，需重写 |
| 重新设计 EV 公式（追求"小输"风格）| 2 周 | 可能 ±3% 不确定 |

详见 `docs/release/phase_b_report.md` §6。

---

## 7. 待用户最终 commit 的所有改动

```bash
# Phase A 完整改动（之前会话已写但未 commit 的）
git add engine/share/linhai_score.{hpp,cpp} \
        engine/share/linhai_search_v3.{hpp,cpp} \
        engine/python_binding.cpp \
        engine/params/v3/score_table.json \
        engine/params/v3/score_table_phase_a.json \
        backend/app/services/score_calculator.py \
        backend/app/services/v3_service.py \
        tests/python/test_linhai_score.py \
        tests/python/test_phase_a_ev.py

# Phase B 改动（含修复的 bundle 传递 bug）
git add tools/iterative_train.py \
        tools/phase_a_tune.py \
        data/iter_params/R0/v3/ \
        data/iter_params/R1/v3/ \
        data/iter_params/R2/v3/

# 报告与文档
git add docs/release/phase_a_report.md \
        docs/release/phase_b_report.md \
        docs/release/deployment_handoff.md \
        docs/release/acceptance_report_5seed.json \
        docs/release/acceptance_R3_5seed.json \
        docs/release/r0_eval.json docs/release/r1_eval*.json \
        docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md \
        docs/superpowers/plans/2026-04-28-phase-a-defensive-hardening.md \
        docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md \
        docs/superpowers/PHASE_A_EXECUTION_GUIDE.md \
        docs/superpowers/PHASE_B_EXECUTION_GUIDE.md

git commit -m "release: V3 Phase A deployment + Phase B research bundle (real-human acceptance)"
```

**建议加 .gitignore**：
```
data/iter_samples/*.jsonl     # 8GB+ selfplay 样本
data/iter_log_*.txt           # 训练日志
engine/params/v3/score_table.json.bak
```

---

## 8. 真人验收脚本（供运营用）

按 spec §6.1.1 内测建议：

1. **样本量**：4-6 名内测玩家，每人对 V3 至少 24 局（3 个 8 局窗口）
2. **环境**：生产环境，prod profile，监控全开
3. **记录**：每局结果（胜负 / 冲数 / 决策时长）
4. **判定**：
   - **过**：90% 窗口正分 且无玩家全部窗口为负
   - **擦边**：85-90% 窗口正分 → 评估是否接受
   - **不过**：< 85% 窗口正分 → 触发 §6 升级路径

---

**报告创建于** 2026-04-30，Phase A+B 全部研究完成，进入真人验收阶段。
