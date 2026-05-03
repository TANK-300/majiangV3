# 临海麻将 V3 — Phase A 执行指南（下次会话直接读这一篇）

> **本文件用途**：完整记录 2026-04-28 这次会话的诊断、决策、计划、当前进度与下次会话如何继续。新会话只读这一篇 + 引用的 spec / plan 即可恢复全部上下文。

---

## 1. 一句话总结

把临海麻将 V3 引擎的 **5-seed × 200 局验收**从当前 **106/125 (84.8%)** 推到 **125/125** 全正、放炮率 ≤12%、最差窗口 ≥+2 冲。**A+B 组合迭代**：先做 Phase A 推理侧防守加固（3-5 天，不重训），再做 Phase B 迭代蒸馏（4 周）。

---

## 2. 为什么做（诊断）

`docs/release/acceptance_report_5seed.json` 实测（V3 现状 vs ReferenceHumanPolicy，5 seed × 200 局）：

| 指标 | 现状 | 验收门槛 | 判定 |
|---|---|---|---|
| 25/25 全正窗口数 | 106/125（84.8%）| 125/125 | ❌ |
| 最差单窗口 | -43 冲 | ≥ +2 | ❌ |
| 总放炮率 | 14.8% | ≤ 12% | ❌ |
| 平均冲数 / 局 | 3.972 | ≥ 0.5 | ✅ |
| 单 seed 胜率 | 65-71% | — | ✅ |

**核心瓶颈**：胜率与赢面充足，问题是**爆炸性输局**（min_window -43 集中在 seed 2/4，对应"被对手清一色/树掉还原打爆"）。**不是模型欠拟合，是 EV 公式风险定价偏弱与防守阈值不灵敏**。单纯重训不一定打中防守，所以选 A+B 组合。

---

## 3. 怎么做（A+B 组合）

### Phase A — 推理侧防守加固（3-5 天，不重训）

不动模型，只调可控参数。三个核心改造：

1. **风险厌恶 EV 重加权**：`total_ev = E[gain] - λ·E[loss] - μ·max(0, P(opp ≥4 冲) · expected_loss)`，λ/μ 根据 `opp_pressure_score` 动态变大
2. **betaori 阈值动态化**：基础 `houjuu_prob > 0.20` 改成基础 0.15 + 对手副露 -0.03 + 疑似清一色 -0.05
3. **副露压力感知特征**：C++ 推理侧实现 `opp_meld_pressure` 与 `safety_genbutsu_t_X` 两个高价值特征作为 EV 修正

### Phase B — 迭代自对弈蒸馏（4 周，CPU ~30h 训练）

按现有计划 `docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md` 跑 R0-R3：

| 轮 | 对手对 | 样本量 | 训练动作 |
|---|---|---|---|
| R0 | ReferenceHumanPolicy vs random | 50k 局 | 6 head GBDT，产出 V3(R0) |
| R1 | V3(R0) vs ReferenceHumanPolicy | 50k 局 | 6 head 重训，重点校准 houjuu/betaori |
| R2 | V3(R1) vs V3(R1)（温度 0.5）| 50k 局 | depth=3，再训 |
| R3 | V3(R2) vs ReferenceHumanPolicy | — | 验收回归 |

### A+B 后预期指标

| 指标 | 现状 | A 后 | **A+B 后** | 验收门槛 |
|---|---|---|---|---|
| 25/25 全正 | 106/125 | 118/125 (94%) | **123-125/125** | 125/125 |
| 最差窗口 | -43 | -8~-12 | **0~+5** | ≥ +2 |
| 放炮率 | 14.8% | 10-11% | **8-10%** | ≤ 12% |
| 平均冲数 | 3.972 | 2.5-3.0 | **3.5-4.5** | ≥ 0.5 |

---

## 4. 已经做了什么（截至 2026-04-28）

### 4.1 Spec 与计划文档（已写完，待提交）

- ✅ **Spec 已更新**：`docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md`（git status: `MM`）
  - §1.3 加入 5-seed 实测基线（106/125 等）
  - §3.5 加入 Phase A 完整章节（5 个子节）
  - §9 实施顺序按 Phase A → Phase B 重排
  - 顶部策略代号改为「A+B 组合迭代」

- ✅ **Phase A 实施计划**：`docs/superpowers/plans/2026-04-28-phase-a-defensive-hardening.md`（git status: `??` 未跟踪）
  - 8 个 task，每个 task 有 TDD 风格的 step-by-step 步骤
  - 每个 task 有"提交边界"，给用户的 git 命令而非 AI 自动 commit

- ✅ **Phase B 实施计划**：`docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md`（已存在，需在 Task A8 加 Phase A 入口节）

### 4.2 代码现状（Phase B groundwork 已大部分实现，未提交）

**已修改文件**（git status `M`，对应 spec §1-§4 的基础工作）：
- `backend/app/services/v3_service.py` — orchestrator
- `engine/share/linhai_search_v3.cpp/hpp` — V3 引擎，含 score-aware EV
- `engine/setup.py`
- `tools/extract_canonical_states.py` — 训练侧特征提取
- `tools/selfplay_eval.py / selfplay_sample.py / train_model_stub.py`
- `docs/EVAL.md / FEATURES.md / PARAMS.md`

**已新增文件**（git status `??`，Phase B groundwork）：
- `backend/app/services/reference_human.py` — 离线代理对手
- `backend/app/services/score_calculator.py` — Python 番数计算
- `engine/share/linhai_score.cpp/hpp` — 番数引擎
- `engine/params/v3/score_table.json` — 番数参数表（**Phase A 要扩展这个**）
- `engine/params/v3/VERSION`
- `tools/acceptance_25_8.py / calibrate_reference.py / iterative_train.py / perf_regression.py`
- `tests/cpp/test_linhai_score.cpp`
- `tests/python/test_acceptance_25_8.py / test_engine_status_version.py / test_feature_parity.py / test_linhai_score.py / test_perf_regression.py / test_reference_human.py / test_score_label_pipeline.py`
- `docs/DEPLOYMENT.md`
- `docs/release/`（含 `acceptance_report_5seed.json` 基线、`reference_calibration.json`）

### 4.3 Phase A 任务进度（**2026-04-29 已全部完成**）

| Task | 状态 | 关键产出 |
|---|---|---|
| A1 | ✅ | ScoreTable 扩展 + 防漂移 pin 测试 + λ_base 修正到 1.5 |
| A2 | ✅ | score_table.json 写入 Phase A 默认参数 |
| A3 | ✅ | `compute_opp_pressure_score` C++ 实现 + pybind11 暴露 |
| A4 | ✅ | 动态 λ/μ EV 风险加权（4/4 集成测试过）|
| A5 | ✅ | 硬 betaori 门实现（实测发现误伤攻击型局面，最终配置中禁用）|
| A6 | ✅ | `tools/phase_a_tune.py` 网格搜索工具 |
| A7 | ✅ | 5-seed × 200 局验收：**107/125 (85.6%)**、houjuu 14.4%——未达 ≥118/125 门槛但 baseline+1pp |
| A8 | ✅ | Phase B 计划已加入 Phase A 入口节，参数固化到 score_table.json |

**Phase A 最终结论**：推理侧调参空间已穷尽，剩余 18 个不过窗口需要 Phase B 蒸馏才能解决。详见 `docs/release/phase_a_report.md`。

---

## 5. Phase B 已部分启动 — 接力指南

**2026-04-29 进度**：Phase B 代码 99% 已就绪，R0 训练已完成。下次会话直接从 R1 开始。

完整 Phase B 指南：`docs/superpowers/PHASE_B_EXECUTION_GUIDE.md`

**最简启动**：
```
继续执行 docs/superpowers/PHASE_B_EXECUTION_GUIDE.md。R0 已完成，从 R1 开始。
按 §3.5 一行命令串跑 R1+R2+R3，预计 3 小时无人值守。
```

---

## 5. 下次会话如何继续（执行 playbook）

### 5.1 第一步：恢复上下文

新会话开始时，按顺序读：

1. **本文件** — 全局摘要与状态
2. `docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md` §3.5 — Phase A 完整设计
3. `docs/superpowers/plans/2026-04-28-phase-a-defensive-hardening.md` — 8 个 task 详细步骤
4. （可选）`docs/release/acceptance_report_5seed.json` — 当前基线数据

### 5.2 执行模式

用户已选定 **Subagent-Driven Development**（superpowers:subagent-driven-development）：每个 task 派 fresh subagent 实施，task 间两阶段 review（spec compliance → code quality）。

### 5.3 严格遵守的用户偏好

> **🚨 用户 memory 规则**：`不要执行 git commit / git add — 用户自己负责 git 提交，AI 只编辑文件`
>
> 在每个 task 末尾**只输出修改文件清单 + 给用户的 git 命令**，**不直接调用 git**。
> Implementer subagent 也必须遵守这一点（在 dispatch prompt 里明确告知）。

### 5.4 启动序列

1. **检查 task 状态**：用 `TaskList` 看 #3-#10（Task A1-A8）是否存在；不存在则按本文件 §6 重建
2. **从 Task A1 开始**：用 implementer prompt 模板派 subagent，prompt 里**包含 plan A1 的完整文本**（不让 subagent 读 plan 文件，按 skill 要求）
3. **Task A1 完成 → spec reviewer → code quality reviewer → 用户 commit → Task A2**
4. **顺序执行到 Task A7**（A7 含 ~3 小时全量网格搜索，建议后台跑）
5. **Task A7 通过门槛（≥118/125）后**，Task A8 把 Phase A 入口节加到 Phase B 计划，正式进入 Phase B
6. **Phase B**：按 `docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md` 的 Phase 6-7 继续

### 5.5 失败回退路径

- **Task A4/A5 实现后单测过不了**：检查是否 C++ 字段（`opponent_honor_triplets` 等）需要先在 `types.hpp` 补
- **Task A7 全量验收不达 ≥118/125**：回 A6 扩大网格（加 `mu_base` 维度），或调 `houjuu_base_coeff` 从 5500 升到 6500
- **Phase A 反复调不上去**：升级 dispatch model 到 opus，或考虑直接进 Phase B（更长周期）

---

## 6. Phase A 任务清单（8 个 task，task ID 备查）

| Task ID | Subject | 状态 | 预计耗时 | Plan 锚点 |
|---|---|---|---|---|
| #3 | Task A1: 扩展 ScoreTable + JSON 解析（C++ + Python 镜像）| pending | 0.5 d | Plan §Task A1 |
| #7 | Task A2: score_table.json 写入 Phase A 默认参数 | pending | 0.25 d | Plan §Task A2 |
| #9 | Task A3: 实现 compute_opp_pressure_score（C++）| pending | 0.5 d | Plan §Task A3 |
| #4 | Task A4: EV 公式风险厌恶重加权（动态 λ/μ）| pending | 0.75 d | Plan §Task A4 |
| #10 | Task A5: 硬 betaori 门（危险局面强制保命）| pending | 0.5 d | Plan §Task A5 |
| #6 | Task A6: phase_a_tune.py 网格搜索 | pending | 0.75 d | Plan §Task A6 |
| #5 | Task A7: Phase A 全量验收实跑 + 报告 | pending | 0.5 d | Plan §Task A7 |
| #8 | Task A8: Phase B 衔接（参数固化 + 移交记录）| pending | 0.25 d | Plan §Task A8 |

> **如果 task 列表丢失**，按上表用 `TaskCreate` 重建；description 字段引用对应的 Plan §Task An 即可。

---

## 7. 关键决策记录（避免下次再讨论）

| 决策项 | 决议 | 来源 |
|---|---|---|
| 优化策略 | A+B 组合（不是单 A 或单 B）| 用户确认 2026-04-28 会话 |
| Phase A 范围 | 只实现 2 个高价值特征（`opp_meld_pressure` + `safety_genbutsu_t_X`），其余 3 个留给 Phase B | spec §3.5.3 |
| Phase A 验证门槛 | ≥118/125、最差 ≥-8、放炮 ≤11%、平均冲 ≥1.5 | spec §3.5.4 |
| Phase A 默认参数 | λ_base=1.0, μ_base=2.0, betaori_base=0.15 | spec §3.5.1/§3.5.2 |
| 执行模式 | Subagent-Driven Development | 用户选项 1 |
| Git 提交策略 | AI 只编辑文件，用户手动 commit | 用户 memory 规则 |
| Phase B 周期 | 4 周（R0=6h、R1=8h、R2=12h、R3=1h），CPU 训练 | spec §4.2 |
| Phase A 时间 | 3-5 天 | spec §3.5 标题 |
| Phase A+B 总预期 | 25/25 全正 98-100% 概率，最后 1-2 个边缘窗口可能需要微调 | 会话讨论 |

---

## 8. 关键文件路径速查

```
docs/superpowers/
├── PHASE_A_EXECUTION_GUIDE.md           ← 本文件
├── specs/
│   └── 2026-04-28-linhai-v3-acceptance-design.md   ← spec（A+B 完整设计，§3.5 是 Phase A）
└── plans/
    ├── 2026-04-28-phase-a-defensive-hardening.md   ← Phase A 实施计划（8 个 task）
    └── 2026-04-28-linhai-v3-acceptance-plan.md     ← Phase B 实施计划（已存在，Task A8 加入口节）

engine/
├── params/v3/
│   └── score_table.json                 ← Phase A 主要修改对象（A1-A2）
└── share/
    ├── linhai_score.hpp / cpp           ← ScoreTable 扩展（A1）
    └── linhai_search_v3.hpp / cpp       ← EV 公式与 opp_pressure_score（A3-A5）

backend/app/services/
└── score_calculator.py                  ← Python 镜像默认值（A1）

tools/
├── acceptance_25_8.py                   ← 验收回归（A7 用）
├── perf_regression.py                   ← 性能回归（A4 用）
└── phase_a_tune.py                      ← Phase A 网格搜索（A6 新增）

tests/
├── cpp/test_phase_a_risk_ev.cpp         ← C++ 单测（A3 新增）
└── python/
    ├── test_linhai_score.py             ← 加 Phase A 字段测试（A1）
    └── test_phase_a_ev.py               ← Phase A EV 集成测试（A4-A5 新增）

docs/release/
├── acceptance_report_5seed.json         ← 当前基线（106/125）
├── phase_a_acceptance.json              ← Phase A 验收输出（A7 新增）
├── phase_a_report.md                    ← Phase A 验收报告（A7 新增）
└── phase_a_tune_full.log                ← 网格搜索日志（A7 新增）
```

---

## 9. 下次会话最简启动指令模板

复制粘贴下面这段给新会话（替换 `<TASK_ID>` 为下一个待执行 task）：

```
我要继续执行 docs/superpowers/PHASE_A_EXECUTION_GUIDE.md 里的 Phase A。
请先读这份指南了解上下文，然后用 Subagent-Driven Development 模式从 Task A<n> 开始派 fresh subagent 实施。
注意我的 memory 规则：AI 只编辑文件，所有 git add/commit 由我手动执行。
派 implementer 时把 docs/superpowers/plans/2026-04-28-phase-a-defensive-hardening.md 里 Task A<n> 的完整文本嵌入 prompt（不要让 subagent 读 plan 文件）。
```

---

## 10. 风险与回滚

| 风险 | 缓解 |
|---|---|
| Phase A 调到上限（~95%）仍未到 100% | 进入 Phase B 即可，本来就需要 |
| Phase A 推理改造让 P95 退化 | `tools/perf_regression.py` 在每个 task 后跑，超 800ms 阻塞 |
| `score_table.json` 改坏 | 每次 A6 网格搜索前自动备份到 `score_table.json.bak`；Phase A 失败时直接 `cp .bak json` 回滚 |
| C++ 编译失败 | Task A1/A3/A4/A5 后必须跑 `cd engine && python3 setup.py build_ext --inplace` |
| Subagent 不遵守 "AI 不 commit" 规则 | dispatch prompt 必须显式包含此规则；review subagent 也要 verify |
| 上下文窗口耗尽 | 本文件 + spec + plan 三份核心文档约 30k tokens，单 task 派遣 prompt < 5k tokens，足够 |

---

**文件创建于**：2026-04-28，与 spec 和 plan 同 commit。
**最后更新**：2026-04-28 会话结束时（Task A1 即将派发但被打断）。
