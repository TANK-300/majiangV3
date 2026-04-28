# 临海麻将 V3 — 算法验收方案设计（2026-04-28）

> 这是甲方算法验收（"200 局 25 次 8-局结算积分均为正 / 拿牌到返回 ≤2s P95 / 临海规则与番数完整"）的实施前 spec。  
> 实现策略代号：**B —— 番数感知 EV + 迭代自对弈蒸馏**。  
> 写入文档前已与用户对齐（双人临海主验收、1冲=1分、CPU 训练为主、4 周节奏）。

---

## 1. 背景与差距盘点

### 1.1 现状（2026-04-28）

`majiangV3` 仓库已经具备：

- C++ 主引擎 `engine/share/linhai_search_v3.cpp`：Akochan 风格 EV 搜索，Chance/Decision Node，深度 0-3，beam 剪枝，shanten cache。
- 6 个估值 head（agari / tenpai / houjuu / betaori / tsumo_num / ryukyoku），同时支持 sklearn linear、LightGBM GBDT、PyTorch MLP→ONNX 三种训练入口。
- 临海特色规则：白板万能、抓冲、三键承包、过胡不胡、副露与响应（吃碰杠胡 / pass）。
- FastAPI 接口：`/ai/recommend`、`/ai/recommend-response`、`/ai/recommend-all`、`/debug/engine_status`。
- V3 → V2 → heuristic 三层 fallback；自对弈胜率回归脚本 `tools/selfplay_eval.py`。
- 实测：500 局 vs heuristic，胜率 68.2%，放炮率 13.6%，单步 P95 ~50ms。

### 1.2 验收差距

按甲方需求图（`docs/Algorithm/c65d2d63dacbd74d47c9aa7a336d8752_compress.jpg`）：

| 需求 | 现状 | 差距 |
|---|---|---|
| 每 8 局结算一次积分，每次为正 | 自对弈用简化胜率，无番数→冲→分结算器 | 缺：番数引擎 + 8-局滚动结算 |
| 200 局 25 次结算全为正 | vs heuristic 68% 单局胜率，未在 8-局窗口验证；vs 真人未知 | 缺：硬性回归脚本 + 训练目标对齐"番数加权 EV" |
| 引擎计算 P95 ≤ 2s | prod 单步 ~50ms，已达标 | 验证最坏牌型（4 白板 / 复杂副露 / 多抢杠 / 抓冲全开）即可 |
| 临海番数（硬碰硬 / 翻倍 / 树掉还原 / 混一色清一色） | `calc_yaku.cpp` + `linhai_bonus.cpp` 部分实现，未必逐条覆盖图条款 | 需逐条核对并完整接进结算器 |
| 抓冲 / 三键承包 / 翻屁股 / 四人抓红头 | 字段已存在，红头/翻屁股需核对 | 需逐条核对，四人玩法仅做发牌机制 |
| 数据需求指标（图中红字标题） | 无真实牌谱 | 全部样本由项目内自对弈产生 |

---

## 2. 验收口径与离线代理对手

### 2.1 双轨验收

- **主验收**：甲方真人 vs AI 200 局 25 次 8-局结算正分。**不可在 CI 复现**。
- **离线代理**：`tools/acceptance_25_8.py` 跑 AI vs `ReferenceHumanPolicy` 200 局 25 次 8-局结算，CI 必须 25/25 全正。

### 2.2 ReferenceHumanPolicy 设计

新增文件 `backend/app/services/reference_human.py`，作为可复现的"近似真人"代理对手：

| 模块 | 行为 | 目的 |
|---|---|---|
| 基础动作 | 现有 V2 baseline (`v2_fallback.py`) | 真人级 EV 估算 |
| 防守层 | houjuu_prob > 0.25 时有 30% 概率 betaori | 模拟真人保守 |
| 番数倾向 | 检测到清一色 / 混一色 雏形（≥6 张同色）时偏向追大番 | 模拟真人贪番 |
| 错牌噪声 | 5% 概率从 top-3 候选随机选 | 模拟真人失误 |
| 抓冲倾向 | 抓冲红头存在时优先满足门风牌 | 模拟真人偏好 |

**作用**：让 AI 必须真懂麻将而不是钻 heuristic 的漏洞；离线代理输了，对真人也大概率会输。

### 2.3 真人验收兜底

- 上线时强制 `LINHAI_V3_PROFILE=prod`。
- `/debug/engine_status` 增加 `model_version`、`bundle_hash` 字段，方便甲方核对部署一致性。
- 服务监控 P95 引擎延迟，>1800ms 自动触发 V2 fallback 并写入告警日志。

---

## 3. 番数引擎与 EV 整合

### 3.1 新增模块 `engine/share/linhai_score`

```cpp
// linhai_score.hpp
struct ChongBreakdown {
    int base_chong;          // 1 / 2 / 4 / 8
    int multiplier_2x;       // 硬碰硬 / 特殊牌翻倍
    int multiplier_4x;       // 树掉还原（白板暗刻当 3 财神）
    int extra_chong;         // 抓冲红头（不计入 8 封顶）
    int qiang_gang;          // 抢杠胡 +2 冲
    int contract_amount;     // 三键承包平摊本数
    int final_chong;         // min(base * mult, 8) + extra + qiang_gang
    std::string explain;     // 调试可读
};

ChongBreakdown calc_chong(const Tehai& hand,
                          const std::vector<Meld>& melds,
                          int white_count_in_hand,
                          int white_count_in_melds,
                          bool is_tsumo,
                          bool is_qiang_gang,
                          const ContractState& contract,
                          const RedHeadCatch& redhead);
```

### 3.2 番数表（图条款逐条映射）

| 牌型 / 修饰 | 冲数效果 | 备注 / 来源 |
|---|---|---|
| 普通胡（自摸 / 放炮） | base = 1 | 图基础 |
| 硬碰硬（无白板） | ×2 | 图条款 1 |
| 特殊牌翻倍（中/发/白/自风触发碰杠或暗刻） | ×2（与硬碰硬叠加） | 图条款 2 |
| 树掉还原（白板暗刻当 3 财神） | base 1 → 4，并按硬碰硬叠加 | 图条款 3，"按硬碰硬、白板暗刻 3 财神还原 ×4 冲" |
| 混一色（有树掉） | base = 2 | 图条款 4 |
| 混一色（硬碰硬） | base = 4 | 图条款 4 |
| 清一色 | base = 8（封顶） | 图条款 4 |
| 字一色（图未明示） | base = 8（按清一色等价，T1 默认） | spec 决议，参数化 |
| 抢杠胡 | +2 冲（T2 默认） | spec 决议，参数化 |
| 抓冲红头（每张对应门风） | +1 冲/张，不计入 8 封顶 | 图特殊说明 1 |
| 三键承包（自摸） | 双家承包：两家平摊冲数（T3 默认） | 图特殊说明 2 |
| 翻屁股红头 | 一万 +1 冲，九万 +9 冲，中/发/白/东/南/西/北 +5 冲 | 图特殊说明 3 |

### 3.3 EV 标签整合

- 当前 `task_labels.can_win_label` (0/1) → 新增 `task_labels.score_label`（实际冲数，正负零）。
- `score_label` 的赋值规则：把本局最终结算冲数（自家视角，胡牌 +N，放炮 -N，旁观 0，流局 0）回填到该玩家在本局的每一个决策样本上；不做衰减或 credit 分配，由模型从聚合数据中学习。
- 训练 head 调整：
  - `agari_prob` → `agari_score_head`（regression，预测自家本局期望冲数）
  - `houjuu_prob` → `houjuu_score_head`（regression，预测自家本局期望失分）
  - 其余四个 head 保留概率口径，用于辅助特征。
- C++ 推理：`total_ev = E[my_score] - E[opp_score]` 作为弃牌排序主键。

### 3.4 默认参数表（spec 决议项）

| 项 | 默认值 | 改动方式 |
|---|---|---|
| T1 字一色冲数 | 8（与清一色同） | `engine/params/v3/score_table.json::ziyise_base` |
| T2 抢杠胡冲数 | +2 | `qianggang_bonus` |
| T3 三键承包公式 | 两家平摊赢家全部冲数 | `contract_split_ratio` |
| T4 翻屁股每人发牌数 | 2 张 | `redhead_per_player` |
| T5 流局张数 | 6 张（与 T4 联动） | `liuju_remaining` |
| 1 冲 = 多少分 | 1 分 | `chong_to_score` |

所有项写入 `engine/params/v3/score_table.json`，**修改后不需要重新编译 C++**。

---

## 4. 训练管线 — 迭代自对弈蒸馏

### 4.1 新增 / 改造脚本

| 文件 | 动作 | 说明 |
|---|---|---|
| `tools/selfplay_eval.py` | 改造 | 接进 `linhai_score`，每局产出真实冲数 |
| `tools/selfplay_sample.py` | 改造 | 样本带 `score_label`（regression target） |
| `tools/train_model_stub.py` | 改造 | `--task agari_score / houjuu_score`，objective=regression |
| `tools/acceptance_25_8.py` | **新增** | 200 局 vs ReferenceHumanPolicy，输出 25 个窗口分数 + worst-case 复盘 |
| `tools/perf_regression.py` | **新增** | 4 个最坏牌型 P95 检测，超 800ms 失败 |
| `tools/iterative_train.py` | **新增** | 串联 R0-R3 多轮训练流水线 |
| `backend/app/services/reference_human.py` | **新增** | 离线代理对手 |

### 4.2 迭代轮次

| 轮 | 对手对 | 样本量 | 训练动作 | 预计耗时（CPU） |
|---|---|---|---|---|
| R0 | ReferenceHumanPolicy vs random（V3 尚未训练，仅生成数据） | 50k 局 | 6 head 全 GBDT，产出 V3(R0) | ~6 小时 |
| R1 | V3(R0) vs ReferenceHumanPolicy | 50k 局 | 6 head 重训，重点校准 houjuu / betaori | ~8 小时 |
| R2 | V3(R1) vs V3(R1)（温度采样 0.5） | 50k 局 | 加深 search depth 到 3，再训 | ~12 小时 |
| R3 | V3(R2) vs ReferenceHumanPolicy | — | 验收回归 | ~1 小时 |

总预计：单机 CPU 约 30 小时纯训练 + 调试余量；4 周节奏宽松。

### 4.3 GPU 入口（暂不实现，留接口）

`tools/iterative_train.py` 增加 `--accelerator {cpu,gpu}`：
- `cpu`（默认）：LightGBM GBDT，本期实现。
- `gpu`：预留 PyTorch MLP 多 head 路径（已经有 `tools/train_neural.py`），目前只输出"GPU path: not implemented in this milestone, contact for quote"。
- 报价说明在 `docs/PRICING.md`（独立文档，不在本 spec 范围）。

### 4.4 防守特征扩展（per-tile）

新增特征族（**训练侧 `tools/extract_canonical_states.py::build_model_features` 与 C++ 侧 `linhai_search_v3.cpp::build_state_features` 必须同步加**，参考 `docs/FEATURES.md` 既有合约）：

| 前缀 | 含义 |
|---|---|
| `safety_genbutsu_t_X` | X 是否在对手牌河（现物） |
| `safety_suji_t_X` | X 是否被筋（如 4w 出过则 1w/7w 安全度+） |
| `safety_kabe_t_X` | X 可见数 ≥ 3 时相邻数牌安全度+ |
| `opp_meld_pressure` | 对手副露数 / 副露牌威胁度 |
| `opp_riichi_proxy` | 对手疑似听牌（弃牌从中张转字风牌） |

特征名同步检查由 `tests/python/test_feature_parity.py` 兜底（**新增**）。

---

## 5. 性能预算

### 5.1 预算分配（P95）

| 项 | 预算 | 说明 |
|---|---|---|
| 弃牌主搜（depth 3 + beam 4 + GBDT） | 650ms | 主消耗 |
| 响应主搜 | 500ms | 吃碰杠胡决策 |
| 番数计算 | 50ms | `linhai_score::calc_chong` |
| shanten / EV 缓存 | <5ms | 命中即返回 |
| **引擎合计 P95** | **≤ 800ms** | 留 1200ms 给网络/序列化/抖动 |
| **端到端 P95** | **≤ 2000ms** | 满足图条款 |

### 5.2 自适应降级

```
if elapsed > 60% budget:  beam = max(2, beam - 1)
if elapsed > 80% budget:  depth = max(1, depth - 1)
if elapsed > 95% budget:  switch to V2 fallback
```

### 5.3 性能回归门禁

`tools/perf_regression.py` 测 4 个最坏牌型，P95 > 800ms 阻塞 merge：
- 4 张白板 + 普通牌型
- 3 个副露 + 听牌响应
- 多抢杠候选场景
- 抓冲红头全开 + 树掉还原

---

## 6. 验收与交付

### 6.1 离线 25/25 回归（CI 强制）

`tools/acceptance_25_8.py`：
- 200 局 AI vs ReferenceHumanPolicy，按 8-局窗口算积分
- 输出：`acceptance_report.json`（25 个窗口分数 + 通过/失败标记 + 最差 3 个窗口对局复盘）
- 通过门槛（T6 默认）：**25/25 窗口分数 > 0** 且 **平均冲数 / 局 ≥ 0.3**
- CI 失败条件：任一窗口 ≤ 0 或 P95 引擎延迟 > 800ms

### 6.2 交付物清单（按图"三、交付内容"）

| 项 | 路径 / 说明 |
|---|---|
| 完整源代码（C++ + Python） | 仓库本身，无 `.pyc` 隐藏；`engine/`、`backend/`、`tools/`、`tests/` |
| 6 个 head 的 model 包 | `engine/params/v3/{agari_score,houjuu_score,...}/model.json` + `model_meta.json` |
| 自对弈样本子集 | `data/release/selfplay_sample_R3_10pct.jsonl`（10% 抽样，约 5w 行） |
| 番数表配置 | `engine/params/v3/score_table.json` |
| 部署运行手册 | `docs/DEPLOYMENT.md`（含 `/debug/engine_status` 检查方法） |
| 25/25 回归报告 | `docs/release/acceptance_report.html` + `acceptance_report.csv` |
| 测试套件 | `tests/cpp/`、`tests/python/`，含本期新增的 `test_linhai_score.py` 等 |

### 6.3 未尽决策项（spec 决议默认值，可后期调整）

| # | 项 | 默认值 |
|---|---|---|
| T1 | 字一色冲数 | 8 |
| T2 | 抢杠胡冲数 | +2 |
| T3 | 三键承包公式 | 两家平摊全部冲数 |
| T4 | 翻屁股每人发牌数 | 2 |
| T5 | 流局张数 | 6 |
| T6 | 验收 R3 通过门槛 | 25/25 全正 + 平均冲数 ≥ 0.3 |
| T7 | 训练资源 | CPU；GPU 留接口不实现 |
| T8 | 验收周期 | 4 周（R0-R3） |
| T9 | 四人抓红头玩法 AI | 不做四人 AI；只做发牌/抓冲机制；AI 估值仍用双人模型 |

---

## 7. 风险与回滚

| 风险 | 影响 | 缓解 |
|---|---|---|
| 真人对战胜率不达预期 | 验收失败 | R3 后增加 "human-in-the-loop" 调参轮，提前内部找几人测试 |
| 番数引擎漏条款 | 自对弈结算与真人结算不一致 | 单元测试覆盖图全部条款；`tests/python/test_linhai_score.py` 保证 |
| 训练 / 推理特征漂移 | head 静默退化为 0 | `test_feature_parity.py` CI 兜底 |
| 性能回归 | P95 > 800ms | `perf_regression.py` 阻塞 merge |
| 三层 fallback 静默生效 | 实际跑 heuristic | `/debug/engine_status` 必查；`active_engine != "v3"` 上线前阻塞 |

回滚：每轮训练前打 `git tag pre-Rn`，参数 bundle 按 `params/v3/release/Rn/` 隔离，可一键切回上一版。

---

## 8. 不在本 spec 范围

- 四人抓红头玩法的 AI 决策（仅做发牌 / 抓冲机制）
- GPU 训练路径的实际实现（仅留入口）
- 真人对战的实时数据回流闭环（一期不做）
- 客户端前端改动（仅引擎与服务层）

---

## 9. 实施顺序（顶层）

具体 step-by-step 计划由 writing-plans skill 在本 spec 通过后产出。顶层顺序：

1. 番数引擎 `linhai_score` + 单元测试 + score_table.json
2. ReferenceHumanPolicy + acceptance_25_8.py（先用现有模型跑通流水）
3. 改造 selfplay_sample / train_model_stub 走 score_label 路径
4. 防守特征扩展 + feature_parity 测试
5. iterative_train R0-R2
6. 性能回归 perf_regression.py
7. R3 验收 + 交付物打包
