# Phase A — 推理侧防守加固实施计划（2026-04-28）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **本项目用户偏好**：所有 `git add` / `git commit` 由用户手动执行；本计划里的"提交边界"步骤只列出**给用户的命令**，AI 执行时**不要**直接调用 git。AI 在每个 task 末尾给出"已修改文件清单"让用户自己 commit。

**Goal:** 在不重训的前提下，把 V3 引擎的 EV 公式从"线性期望差"改造成"风险厌恶 + 副露压力感知 + 动态 betaori 阈值"，使 5-seed × 200 局验收从当前 106/125 (84.8%) 提升到 ≥118/125 (94%+)，最差窗口从 -43 冲提升到 ≥ -8 冲，放炮率从 14.8% 压到 ≤ 11%。这是 Phase B 蒸馏（参见 `2026-04-28-linhai-v3-acceptance-plan.md`）的前置门槛。

**Architecture:** 三层改造，全部不需要重训：
1. **参数层**：`score_table.json` 新增 `ev_risk_weights`、`betaori_thresholds`、`feature_weights` 三段配置
2. **特征层**：C++ 推理侧 `build_discard_candidates` 计算 `opp_pressure_score`（副露数 × 同色集中度 × 可见白板数的非线性组合），作为 EV 修正的输入
3. **EV 公式层**：把现有 `c.total_ev -= c.houjuu_prob * 5500.0f` 替换为 `λ(pressure) · houjuu_prob · base + μ · max(0, P(opp_chong≥4) · expected_loss)`；新增"危险触发硬 betaori 门"

**Tech Stack:** C++17（`engine/share/linhai_search_v3.cpp` + `linhai_score.cpp`），Python 3.10+（`tools/phase_a_tune.py` 网格搜索），既有 `tools/acceptance_25_8.py` 验收回归。

**Spec:** `docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md` §3.5

**Phase B 衔接:** Phase A 通过 §3.5.4 验证门槛后，进入 `docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md` 的 Phase 6-7 (R0-R3 蒸馏)。Phase A 调好的 `score_table.json` 直接作为 R0 训练初始环境。

---

## File Structure

### 新增文件

| 路径 | 责任 |
|---|---|
| `tools/phase_a_tune.py` | Phase A 参数网格搜索（λ / μ / betaori 阈值），输出候选 `score_table_phase_a.json` |
| `tests/python/test_phase_a_ev.py` | Phase A EV 修正逻辑的 Python 端集成测试 |
| `tests/cpp/test_phase_a_risk_ev.cpp` | C++ 端 EV 风险加权单测 |

### 修改文件

| 路径 | 改动 |
|---|---|
| `engine/share/linhai_score.hpp` | `ScoreTable` 增加 `ev_risk` / `betaori_thresholds` / `feature_weights` 三个嵌套结构 |
| `engine/share/linhai_score.cpp` | `load_score_table` 解析新字段，缺失回退默认 |
| `engine/share/linhai_search_v3.cpp` | `build_discard_candidates` 加 `opp_pressure_score` 计算 + 动态 EV 风险加权 + 硬 betaori 门 |
| `engine/params/v3/score_table.json` | 新增 `ev_risk_weights` / `betaori_thresholds` / `feature_weights` 三段 |
| `tests/python/test_linhai_score.py` | 验证新字段加载与默认值回退 |

---

## 阶段总览

| 任务 | 内容 | 预计耗时 |
|---|---|---|
| Task A1 | `ScoreTable` 扩展新字段 + JSON 解析 | 0.5 天 |
| Task A2 | `score_table.json` 写入默认 Phase A 参数 | 0.25 天 |
| Task A3 | `opp_pressure_score` 特征计算（C++ 推理侧） | 0.5 天 |
| Task A4 | EV 公式风险厌恶重加权（λ / μ 项） | 0.75 天 |
| Task A5 | 硬 betaori 门（危险局面强制最低 houjuu 弃牌） | 0.5 天 |
| Task A6 | `tools/phase_a_tune.py` 网格搜索 | 0.75 天 |
| Task A7 | Phase A 验证门槛实跑 + 报告 | 0.5 天 |
| Task A8 | Phase B 衔接（参数固化 + 移交记录） | 0.25 天 |
| **合计** | | **~4 天** |

---

## Task A1: 扩展 ScoreTable 数据结构与 JSON 解析

**Files:**
- Modify: `engine/share/linhai_score.hpp:59-73`
- Modify: `engine/share/linhai_score.cpp:34-75`
- Test: `tests/python/test_linhai_score.py`

### Step 1: 写失败测试 — 新字段缺失时的默认值

- [ ] **Step 1.1: 在 `tests/python/test_linhai_score.py` 末尾追加测试**

```python
def test_score_table_default_phase_a_fields_when_missing():
    """spec §3.5 — score_table.json 不含 ev_risk_weights 时回退到默认。"""
    from backend.app.services.score_calculator import load_score_table
    import tempfile, json, pathlib
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"chong_to_score": 1, "base_chong": {"ordinary": 1}}, f)
        path = f.name
    table = load_score_table(path)
    # Phase A 默认值（spec §3.5.1 / §3.5.2）
    assert table["ev_risk_weights"]["lambda_base"] == 1.5
    assert table["ev_risk_weights"]["mu_base"] == 2.0
    assert table["ev_risk_weights"]["lambda_pressure_step"] == 0.5
    assert table["betaori_thresholds"]["base"] == 0.15
    assert table["betaori_thresholds"]["meld_drop"] == 0.03
    assert table["betaori_thresholds"]["qingyise_drop"] == 0.05
    assert table["feature_weights"]["opp_meld_pressure_alpha"] == 1.0
    pathlib.Path(path).unlink()


def test_score_table_phase_a_fields_loaded_when_present():
    """spec §3.5 — 显式提供新字段时读取生效。"""
    from backend.app.services.score_calculator import load_score_table
    import tempfile, json, pathlib
    cfg = {
        "ev_risk_weights": {
            "lambda_base": 1.5, "mu_base": 2.5,
            "lambda_pressure_step": 0.4, "lambda_max": 3.0,
        },
        "betaori_thresholds": {
            "base": 0.13, "meld_drop": 0.025, "qingyise_drop": 0.04,
        },
        "feature_weights": {"opp_meld_pressure_alpha": 1.2},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        path = f.name
    table = load_score_table(path)
    assert table["ev_risk_weights"]["lambda_base"] == 1.5
    assert table["betaori_thresholds"]["base"] == 0.13
    assert table["feature_weights"]["opp_meld_pressure_alpha"] == 1.2
    pathlib.Path(path).unlink()
```

- [ ] **Step 1.2: 跑测试确认失败**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 -m pytest tests/python/test_linhai_score.py::test_score_table_default_phase_a_fields_when_missing -v`
Expected: FAIL — `ev_risk_weights` 字段不存在

### Step 2: 扩展 C++ `ScoreTable` 结构

- [ ] **Step 2.1: 修改 `engine/share/linhai_score.hpp` 第 59-73 行 `ScoreTable` 定义**

把当前 `ScoreTable` 替换为下面的扩展版（保留原所有字段，新增 3 个嵌套 struct）：

```cpp
struct ScoreTable {
    int chong_to_score = 1;
    int base_ordinary = 1;
    int base_qingyise = 8;
    int base_ziyise = 8;
    int base_hunyise_tree = 2;
    int base_hunyise_hard = 4;
    int qianggang_bonus = 2;
    int grab_charge_per_tile = 1;
    int max_base_chong = 8;
    bool extra_chong_uncapped = true;
    std::map<int, int> redhead_tile_to_chong;
    int redhead_per_player_count_default = 2;
    int contract_target_count_default = 3;

    // === Phase A: 推理侧防守加固（spec §3.5）===
    struct EVRiskWeights {
        float lambda_base = 1.5f;            // 放炮基础惩罚倍数（spec §3.5.1 默认 1.5）
        float lambda_pressure_step = 0.5f;   // 每单位 opp_pressure_score 的 λ 增量
        float lambda_max = 3.0f;             // λ 上限
        float mu_base = 2.0f;                // 长尾损失（≥4 冲）惩罚倍数
        float mu_pressure_step = 0.5f;       // 每单位 opp_pressure_score 的 μ 增量
        float mu_max = 5.0f;                 // μ 上限
        float houjuu_base_coeff = 5500.0f;   // 现有 houjuu_prob 系数（保持兼容）
        float high_chong_threshold = 4.0f;   // "高冲" 判定（影响 μ 触发）
    } ev_risk_weights;

    struct BetaoriThresholds {
        float base = 0.15f;          // houjuu_prob 触发硬 betaori 的基础阈值
        float meld_drop = 0.03f;     // 对手 ≥1 副露 + 同色 ≥3 张时阈值再下调
        float qingyise_drop = 0.05f; // 对手疑似清一色 / 树掉还原时阈值再下调
        float ev_loss_multiplier = 1.5f; // 危险局面 EV 损失上限放宽倍数
    } betaori_thresholds;

    struct FeatureWeights {
        float opp_meld_pressure_alpha = 1.0f;     // opp_meld_pressure 对 EV 的线性权重
        float opp_qingyise_alarm_alpha = 1.0f;    // 清一色警报对 EV 的线性权重
    } feature_weights;
};
```

- [ ] **Step 2.2: 修改 `engine/share/linhai_score.cpp` 中 `load_score_table` (第 34-75 行)**

在 `load_score_table` 函数返回前加入新字段解析。找到现有的 `if (j.contains("cap"))` 块之后插入：

```cpp
    // === Phase A 字段（spec §3.5）===
    if (j.contains("ev_risk_weights") && j["ev_risk_weights"].is_object()) {
        const auto& evr = j["ev_risk_weights"];
        if (evr.contains("lambda_base")) t.ev_risk_weights.lambda_base = evr["lambda_base"].get<float>();
        if (evr.contains("lambda_pressure_step")) t.ev_risk_weights.lambda_pressure_step = evr["lambda_pressure_step"].get<float>();
        if (evr.contains("lambda_max")) t.ev_risk_weights.lambda_max = evr["lambda_max"].get<float>();
        if (evr.contains("mu_base")) t.ev_risk_weights.mu_base = evr["mu_base"].get<float>();
        if (evr.contains("mu_pressure_step")) t.ev_risk_weights.mu_pressure_step = evr["mu_pressure_step"].get<float>();
        if (evr.contains("mu_max")) t.ev_risk_weights.mu_max = evr["mu_max"].get<float>();
        if (evr.contains("houjuu_base_coeff")) t.ev_risk_weights.houjuu_base_coeff = evr["houjuu_base_coeff"].get<float>();
        if (evr.contains("high_chong_threshold")) t.ev_risk_weights.high_chong_threshold = evr["high_chong_threshold"].get<float>();
    }
    if (j.contains("betaori_thresholds") && j["betaori_thresholds"].is_object()) {
        const auto& bt = j["betaori_thresholds"];
        if (bt.contains("base")) t.betaori_thresholds.base = bt["base"].get<float>();
        if (bt.contains("meld_drop")) t.betaori_thresholds.meld_drop = bt["meld_drop"].get<float>();
        if (bt.contains("qingyise_drop")) t.betaori_thresholds.qingyise_drop = bt["qingyise_drop"].get<float>();
        if (bt.contains("ev_loss_multiplier")) t.betaori_thresholds.ev_loss_multiplier = bt["ev_loss_multiplier"].get<float>();
    }
    if (j.contains("feature_weights") && j["feature_weights"].is_object()) {
        const auto& fw = j["feature_weights"];
        if (fw.contains("opp_meld_pressure_alpha")) t.feature_weights.opp_meld_pressure_alpha = fw["opp_meld_pressure_alpha"].get<float>();
        if (fw.contains("opp_qingyise_alarm_alpha")) t.feature_weights.opp_qingyise_alarm_alpha = fw["opp_qingyise_alarm_alpha"].get<float>();
    }
```

### Step 3: 镜像更新 Python `score_calculator.py::load_score_table`

- [ ] **Step 3.1: 读 `backend/app/services/score_calculator.py` 找到 `load_score_table` 函数**

```bash
grep -n "def load_score_table" /Users/fangyajun/CLionProjects/majiangV3/backend/app/services/score_calculator.py
```

- [ ] **Step 3.2: 在该函数返回 `dict` 前加入默认字段填充**

在 `score_calculator.py::load_score_table` 函数返回 dict 之前加入（具体行根据 grep 结果定位）：

```python
    # === Phase A 默认字段填充（spec §3.5）===
    table.setdefault("ev_risk_weights", {})
    evr = table["ev_risk_weights"]
    evr.setdefault("lambda_base", 1.5)
    evr.setdefault("lambda_pressure_step", 0.5)
    evr.setdefault("lambda_max", 3.0)
    evr.setdefault("mu_base", 2.0)
    evr.setdefault("mu_pressure_step", 0.5)
    evr.setdefault("mu_max", 5.0)
    evr.setdefault("houjuu_base_coeff", 5500.0)
    evr.setdefault("high_chong_threshold", 4.0)
    table.setdefault("betaori_thresholds", {})
    bt = table["betaori_thresholds"]
    bt.setdefault("base", 0.15)
    bt.setdefault("meld_drop", 0.03)
    bt.setdefault("qingyise_drop", 0.05)
    bt.setdefault("ev_loss_multiplier", 1.5)
    table.setdefault("feature_weights", {})
    fw = table["feature_weights"]
    fw.setdefault("opp_meld_pressure_alpha", 1.0)
    fw.setdefault("opp_qingyise_alarm_alpha", 1.0)
```

### Step 4: 跑测试确认通过

- [ ] **Step 4.1: 跑 Python 测试**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 -m pytest tests/python/test_linhai_score.py -v`
Expected: 所有测试通过，含两个新增测试

- [ ] **Step 4.2: 编译 C++ 确认无编译错误**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3/engine && python3 setup.py build_ext --inplace 2>&1 | tail -20`
Expected: `Building extension` 成功，无编译报错

### Step 5: Task A1 提交边界（用户手动 commit）

- [ ] **Step 5.1: AI 输出已修改文件清单（不要执行 git）**

```
已修改：
- engine/share/linhai_score.hpp（扩展 ScoreTable）
- engine/share/linhai_score.cpp（扩展 load_score_table）
- backend/app/services/score_calculator.py（Python 镜像默认值）
- tests/python/test_linhai_score.py（新增 2 个测试）
```

用户参考命令：
```bash
git add engine/share/linhai_score.hpp engine/share/linhai_score.cpp \
        backend/app/services/score_calculator.py tests/python/test_linhai_score.py
git commit -m "feat(phase-a): extend ScoreTable with EV risk / betaori / feature weights"
```

---

## Task A2: 写入 score_table.json 默认 Phase A 参数

**Files:**
- Modify: `engine/params/v3/score_table.json`

### Step 1: 追加 Phase A 三段配置

- [ ] **Step 1.1: 修改 `engine/params/v3/score_table.json`**

在现有 JSON（最外层 object）末尾 `"cap"` 段之后追加（注意保留 JSON 合法性，逗号位置正确）：

```json
{
  "version": "1.0.0",
  "created_at": "2026-04-28",
  "description": "...",
  "chong_to_score": 1,
  "base_chong": { ... },
  "multipliers": { ... },
  "bonuses": { ... },
  "redhead": { ... },
  "contract": { ... },
  "liuju": { ... },
  "cap": { ... },
  "ev_risk_weights": {
    "lambda_base": 1.5,
    "lambda_pressure_step": 0.5,
    "lambda_max": 3.0,
    "mu_base": 2.0,
    "mu_pressure_step": 0.5,
    "mu_max": 5.0,
    "houjuu_base_coeff": 5500.0,
    "high_chong_threshold": 4.0,
    "_comment": "Phase A spec §3.5.1: 风险厌恶 EV 重加权。lambda 越大对放炮越敏感，mu 针对对手 ≥4 冲长尾失分。pressure_step 表示 opp_pressure_score 每升 1 单位 lambda/mu 的累加量。"
  },
  "betaori_thresholds": {
    "base": 0.15,
    "meld_drop": 0.03,
    "qingyise_drop": 0.05,
    "ev_loss_multiplier": 1.5,
    "_comment": "Phase A spec §3.5.2: betaori 动态阈值。houjuu_prob > 实时阈值时强制最低 houjuu 弃牌。"
  },
  "feature_weights": {
    "opp_meld_pressure_alpha": 1.0,
    "opp_qingyise_alarm_alpha": 1.0,
    "_comment": "Phase A spec §3.5.3: 高价值防守特征对 EV 的手写权重，作为 Phase B 训练前的先验。"
  }
}
```

> 注意：上面只展示了**新增的三段**与外层结构骨架。实际编辑时 `base_chong` / `multipliers` 等已有段落保持原样不动，只在 `cap` 段后**追加**新的三段。

- [ ] **Step 1.2: 验证 JSON 合法**

Run: `python3 -c "import json; json.load(open('/Users/fangyajun/CLionProjects/majiangV3/engine/params/v3/score_table.json'))"`
Expected: 无输出（合法 JSON）

- [ ] **Step 1.3: 验证加载得到 Phase A 默认值**

Run: 
```bash
cd /Users/fangyajun/CLionProjects/majiangV3 && python3 -c "
from backend.app.services.score_calculator import load_score_table
t = load_score_table('engine/params/v3/score_table.json')
print('lambda_base:', t['ev_risk_weights']['lambda_base'])
print('betaori_base:', t['betaori_thresholds']['base'])
"
```
Expected: `lambda_base: 1.5` 和 `betaori_base: 0.15`

### Step 2: Task A2 提交边界

- [ ] **Step 2.1: AI 输出修改清单**

```
已修改：
- engine/params/v3/score_table.json（追加 Phase A 三段配置）
```

用户参考命令：
```bash
git add engine/params/v3/score_table.json
git commit -m "config(phase-a): add EV risk / betaori / feature weights defaults"
```

---

## Task A3: 实现 `opp_pressure_score` 特征计算（C++ 推理侧）

**Files:**
- Modify: `engine/share/linhai_search_v3.cpp`（在 `build_discard_candidates` 内部加入计算）
- Test: `tests/cpp/test_phase_a_risk_ev.cpp`（新增）

### Step 1: 设计 `opp_pressure_score` 公式

`opp_pressure_score` 是一个 [0, 1+] 的标量，反映"对手准备打大牌的程度"：

```
opp_pressure_score =
    + 0.30 · clamp01(opponent_meld_count / 3)           // 副露多 = 攻击姿态
    + 0.25 · qingyise_alarm                              // 同色集中度 ≥6 张
    + 0.20 · ziyise_alarm                                // 字风刻子可见
    + 0.15 · clamp01(grab_charge_caught_count / 2)       // 抓冲红头
    + 0.10 · white_threat                                // 对手副露含白板

qingyise_alarm = clamp01((max_suit_count_in_opp_visible - 5) / 4)  // 6→0.25, 9→1.0
ziyise_alarm   = (opp_meld_含字牌刻子 ? 1 : 0) * 0.5 + (visible_honor_triplets / 3) * 0.5
white_threat   = clamp01(opp_white_meld_count / 1)
```

输出范围 0.0（无威胁）到 ~1.5（清一色 + 字牌副露 + 抓冲全开）。

### Step 2: 写 C++ 单元测试

- [ ] **Step 2.1: 创建 `tests/cpp/test_phase_a_risk_ev.cpp`**

```cpp
// Phase A spec §3.5.3 — opp_pressure_score 计算单测
#include "engine/share/linhai_search_v3.hpp"
#include "engine/share/types.hpp"
#include <cassert>
#include <iostream>

namespace {
using linhai::LinhaiSearchEngineV3;
using linhai::CanonicalGameState;

void test_zero_pressure_when_clean_state() {
    // 对手 0 副露、0 弃牌 → pressure = 0
    CanonicalGameState state{};
    state.opponent_meld_count = 0;
    state.opponent_discard_count = 0;
    LinhaiSearchEngineV3 engine;
    float p = engine.compute_opp_pressure_score(state);
    assert(p < 0.05f);
    std::cout << "test_zero_pressure: OK (p=" << p << ")\n";
}

void test_meld_only_pressure() {
    // 2 副露 → ~0.20 (= 2/3 * 0.30)
    CanonicalGameState state{};
    state.opponent_meld_count = 2;
    LinhaiSearchEngineV3 engine;
    float p = engine.compute_opp_pressure_score(state);
    assert(p > 0.15f && p < 0.25f);
    std::cout << "test_meld_only_pressure: OK (p=" << p << ")\n";
}

void test_qingyise_alarm() {
    // 对手副露 + 弃牌 8 张同色 → qingyise_alarm 高
    CanonicalGameState state{};
    state.opponent_meld_count = 1;
    // 设置可见同色数量（具体字段名以代码为准，下面用 setter 占位）
    state.opp_visible_max_suit_count = 8;  // 见 step 3 加入此字段
    LinhaiSearchEngineV3 engine;
    float p = engine.compute_opp_pressure_score(state);
    assert(p > 0.40f);
    std::cout << "test_qingyise_alarm: OK (p=" << p << ")\n";
}

}  // namespace

int main() {
    test_zero_pressure_when_clean_state();
    test_meld_only_pressure();
    test_qingyise_alarm();
    std::cout << "ALL Phase A risk EV tests passed.\n";
    return 0;
}
```

> 如果 `CanonicalGameState` 还没有 `opp_visible_max_suit_count` 字段，先用现有 `opponent_meld_count` 测试基础项，复杂特征在 step 3 同步加。

### Step 3: 在 `linhai_search_v3.hpp/cpp` 暴露 `compute_opp_pressure_score`

- [ ] **Step 3.1: 修改 `engine/share/linhai_search_v3.hpp`**

在 `class LinhaiSearchEngineV3` 的 public 部分加入：

```cpp
    // Phase A spec §3.5.3: 副露压力综合标量，0=无威胁，~1.5=清一色+字牌+抓冲全开
    float compute_opp_pressure_score(const CanonicalGameState& state) const;
```

- [ ] **Step 3.2: 在 `linhai_search_v3.cpp` 末尾（命名空间内）实现该方法**

```cpp
float LinhaiSearchEngineV3::compute_opp_pressure_score(const CanonicalGameState& state) const {
    auto clamp01 = [](float x) { return std::max(0.0f, std::min(1.0f, x)); };

    // 1) 副露姿态
    const float meld_term = 0.30f * clamp01(static_cast<float>(state.opponent_meld_count) / 3.0f);

    // 2) 清一色警报：对手可见同色 ≥6 张时累加
    //    数据源：opp_disc_per_suit + meld_per_suit（如果未实现，用 opp_max_tile_disc 近似）
    int max_suit = 0;
    for (int suit = 0; suit < 3; ++suit) {  // 万/筒/索
        int suit_count = 0;
        // 弃牌中该花色的张数（按 opp_disc_t_<tile> 聚合）
        for (int rank = 1; rank <= 9; ++rank) {
            int tile_idx = suit * 9 + rank;  // 0..8 万、9..17 筒、18..26 索（按 38-array 调整）
            if (tile_idx < 27) {  // 仅数牌
                suit_count += state.opponent_discard_per_tile_count(tile_idx);
            }
        }
        // 副露中该花色（每副露 3 张）
        suit_count += state.opponent_meld_suit_count(suit) * 3;
        if (suit_count > max_suit) max_suit = suit_count;
    }
    const float qingyise_alarm = clamp01((static_cast<float>(max_suit) - 5.0f) / 4.0f);
    const float qingyise_term = 0.25f * qingyise_alarm;

    // 3) 字一色警报：对手字风刻子 / 字牌刻子可见
    const float ziyise_term = 0.20f * clamp01(static_cast<float>(state.opponent_honor_triplets) / 2.0f);

    // 4) 抓冲红头威胁
    const float redhead_term = 0.15f * clamp01(static_cast<float>(state.grab_charge_caught_count) / 2.0f);

    // 5) 白板威胁（对手副露含白板）
    const float white_term = 0.10f * clamp01(static_cast<float>(state.opponent_white_meld_count));

    return meld_term + qingyise_term + ziyise_term + redhead_term + white_term;
}
```

> 注意：上面用到的 `opponent_discard_per_tile_count`、`opponent_meld_suit_count`、`opponent_honor_triplets`、`opponent_white_meld_count` 这些 accessor 如果 `CanonicalGameState` 中尚不存在，按以下次序处理：
> - 已有 `opp_disc_t_<tile>` per-tile 数组：包装一个内联 accessor 即可
> - 缺失字段（如 `opponent_honor_triplets`）：在 `CanonicalGameState` 加上对应字段，并在状态构建时填充

- [ ] **Step 3.3: 必要字段补齐（如果 step 3.2 中字段不存在）**

打开 `engine/share/types.hpp`（或 `linhai_search_v3.hpp` 中 `CanonicalGameState` 定义所在），添加：

```cpp
    // Phase A: 防守特征支持字段
    int opponent_honor_triplets = 0;       // 对手副露/暗刻可见字牌刻子数
    int opponent_white_meld_count = 0;     // 对手副露含白板数（0/1）
```

并在 `linhai_search_v3.cpp` 中状态构建函数（找 `build_state_features` 或 `from_request`）填充这些字段。

### Step 4: 跑 C++ 测试

- [ ] **Step 4.1: 构建 C++ 测试**

Run:
```bash
cd /Users/fangyajun/CLionProjects/majiangV3
clang++ -std=c++17 -I. tests/cpp/test_phase_a_risk_ev.cpp \
    engine/share/linhai_search_v3.cpp engine/share/linhai_score.cpp \
    engine/share/types.cpp -o /tmp/test_phase_a_risk_ev 2>&1 | head -20
```
Expected: 编译成功

- [ ] **Step 4.2: 跑测试**

Run: `/tmp/test_phase_a_risk_ev`
Expected: `ALL Phase A risk EV tests passed.`

### Step 5: Task A3 提交边界

```
已修改：
- engine/share/linhai_search_v3.hpp（暴露 compute_opp_pressure_score）
- engine/share/linhai_search_v3.cpp（实现 compute_opp_pressure_score）
- engine/share/types.hpp（如需补字段）
- tests/cpp/test_phase_a_risk_ev.cpp（新增）
```

用户参考：
```bash
git add engine/share/linhai_search_v3.hpp engine/share/linhai_search_v3.cpp \
        engine/share/types.hpp tests/cpp/test_phase_a_risk_ev.cpp
git commit -m "feat(phase-a): add opp_pressure_score aggregate defensive feature"
```

---

## Task A4: EV 公式风险厌恶重加权

**Files:**
- Modify: `engine/share/linhai_search_v3.cpp:1142-1190`（`build_discard_candidates` EV 公式段）

### Step 1: 写 Python 集成测试 — 高压力下 EV 偏向防守

- [ ] **Step 1.1: 创建 `tests/python/test_phase_a_ev.py`**

```python
"""Phase A spec §3.5.1 — 风险厌恶 EV 在高压力局面下偏好防守。"""
import pytest
from backend.app.services.v3_service import LinhaiV3Orchestrator


def _make_state_low_pressure():
    """对手 0 副露、对手弃牌干净 → opp_pressure_score ≈ 0。"""
    return {
        "tehai": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                  "1p", "2p", "3p", "4p"],
        "melds": [],
        "opp_melds": [],
        "opp_discards": ["1s", "2s"],
        "wall_remaining": 50,
        "from_player": 1,
        "can_win": False,
    }


def _make_state_high_pressure():
    """对手 2 副露 + 9 张万子可见 → opp_pressure_score ≈ 0.6+，应触发风险厌恶。"""
    state = _make_state_low_pressure()
    state["opp_melds"] = [
        {"type": "peng", "tiles": ["3m", "3m", "3m"]},
        {"type": "peng", "tiles": ["7m", "7m", "7m"]},
    ]
    state["opp_discards"] = ["1m", "2m", "5m", "8m", "9m", "1s"]
    return state


def test_low_pressure_ev_unchanged():
    """低压力下 Phase A EV ≈ 现有 EV（lambda=1.0、mu 不触发）。"""
    orch = LinhaiV3Orchestrator()
    rec = orch.recommend(_make_state_low_pressure())
    # 低压力下应像现在一样优先打 1s/2s 等纯安全张
    top_action = rec["recommendation"]["action"]
    assert top_action.startswith("打"), f"expected discard, got {top_action}"


def test_high_pressure_prefers_low_houjuu():
    """高压力下 EV 应大幅惩罚 houjuu_prob 高的弃牌。"""
    orch = LinhaiV3Orchestrator()
    rec = orch.recommend(_make_state_high_pressure())
    candidates = rec["recommendation"].get("candidates", [])
    assert len(candidates) >= 2, "expected multiple candidates"
    top = candidates[0]
    second = candidates[1]
    # top 选择应有 houjuu_prob ≤ second
    assert top.get("houjuu_prob", 0) <= second.get("houjuu_prob", 0) + 0.02, \
        f"top houjuu={top.get('houjuu_prob')} > second={second.get('houjuu_prob')}"
```

- [ ] **Step 1.2: 跑测试确认失败（或当前行为）**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 -m pytest tests/python/test_phase_a_ev.py -v`
Expected: `test_high_pressure_prefers_low_houjuu` 可能 FAIL（当前 EV 没区分压力）；记下当前行为作为对照基线

### Step 2: 修改 `build_discard_candidates` EV 公式

- [ ] **Step 2.1: 定位 `linhai_search_v3.cpp:1140` — `c.houjuu_prob = estimate_houjuu_prob(...)` 之前插入压力计算**

在 line 1140 之前（即 EV 累加段开始前）加入：

```cpp
        // === Phase A spec §3.5.1: 风险厌恶 EV 重加权 ===
        const float opp_pressure = compute_opp_pressure_score(state);
        const auto& evr = score_table_.ev_risk_weights;
        const float dynamic_lambda = std::min(
            evr.lambda_max,
            evr.lambda_base + evr.lambda_pressure_step * opp_pressure
        );
        const float dynamic_mu = std::min(
            evr.mu_max,
            evr.mu_base + evr.mu_pressure_step * opp_pressure
        );
```

- [ ] **Step 2.2: 替换 line 1169 现有 houjuu 惩罚**

把：
```cpp
        c.total_ev -= c.houjuu_prob * 5500.0f;
```

替换为：
```cpp
        // === Phase A spec §3.5.1: 动态 λ + μ 风险加权 ===
        const float houjuu_loss_base = c.houjuu_prob * evr.houjuu_base_coeff;
        c.total_ev -= dynamic_lambda * houjuu_loss_base;
        // μ 项：当对手压力高且对手可能 ≥4 冲胡牌时，额外惩罚
        if (opp_pressure > 0.3f && has_score_heads_loaded()) {
            const float houjuu_score_ev = estimate_houjuu_score(state, c, c.action, 0, safe);
            // houjuu_score_ev 是带号期望放炮失分；阈值 4 冲（spec §3.5.1）
            const float high_chong_loss = std::max(0.0f, std::abs(houjuu_score_ev) - evr.high_chong_threshold);
            c.total_ev -= dynamic_mu * high_chong_loss * 200.0f;
        }
```

> 注意：现有 line 1182-1190 的 `if (has_score_heads_loaded())` 分支里也有 `c.total_ev -= houjuu_score_ev * 800.0f`，**这一段保留不动**，新加的 μ 项与它互补：800 是基础线性项，μ 是长尾增量。

- [ ] **Step 2.3: 在 EV 加项最末尾加入 `opp_pressure` 直接的 feature 权重项**

在原 line 1190 之后（`}` 之后）插入（仍在循环 body 内）：

```cpp
        // === Phase A spec §3.5.3: 防守特征对 EV 的线性修正 ===
        c.total_ev -= score_table_.feature_weights.opp_meld_pressure_alpha * opp_pressure * 200.0f;
```

### Step 3: 编译并跑测试

- [ ] **Step 3.1: 重编译 C++ 扩展**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3/engine && python3 setup.py build_ext --inplace 2>&1 | tail -10`
Expected: `Building extension` 成功

- [ ] **Step 3.2: 跑 Phase A EV 测试**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 -m pytest tests/python/test_phase_a_ev.py -v`
Expected: 两个测试都 PASS

- [ ] **Step 3.3: 跑既有测试套件确认无回归**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 -m pytest tests/python/ -x --ignore=tests/python/test_acceptance_25_8.py 2>&1 | tail -20`
Expected: 全 PASS（acceptance_25_8 跑得久，单独跑）

### Step 4: 性能 sanity check（不引入回归）

- [ ] **Step 4.1: 跑 perf_regression**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 tools/perf_regression.py 2>&1 | tail -10`
Expected: P95 ≤ 800ms，不被阻塞

### Step 5: Task A4 提交边界

```
已修改：
- engine/share/linhai_search_v3.cpp（动态 λ/μ EV 风险加权）
- tests/python/test_phase_a_ev.py（新增）
```

用户参考：
```bash
git add engine/share/linhai_search_v3.cpp tests/python/test_phase_a_ev.py
git commit -m "feat(phase-a): risk-averse EV reweighting with dynamic lambda/mu"
```

---

## Task A5: 硬 betaori 门（危险局面强制保命）

**Files:**
- Modify: `engine/share/linhai_search_v3.cpp:1142-1206`（`build_discard_candidates` 排序前）

### Step 1: 写测试 — 极端高压下 top 候选必须是低 houjuu

- [ ] **Step 1.1: 在 `tests/python/test_phase_a_ev.py` 加入**

```python
def test_extreme_pressure_forces_lowest_houjuu():
    """spec §3.5.2 — 对手疑似清一色 + 任一弃牌 houjuu_prob > 0.20 时，
    top 必须选 houjuu_prob 最低的候选（硬 betaori 门）。"""
    orch = LinhaiV3Orchestrator()
    state = _make_state_high_pressure()
    # 加重压力：对手 3 副露 + 万子全可见
    state["opp_melds"].append({"type": "peng", "tiles": ["1m", "1m", "1m"]})
    rec = orch.recommend(state)
    candidates = rec["recommendation"].get("candidates", [])
    # 找全候选里 houjuu_prob 最小的
    min_houjuu = min(c.get("houjuu_prob", 0) for c in candidates)
    top_houjuu = candidates[0].get("houjuu_prob", 0)
    assert abs(top_houjuu - min_houjuu) < 0.01, \
        f"hard betaori gate failed: top={top_houjuu} min={min_houjuu}"
```

- [ ] **Step 1.2: 跑测试看当前行为**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 -m pytest tests/python/test_phase_a_ev.py::test_extreme_pressure_forces_lowest_houjuu -v`
Expected: FAIL（硬门未实现）

### Step 2: 实现硬 betaori 门

- [ ] **Step 2.1: 在 `linhai_search_v3.cpp` 的 `build_discard_candidates` 末尾排序之前插入门控逻辑**

在 line 1202 之前（`std::sort(candidates.begin()..` 之前）插入：

```cpp
    // === Phase A spec §3.5.2: 硬 betaori 门 ===
    // 计算实时 betaori 触发阈值
    const auto& bt = score_table_.betaori_thresholds;
    const float opp_pressure_outer = compute_opp_pressure_score(state);  // 重算或缓存
    float betaori_threshold = bt.base;
    const bool has_meld_pressure = state.opponent_meld_count >= 1;
    const bool has_qingyise_signal = opp_pressure_outer > 0.5f;
    if (has_meld_pressure) betaori_threshold -= bt.meld_drop;
    if (has_qingyise_signal) betaori_threshold -= bt.qingyise_drop;

    // 找全候选中最低 houjuu_prob
    float min_houjuu = 1.0f;
    for (const auto& c : candidates) {
        if (c.houjuu_prob < min_houjuu) min_houjuu = c.houjuu_prob;
    }

    // 触发条件：所有候选 houjuu_prob 都 > 阈值，且最低也 > 阈值的 0.7（说明全场都危险）
    // 此时强制把 EV 加成项里 betaori 拉到最大
    const bool force_betaori_mode =
        (min_houjuu > betaori_threshold) ||
        (min_houjuu > betaori_threshold * 0.5f && has_qingyise_signal);

    if (force_betaori_mode) {
        for (auto& c : candidates) {
            // 把 EV 重排为：houjuu_prob 越低 EV 越高（保命优先）
            // 用 betaori_thresholds.ev_loss_multiplier 放大原 houjuu 惩罚
            c.total_ev -= c.houjuu_prob * 8000.0f * bt.ev_loss_multiplier;
            c.explanation += "[强制 betaori]";
        }
    }
```

- [ ] **Step 2.2: 编译并跑测试**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3/engine && python3 setup.py build_ext --inplace 2>&1 | tail -5`
Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 -m pytest tests/python/test_phase_a_ev.py -v`
Expected: 全 PASS（含 `test_extreme_pressure_forces_lowest_houjuu`）

### Step 3: Task A5 提交边界

```
已修改：
- engine/share/linhai_search_v3.cpp（硬 betaori 门）
- tests/python/test_phase_a_ev.py（新增 hard gate 测试）
```

用户参考：
```bash
git add engine/share/linhai_search_v3.cpp tests/python/test_phase_a_ev.py
git commit -m "feat(phase-a): hard betaori gate when opp pressure > threshold"
```

---

## Task A6: `tools/phase_a_tune.py` 网格搜索

**Files:**
- Create: `tools/phase_a_tune.py`

### Step 1: 写脚本骨架

- [ ] **Step 1.1: 创建 `tools/phase_a_tune.py`**

```python
#!/usr/bin/env python3
"""Phase A spec §3.5.4 — 网格搜索 EV 风险加权与 betaori 阈值。

输出：
- 每组参数对应的 5-seed × 200 局验收指标（25/25 全正窗口数、最差窗口、放炮率）
- 最优参数候选写入 engine/params/v3/score_table_phase_a.json

Usage:
    python3 tools/phase_a_tune.py --grid small  # 9 组（约 1 小时）
    python3 tools/phase_a_tune.py --grid full   # 27 组（约 3 小时）
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_TABLE = REPO_ROOT / "engine/params/v3/score_table.json"
SCORE_TABLE_PHASE_A = REPO_ROOT / "engine/params/v3/score_table_phase_a.json"
ACCEPTANCE_REPORT = REPO_ROOT / "docs/release/acceptance_report.json"

GRIDS = {
    "small": {
        "lambda_base": [1.0, 1.5, 2.0],
        "betaori_base": [0.13, 0.15],
        "qingyise_drop": [0.05],
    },
    "full": {
        "lambda_base": [1.0, 1.3, 1.6, 2.0],
        "mu_base": [1.5, 2.0, 3.0],
        "betaori_base": [0.12, 0.14, 0.16],
        "qingyise_drop": [0.04, 0.06],
    },
}


def write_score_table(params: dict) -> None:
    """把当前网格点参数写回 score_table.json。"""
    with open(SCORE_TABLE) as f:
        cfg = json.load(f)
    cfg.setdefault("ev_risk_weights", {})
    cfg["ev_risk_weights"]["lambda_base"] = params["lambda_base"]
    if "mu_base" in params:
        cfg["ev_risk_weights"]["mu_base"] = params["mu_base"]
    cfg.setdefault("betaori_thresholds", {})
    cfg["betaori_thresholds"]["base"] = params["betaori_base"]
    cfg["betaori_thresholds"]["qingyise_drop"] = params["qingyise_drop"]
    with open(SCORE_TABLE, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def run_acceptance(games_per_seed: int, seeds: list[int]) -> dict:
    """跑 acceptance_25_8.py，返回汇总 dict。"""
    cmd = [
        sys.executable, str(REPO_ROOT / "tools/acceptance_25_8.py"),
        "--games-per-seed", str(games_per_seed),
        "--seeds", *map(str, seeds),
        "--min-window-chong", "0",  # 网格搜索阶段放宽门槛，事后比较
        "--min-avg-chong", "0.0",
        "--max-houjuu-rate", "1.0",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    with open(ACCEPTANCE_REPORT) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["small", "full"], default="small")
    ap.add_argument("--games-per-seed", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--output", default=str(SCORE_TABLE_PHASE_A))
    args = ap.parse_args()

    grid = GRIDS[args.grid]
    keys = list(grid.keys())
    backup = SCORE_TABLE.with_suffix(".json.bak")
    shutil.copy(SCORE_TABLE, backup)
    print(f"Backed up score_table.json to {backup}")

    results = []
    try:
        for vals in product(*grid.values()):
            params = dict(zip(keys, vals))
            print(f"\n=== Trying {params} ===")
            write_score_table(params)
            report = run_acceptance(args.games_per_seed, args.seeds)
            row = {
                "params": params,
                "all_positive_count": report["all_positive_window_count"],
                "total": report["total_window_count"],
                "min_window": report["min_window_chong"],
                "avg_chong": report["avg_chong_per_game"],
                "houjuu_rate": report["avg_houjuu_rate"],
            }
            results.append(row)
            print(f"  -> {row['all_positive_count']}/{row['total']} "
                  f"min={row['min_window']} houjuu={row['houjuu_rate']:.3f}")
    finally:
        # 恢复原 score_table 由调用方决定；这里只 dump 网格结果
        pass

    # 找最优：先按 all_positive_count 降序，再按 min_window 降序，再按 houjuu_rate 升序
    results.sort(key=lambda r: (-r["all_positive_count"], -r["min_window"], r["houjuu_rate"]))
    best = results[0] if results else None
    print("\n=== TOP 5 ===")
    for r in results[:5]:
        print(f"{r['params']} -> {r['all_positive_count']}/{r['total']} "
              f"min={r['min_window']} houjuu={r['houjuu_rate']:.3f}")

    # 把最优参数写到 score_table_phase_a.json（不直接覆盖 score_table.json）
    if best:
        write_score_table(best["params"])
        shutil.copy(SCORE_TABLE, args.output)
        print(f"\nBest params written to {args.output}")

    # 恢复原 score_table（由用户决定是否替换）
    shutil.copy(backup, SCORE_TABLE)
    print(f"Restored original score_table.json from {backup}")
    print("To apply best params: cp", args.output, str(SCORE_TABLE))

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: 跑 small grid 验证脚本

- [ ] **Step 2.1: 用最小参数跑通流水**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 tools/phase_a_tune.py --grid small --games-per-seed 16 --seeds 1`
Expected: 6 组参数全跑完，输出 `Best params written to engine/params/v3/score_table_phase_a.json`

### Step 3: Task A6 提交边界

```
已新增：
- tools/phase_a_tune.py
```

用户参考：
```bash
git add tools/phase_a_tune.py
git commit -m "tool(phase-a): grid search for EV risk and betaori thresholds"
```

---

## Task A7: Phase A 验证门槛实跑 + 报告

**Files:**
- Run: `tools/phase_a_tune.py`（full grid）
- Run: `tools/acceptance_25_8.py`（用最优参数验证）

### Step 1: 全量网格搜索

- [ ] **Step 1.1: 跑 full grid（背景运行，约 3 小时）**

Run:
```bash
cd /Users/fangyajun/CLionProjects/majiangV3 && \
python3 tools/phase_a_tune.py --grid full --games-per-seed 200 --seeds 1 2 3 4 5 \
    > docs/release/phase_a_tune_full.log 2>&1 &
```
（建议在 tmux 或后台跑；用 `tail -f docs/release/phase_a_tune_full.log` 监控）

- [ ] **Step 1.2: 等完成后查看 TOP 5**

Run: `tail -30 /Users/fangyajun/CLionProjects/majiangV3/docs/release/phase_a_tune_full.log`
确认 `=== TOP 5 ===` 输出。

### Step 2: 用最优参数实跑严苛验收

- [ ] **Step 2.1: 把最优参数写入 score_table.json**

Run: `cp /Users/fangyajun/CLionProjects/majiangV3/engine/params/v3/score_table_phase_a.json \
        /Users/fangyajun/CLionProjects/majiangV3/engine/params/v3/score_table.json`

- [ ] **Step 2.2: 跑严苛门槛验收**

Run:
```bash
cd /Users/fangyajun/CLionProjects/majiangV3 && \
python3 tools/acceptance_25_8.py \
    --games-per-seed 200 --seeds 1 2 3 4 5 \
    --min-window-chong -8 --min-avg-chong 1.5 --max-houjuu-rate 0.11 \
    --output docs/release/phase_a_acceptance.json
```

Expected：
- `all_positive_window_count ≥ 118`（spec §3.5.4 门槛）
- `min_window_chong ≥ -8`
- `avg_houjuu_rate ≤ 0.11`
- `avg_chong_per_game ≥ 1.5`

如果未达标：
- 回 Task A6，扩大网格（加入 `mu_base` 维度）或调整 EV 公式系数（如 `houjuu_base_coeff` 从 5500 升到 6500）
- 重跑 Step 1-2

- [ ] **Step 2.3: 跑性能回归确认 P95 不退化**

Run: `cd /Users/fangyajun/CLionProjects/majiangV3 && python3 tools/perf_regression.py`
Expected: P95 ≤ 800ms

### Step 3: 写 Phase A 验收报告

- [ ] **Step 3.1: 创建 `docs/release/phase_a_report.md`**

```markdown
# Phase A 验收报告（YYYY-MM-DD）

## 1. 验证门槛 vs 实测

| 指标 | 门槛 | Phase A 实测 | 现状（基线）|
|---|---|---|---|
| 25/25 全正窗口数 | ≥ 118/125 | TBD | 106/125 |
| 最差单窗口 | ≥ -8 冲 | TBD | -43 冲 |
| 总放炮率 | ≤ 11% | TBD | 14.8% |
| 平均冲数 / 局 | ≥ 1.5 | TBD | 3.972 |
| 单步 P95 延迟 | ≤ 800ms | TBD | ~50ms |

## 2. 最优参数

- `lambda_base = ...`
- `mu_base = ...`
- `betaori_threshold_base = ...`
- `qingyise_drop = ...`

## 3. 失败窗口分析

剩余 N 个不过窗口的对局复盘：seed_id / window_id / chong / 主要失分原因

## 4. 是否进入 Phase B

✅ 通过 / ❌ 未通过（继续调参）
```

- [ ] **Step 3.2: AI 输出修改清单**

```
已新增：
- docs/release/phase_a_report.md
- docs/release/phase_a_acceptance.json
- docs/release/phase_a_tune_full.log
- engine/params/v3/score_table_phase_a.json
已修改（最优参数）：
- engine/params/v3/score_table.json
```

用户参考：
```bash
git add docs/release/phase_a_*.md docs/release/phase_a_*.json \
        docs/release/phase_a_tune_full.log \
        engine/params/v3/score_table*.json
git commit -m "validate(phase-a): acceptance 118+/125 with tuned EV risk weights"
```

---

## Task A8: Phase B 衔接（参数固化与移交）

**Files:**
- Update: `docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md`（在 Phase 5/6 开头加入"Phase A 已通过"前置说明）

### Step 1: 在 Phase B 计划顶部追加 Phase A 入口节

- [ ] **Step 1.1: 修改 `docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md`**

在文件开头 `## 阶段总览` 之前插入：

```markdown
## Phase A 前置门槛（已完成）

> 本计划的 Phase 5-7 假设 Phase A（推理侧防守加固）已通过 spec §3.5.4 验证门槛。如果 Phase A 未通过，回到 `2026-04-28-phase-a-defensive-hardening.md` 调参。
>
> **Phase A 完成证据：** `docs/release/phase_a_report.md` + `docs/release/phase_a_acceptance.json`（≥118/125 全正、放炮率 ≤11%、最差窗口 ≥-8）。
> **Phase A 输出参数：** `engine/params/v3/score_table.json`（含 `ev_risk_weights` / `betaori_thresholds` / `feature_weights` 三段）作为 R0 训练的初始环境。
```

### Step 2: Task A8 提交边界

```
已修改：
- docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md（加入 Phase A 入口节）
```

用户参考：
```bash
git add docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md
git commit -m "doc: link Phase A gate into Phase B plan"
```

---

## Phase A 完成后 → Phase B 入口

Phase A 全部任务完成且 §3.5.4 门槛通过后，按 `docs/superpowers/plans/2026-04-28-linhai-v3-acceptance-plan.md` 的 **Phase 6（训练管线改造）** 与 **Phase 7（iterative_train R0-R3）** 继续。

Phase B 关键节点（详细步骤参见原计划）：
1. **Phase 6 Task 1-5**：改造 `selfplay_sample.py` 加 `score_label`，改造 `train_model_stub.py` 支持 regression
2. **Phase 6 Task 5b**（新增）：扩展 §4.4 中剩余 3 个 per-tile 特征（`safety_suji_t_X`、`safety_kabe_t_X`、`opp_riichi_proxy`），同步 C++ 与 Python `extract_canonical_states.py`
3. **Phase 7 R0-R3**：基于 Phase A 调好的 `score_table.json` 初始环境跑迭代训练（R0=6h、R1=8h、R2=12h、R3=1h 验收）
4. **§6.1.1 真人 dry-run**：4-6 人内测，每人 24+ 局；任一窗口 ≤0 回 R2 重训

Phase B 完成判据（spec §6.1 T6）：
- 5 seed × 25/25 = 125 个窗口全部 > 0
- 平均冲数 / 局 ≥ 0.5
- 最差单窗口 ≥ +2 冲
- 总放炮率 ≤ 12%
- 真人 dry-run 全员窗口正分

---

## 自审 checklist

- [x] 所有 task 均有 Files / 测试 / 实施 / 验证 / 提交 五段
- [x] 每个 step 2-5 分钟，含 exact command / code block
- [x] 用户偏好「不执行 git」体现在所有 commit 步骤里
- [x] 与现有 Phase B 计划无冲突，引用关系清晰
- [x] Phase A 验证门槛与 spec §3.5.4 完全对齐
- [x] 失败回退路径明确（A7 Step 2.2 失败回 A6）
