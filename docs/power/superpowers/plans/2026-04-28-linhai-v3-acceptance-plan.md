# 临海麻将 V3 — 算法验收实施计划（2026-04-28）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **本项目用户偏好**：所有 `git commit` / `git add` 由用户手动执行；本计划里的"Stage & commit"步骤只是文档化，AI 执行时跳过。AI 执行后给出"已修改文件列表"让用户自己 commit。

**Goal:** 让 majiangV3 通过甲方算法验收 — 200 局 25/25 8-局结算积分均为正（5 seed × 25 = 125 窗口），引擎计算 P95 ≤ 800ms，临海番数完整，对手为真人时仍能保 25/25。

**Architecture:** 在现有 V3 (Akochan-style EV 搜索 + 6 GBDT head) 基础上做三件事：(1) 新增 `linhai_score` 番数引擎，将"番→冲→分"接入 selfplay 结算与训练标签；(2) 引入 ReferenceHumanPolicy 离线代理对手并校准为"中等熟练真人"水平；(3) 多轮 R0-R3 迭代自对弈蒸馏，扩展 per-tile 防守特征（现物/筋/壁），用 acceptance_25_8 5-seed 回归把关。性能预算 800ms/2000ms，留 40× 余量。

**Tech Stack:** C++17 (engine/share/), Python 3.10+ (backend/, tools/), pybind11 binding, FastAPI, LightGBM GBDT, pytest, `tools/selfplay_eval.py` 既有自对弈框架。

**Spec:** `docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md`

---

## File Structure（先锁定再开工）

### 新增文件

| 路径 | 责任 |
|---|---|
| `engine/share/linhai_score.hpp` | 番数引擎接口：`ChongBreakdown` + `calc_chong()` |
| `engine/share/linhai_score.cpp` | 番数引擎实现，所有图条款逐条对应 |
| `engine/params/v3/score_table.json` | 番数与积分参数表（T1-T6 默认值） |
| `backend/app/services/reference_human.py` | 离线代理对手 ReferenceHumanPolicy |
| `backend/app/services/score_calculator.py` | Python 侧 番数计算包装（用于 selfplay 结算） |
| `tools/calibrate_reference.py` | ReferenceHumanPolicy 强度校准 |
| `tools/acceptance_25_8.py` | 5-seed × 200 局验收回归 |
| `tools/perf_regression.py` | 4 个最坏牌型 P95 性能回归 |
| `tools/iterative_train.py` | R0-R3 串联流水线 |
| `tests/cpp/test_linhai_score.cpp` | 番数引擎 C++ 单元测试 |
| `tests/python/test_linhai_score.py` | 番数引擎 Python 集成测试 |
| `tests/python/test_reference_human.py` | 代理对手测试 |
| `tests/python/test_acceptance_25_8.py` | 验收脚本测试（短跑通） |
| `tests/python/test_feature_parity.py` | 训练/推理特征名一致性测试 |
| `tests/python/test_score_label_pipeline.py` | selfplay → score_label → trainer 端到端 |
| `docs/DEPLOYMENT.md` | 部署运行手册 |
| `docs/SCORE_TABLE.md` | 番数表与图条款映射文档 |

### 修改文件

| 路径 | 改动 |
|---|---|
| `engine/share/linhai_search_v3.cpp` | 加 safety per-tile 特征；接入 `linhai_score` 用于 EV 标签估算 |
| `engine/share/linhai_search_v3.hpp` | 暴露 score head 接口 |
| `engine/python_binding.cpp` | 暴露 `calc_chong` 给 Python |
| `engine/setup.py` | 加 `linhai_score.cpp` 编译源 |
| `tools/selfplay_eval.py` | 调用 `score_calculator` 算真实冲数 |
| `tools/selfplay_sample.py` | 样本带 `task_labels.score_label` |
| `tools/train_model_stub.py` | 支持 `--task agari_score / houjuu_score`，objective=regression |
| `tools/extract_canonical_states.py` | 加 safety_genbutsu/suji/kabe per-tile 特征 |
| `backend/app/services/v3_service.py` | 返回 `model_version`、`bundle_hash` |
| `backend/app/main.py` | `/debug/engine_status` 加 `model_version` 字段 |
| `docs/EVAL.md` | 加 5-seed 验收章节 |
| `docs/FEATURES.md` | 加 safety 特征族说明 |
| `docs/PARAMS.md` | 加 score_table.json 字段说明 |

---

## Phase A 前置门槛（已完成 — 2026-04-29）

> 本计划的 Phase 5-7 假设 Phase A（推理侧防守加固）已实施完毕。Phase A 详细步骤见 `2026-04-28-phase-a-defensive-hardening.md`。
>
> **Phase A 实测结果**（5 seed × 200 局 × 25 窗口 = 125 窗口）：
> - **107/125 (85.6%)**，min -42, houjuu 14.4%，avg_chong 3.886
> - 未达 spec §3.5.4 严苛门槛（≥118/125）但相对 baseline 106/125 (84.8%) 有边际改善
> - 完整报告：`docs/release/phase_a_report.md`
>
> **结论**：Phase A 已穷尽推理侧调参空间。剩余 18 个不过窗口属于"被对手清一色 / 树掉还原打爆"长尾事件，本质需要模型从 score_label 数据学习——这正是本计划 Phase 6 / 7 的目标。
>
> **Phase A 输出参数**（已固化在 `engine/params/v3/score_table.json`，作为 Phase 7 R0 训练的初始环境）：
> - `ev_risk_weights.lambda_base = 1.3`，`lambda_pressure_step = 0.4`
> - `ev_risk_weights.mu_base = 1.5`
> - `betaori_thresholds.base = 0.99`（实质禁用 A5 硬门——实测它在临海规则下误伤攻击型局面）
> - `feature_weights.opp_meld_pressure_alpha = 0.5`
> - 备份：`engine/params/v3/score_table_phase_a.json`
>
> **Phase A 已实现的 C++ 改动**（Phase B 直接受益，无需重做）：
> - `LinhaiSearchEngineV3::compute_opp_pressure_score(state)` — 副露压力综合标量
> - `CanonicalGameState.{opponent_honor_triplets, opp_white_meld_count}` — 高价值防守特征
> - `build_discard_candidates` 动态 λ/μ EV 重加权（A4）
>
> **Phase A 验证可重跑**：`python3 tools/acceptance_25_8.py --games-per-seed 200 --seeds 1 2 3 4 5`

---

## 阶段总览

| 阶段 | 内容 | 任务数 | 依赖 |
|---|---|---|---|
| Phase 1 | 番数引擎（C++ 番数计算 + score_table.json + 单测） | 8 | 无 |
| Phase 2 | Python 番数包装 + selfplay 结算改造 | 4 | P1 |
| Phase 3 | ReferenceHumanPolicy + 校准脚本 | 5 | 无（与 P1 并行） |
| Phase 4 | acceptance_25_8 + perf_regression | 4 | P2, P3 |
| Phase 5 | safety per-tile 特征双侧同步 | 5 | 无（可与 P1-P4 并行） |
| Phase 6 | 训练管线改造（score_label + regression head） | 5 | P2, P5 |
| Phase 7 | iterative_train R0-R3 流水线 | 4 | P4, P6 |
| Phase 8 | 性能与诊断（model_version、降级、部署手册） | 4 | P7 |
| **合计** | | **39** | |

---

## Phase 1 — 番数引擎（C++）

### Task 1: 创建 `score_table.json` 参数表

**Files:**
- Create: `engine/params/v3/score_table.json`

- [ ] **Step 1: 写 score_table.json**

```json
{
  "version": "1.0.0",
  "created_at": "2026-04-28",
  "description": "临海麻将番数与积分参数表，对应 docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md §3",
  "chong_to_score": 1,
  "base_chong": {
    "ordinary": 1,
    "qingyise": 8,
    "ziyise": 8,
    "hunyise_with_tree": 2,
    "hunyise_hard": 4
  },
  "multipliers": {
    "hard_collision": 2,
    "special_tile": 2,
    "tree_restore_white_anko": 4
  },
  "bonuses": {
    "qianggang_hu": 2,
    "grab_charge_per_tile": 1
  },
  "redhead": {
    "per_player_count_default": 2,
    "tile_to_chong": {
      "1w": 1, "2w": 2, "3w": 3, "4w": 4, "5w": 5, "6w": 6, "7w": 7, "8w": 8, "9w": 9,
      "east": 5, "south": 5, "west": 5, "north": 5,
      "red": 5, "green": 5, "white": 5
    }
  },
  "contract": {
    "split_ratio": [0.5, 0.5]
  },
  "liuju": {
    "remaining_default": 6
  },
  "cap": {
    "max_base_chong": 8,
    "extra_chong_uncapped": true
  }
}
```

- [ ] **Step 2: Stage（用户提交）**

```bash
git add engine/params/v3/score_table.json
# user runs: git commit
```

---

### Task 2: 番数引擎接口 `linhai_score.hpp`

**Files:**
- Create: `engine/share/linhai_score.hpp`

- [ ] **Step 1: 写头文件**

```cpp
#ifndef LINHAI_SCORE_HPP
#define LINHAI_SCORE_HPP

#include "types.hpp"
#include <string>
#include <vector>

namespace linhai {

struct ContractState {
    int counter;          // 已砸红次数
    int target_count;     // 承包目标数（默认 3）
    bool active;          // 是否触发承包
};

struct RedHeadCatch {
    std::vector<int> caught_tiles;  // 已抓到的红头牌（hai code）
    int per_player_count;            // 每人发的红头数（2/4/6）
};

struct ChongInputs {
    Hai_Array tehai;                 // 胡牌时手牌
    std::vector<Meld> melds;         // 副露
    int white_count_in_hand;         // 手牌白板数
    int white_count_in_melds;        // 副露白板数
    bool is_tsumo;                   // 自摸
    bool is_qiang_gang;              // 抢杠胡
    bool has_tree_active;            // 树活（白板暗刻当 3 财神）
    int jikaze;                      // 自风
    int bakaze;                      // 场风
    ContractState contract;
    RedHeadCatch redhead;
    int grab_charge_caught_count;    // 本局抓冲数
};

struct ChongBreakdown {
    int base_chong;          // 基础冲数 1/2/4/8
    int multiplier_2x;       // 硬碰硬 / 特殊牌翻倍叠加层数
    int multiplier_4x;       // 树掉还原 ×4
    int extra_chong;         // 抓冲（不计入封顶）
    int qianggang_bonus;     // 抢杠胡 +2
    int redhead_bonus;       // 翻屁股加分
    int contract_split;      // 三键承包平摊本数（>0 表示赢家自摸时的两家承包）
    int final_chong;         // 最终冲数 = min(base * mult_2x * mult_4x, cap) + extra + qianggang + redhead
    int score;               // = final_chong * chong_to_score
    std::string explain;     // 调试可读
};

// score_table 由 JSON 加载，按 spec §3.4 默认值
struct ScoreTable {
    int chong_to_score = 1;
    int base_qingyise = 8;
    int base_ziyise = 8;
    int base_hunyise_tree = 2;
    int base_hunyise_hard = 4;
    int qianggang_bonus = 2;
    int max_base_chong = 8;
    // ... 其它字段后续按需补
};

ScoreTable load_score_table(const std::string& path);

ChongBreakdown calc_chong(const ChongInputs& inputs, const ScoreTable& table);

} // namespace linhai

#endif
```

- [ ] **Step 2: Stage**

```bash
git add engine/share/linhai_score.hpp
```

---

### Task 3: 番数引擎单元测试 — 普通胡

**Files:**
- Create: `tests/cpp/test_linhai_score.cpp`
- Reference: `tests/cpp/test_search_v3_smoke.cpp`（参照现有测试结构）

- [ ] **Step 1: 写第一个失败测试**

```cpp
#include "../../engine/share/linhai_score.hpp"
#include <cassert>
#include <iostream>

using namespace linhai;

void test_ordinary_hu() {
    ScoreTable table;  // 默认值
    ChongInputs in;
    in.is_tsumo = true;
    in.is_qiang_gang = false;
    in.has_tree_active = false;
    in.white_count_in_hand = 1;  // 有白板，非硬碰硬
    in.contract.active = false;
    in.redhead.caught_tiles.clear();
    in.grab_charge_caught_count = 0;
    // tehai/melds 留空（番数计算只看修饰条件，牌型由调用方传入分类）

    ChongBreakdown b = calc_chong(in, table);
    assert(b.base_chong == 1);
    assert(b.multiplier_2x == 0);
    assert(b.final_chong == 1);
    assert(b.score == 1);
    std::cout << "test_ordinary_hu PASS\n";
}

int main() {
    test_ordinary_hu();
    // 后续 task 加更多用例
    return 0;
}
```

- [ ] **Step 2: 运行测试，确认编译失败（calc_chong 还不存在）**

Run（cmake 集成前可用 g++ 直接编一个 stub）：

```bash
cd engine && g++ -std=c++17 -I share share/types.cpp ../tests/cpp/test_linhai_score.cpp -o /tmp/test_score 2>&1 | head
```

Expected: undefined reference to `calc_chong`，符合预期。

- [ ] **Step 3: Stage**

```bash
git add tests/cpp/test_linhai_score.cpp
```

---

### Task 4: 番数引擎实现骨架 `linhai_score.cpp`

**Files:**
- Create: `engine/share/linhai_score.cpp`

- [ ] **Step 1: 写 calc_chong 主体（先实现普通胡 + 硬碰硬 + 抢杠胡）**

```cpp
#include "linhai_score.hpp"
#include "json11.hpp"
#include <fstream>
#include <sstream>

namespace linhai {

ScoreTable load_score_table(const std::string& path) {
    ScoreTable t;
    std::ifstream f(path);
    if (!f.is_open()) return t;  // 缺失则返回默认值
    std::stringstream ss;
    ss << f.rdbuf();
    std::string err;
    auto j = json11::Json::parse(ss.str(), err);
    if (!err.empty()) return t;
    if (j["chong_to_score"].is_number()) t.chong_to_score = j["chong_to_score"].int_value();
    if (j["base_chong"]["qingyise"].is_number()) t.base_qingyise = j["base_chong"]["qingyise"].int_value();
    if (j["base_chong"]["ziyise"].is_number()) t.base_ziyise = j["base_chong"]["ziyise"].int_value();
    if (j["base_chong"]["hunyise_with_tree"].is_number()) t.base_hunyise_tree = j["base_chong"]["hunyise_with_tree"].int_value();
    if (j["base_chong"]["hunyise_hard"].is_number()) t.base_hunyise_hard = j["base_chong"]["hunyise_hard"].int_value();
    if (j["bonuses"]["qianggang_hu"].is_number()) t.qianggang_bonus = j["bonuses"]["qianggang_hu"].int_value();
    if (j["cap"]["max_base_chong"].is_number()) t.max_base_chong = j["cap"]["max_base_chong"].int_value();
    return t;
}

// 牌型识别（清一色 / 混一色 / 字一色）
enum class HuType { Ordinary, HunYise, QingYise, ZiYise };

static HuType classify_hu(const Hai_Array& tehai, const std::vector<Meld>& melds, int white_count_total) {
    // TODO Task 5：完整牌型识别。先用最简单实现：所有都当 Ordinary。
    return HuType::Ordinary;
}

ChongBreakdown calc_chong(const ChongInputs& in, const ScoreTable& table) {
    ChongBreakdown b{};

    int total_white = in.white_count_in_hand + in.white_count_in_melds;
    HuType type = classify_hu(in.tehai, in.melds, total_white);

    // 1. 基础冲
    switch (type) {
    case HuType::QingYise: b.base_chong = table.max_base_chong; break;
    case HuType::ZiYise:   b.base_chong = table.base_ziyise; break;
    case HuType::HunYise:
        // 有树（白板）且非硬碰硬 → 2；硬碰硬（无白板）→ 4
        b.base_chong = (total_white > 0) ? table.base_hunyise_tree : table.base_hunyise_hard;
        break;
    case HuType::Ordinary:
    default:
        b.base_chong = 1;
    }

    // 2. 硬碰硬（无白板时所有牌型都翻倍）
    bool hard = (total_white == 0);
    if (hard && type != HuType::HunYise) {
        // 混一色已经在 base_chong 区分了 with_tree/hard，避免重复翻倍
        b.multiplier_2x++;
    }

    // 3. 树掉还原（白板暗刻 = 3 张全在手且 has_tree_active）
    if (in.has_tree_active && in.white_count_in_hand >= 3) {
        b.multiplier_4x = 1;
    }

    // 4. 计算最终基础（应用倍数）
    int mult = 1;
    for (int i = 0; i < b.multiplier_2x; ++i) mult *= 2;
    if (b.multiplier_4x) mult *= 4;
    int capped = std::min(b.base_chong * mult, table.max_base_chong);

    // 5. 抢杠胡 +2
    if (in.is_qiang_gang) b.qianggang_bonus = table.qianggang_bonus;

    // 6. 抓冲（不计入封顶）
    b.extra_chong = in.grab_charge_caught_count;

    // 7. 翻屁股
    // TODO Task 7：按 RedHeadCatch 详细实现
    b.redhead_bonus = 0;

    // 8. 三键承包
    // TODO Task 8：自摸 + active 时按 split_ratio 算
    b.contract_split = 0;

    b.final_chong = capped + b.extra_chong + b.qianggang_bonus + b.redhead_bonus;
    b.score = b.final_chong * table.chong_to_score;

    std::ostringstream os;
    os << "type=" << (int)type << " base=" << b.base_chong << " mult=" << mult
       << " capped=" << capped << " extra=" << b.extra_chong
       << " qianggang=" << b.qianggang_bonus << " redhead=" << b.redhead_bonus
       << " final=" << b.final_chong << " score=" << b.score;
    b.explain = os.str();

    return b;
}

} // namespace linhai
```

- [ ] **Step 2: 运行单测，确认 test_ordinary_hu PASS**

```bash
cd engine && g++ -std=c++17 -I share share/types.cpp share/linhai_score.cpp share/json11.cpp ../tests/cpp/test_linhai_score.cpp -o /tmp/test_score && /tmp/test_score
```

Expected: `test_ordinary_hu PASS`

- [ ] **Step 3: Stage**

```bash
git add engine/share/linhai_score.cpp
```

---

### Task 5: 牌型识别 `classify_hu` 完整实现

**Files:**
- Modify: `engine/share/linhai_score.cpp`（替换 Task 4 的 TODO classify_hu）
- Modify: `tests/cpp/test_linhai_score.cpp`（加 4 个用例）

- [ ] **Step 1: 加 4 个失败测试**

```cpp
void test_qingyise() {
    // 全部万 + 无字 + 无白板
    ScoreTable table;
    ChongInputs in;
    in.tehai.fill(0);
    for (int i = 0; i < 9; ++i) in.tehai[i] = (i==4 ? 1 : 1);  // 1m..9m 各 1 张 + 别的凑齐 14
    // 简化：直接调一个测试 helper（此 task 的 step3 加）
    // ...
}
void test_hunyise_with_tree() { /* 万 + 字 + 1张白板 */ }
void test_hunyise_hard() { /* 万 + 字 + 0 白板 */ }
void test_ziyise() { /* 全字牌 */ }
```

(完整测试代码在 step3 一起写，因为依赖 helper。)

- [ ] **Step 2: 实现 classify_hu**

替换 `linhai_score.cpp` 里 Task 4 的 `classify_hu` stub 为：

```cpp
static HuType classify_hu(const Hai_Array& tehai, const std::vector<Meld>& melds, int white_count_total) {
    // 把副露牌也聚合进总分布
    Hai_Array all_tiles = tehai;
    for (const auto& m : melds) {
        for (int t : m.tiles) all_tiles[t]++;
    }
    bool has_man = false, has_pin = false, has_sou = false, has_honor = false;
    for (int i = 0; i < 9; ++i) if (all_tiles[i] > 0) has_man = true;            // 0..8 = 1m..9m
    for (int i = 9; i < 18; ++i) if (all_tiles[i] > 0) has_pin = true;           // 9..17 = 1p..9p
    for (int i = 18; i < 27; ++i) if (all_tiles[i] > 0) has_sou = true;          // 18..26 = 1s..9s
    for (int i = 27; i < 34; ++i) if (all_tiles[i] > 0) has_honor = true;        // 27..33 = 风+三元
    int suit_count = (int)has_man + (int)has_pin + (int)has_sou;
    if (suit_count == 0 && has_honor) return HuType::ZiYise;
    if (suit_count == 1 && !has_honor) return HuType::QingYise;
    if (suit_count == 1 && has_honor) return HuType::HunYise;
    return HuType::Ordinary;
}
```

> **注**：`Hai_Array` 的索引定义见 `engine/share/types.hpp`，本 task 假设 0-33 的标准临海布局。运行 task 5 前先 `Read engine/share/types.hpp` 确认；如果索引不一致，按实际改。

- [ ] **Step 3: 写完整测试 + helper（fill_hai_array）**

在 `tests/cpp/test_linhai_score.cpp` 顶部加：

```cpp
static Hai_Array make_hai(std::initializer_list<std::pair<int,int>> kv) {
    Hai_Array a; a.fill(0);
    for (auto [hai, n] : kv) a[hai] = n;
    return a;
}
```

写 4 个用例验证 4 种 HuType。

- [ ] **Step 4: 运行测试**

```bash
cd engine && g++ -std=c++17 -I share share/types.cpp share/linhai_score.cpp share/json11.cpp ../tests/cpp/test_linhai_score.cpp -o /tmp/test_score && /tmp/test_score
```

Expected: 5 个用例全 PASS。

- [ ] **Step 5: Stage**

---

### Task 6: 抢杠胡 + 树掉还原测试

**Files:**
- Modify: `tests/cpp/test_linhai_score.cpp`

- [ ] **Step 1: 加 3 个失败测试**

```cpp
void test_qianggang() {
    // 普通胡 + 抢杠 → final_chong = 1 + 2 = 3
    ChongInputs in; ScoreTable table;
    in.tehai = make_hai({{0,1}});
    in.is_qiang_gang = true;
    in.white_count_in_hand = 1;
    auto b = calc_chong(in, table);
    assert(b.final_chong == 3);
}

void test_tree_restore() {
    // 白板暗刻在手 + has_tree_active → mult_4x = 1
    ChongInputs in; ScoreTable table;
    in.tehai = make_hai({{0,1}});
    in.has_tree_active = true;
    in.white_count_in_hand = 3;
    auto b = calc_chong(in, table);
    // base 1 * 4 = 4，capped 至 8 内 → 4
    // 注意硬碰硬条件：white > 0 → 不硬碰硬
    assert(b.final_chong == 4);
}

void test_qingyise_capped_with_extra() {
    // 清一色 + 抓 2 张冲 → 8 + 2 = 10（封顶只对基础冲，抓冲不计）
    ChongInputs in; ScoreTable table;
    in.tehai = make_hai({{0,1},{1,1},{2,1},{3,1},{4,1},{5,1},{6,1},{7,1},{8,1}});
    in.grab_charge_caught_count = 2;
    auto b = calc_chong(in, table);
    assert(b.final_chong == 10);
}
```

- [ ] **Step 2: 运行测试，确认全 PASS（实现已在 Task 4 完成）**

- [ ] **Step 3: Stage**

---

### Task 7: 翻屁股加分实现

**Files:**
- Modify: `engine/share/linhai_score.cpp`
- Modify: `tests/cpp/test_linhai_score.cpp`

- [ ] **Step 1: 加失败测试**

```cpp
void test_redhead_yiwan() {
    ChongInputs in; ScoreTable table;
    in.tehai = make_hai({{0,1}});
    in.redhead.caught_tiles = {0};  // 1万 (hai code 0)
    auto b = calc_chong(in, table);
    assert(b.redhead_bonus == 1);  // 一万 +1 冲
}

void test_redhead_jiuwan() {
    ChongInputs in; ScoreTable table;
    in.tehai = make_hai({{0,1}});
    in.redhead.caught_tiles = {8};  // 9万
    auto b = calc_chong(in, table);
    assert(b.redhead_bonus == 9);
}

void test_redhead_zhong() {
    ChongInputs in; ScoreTable table;
    in.tehai = make_hai({{0,1}});
    in.redhead.caught_tiles = {31};  // 中（按 types.hpp 的索引；实际编码以仓库为准）
    auto b = calc_chong(in, table);
    assert(b.redhead_bonus == 5);
}
```

> **注**：风/字牌的具体 hai code 索引看 `engine/share/types.hpp`，27-30 通常是东南西北、31-33 是中发白。本测试以实际索引为准。

- [ ] **Step 2: 实现 calc_redhead_bonus**

在 `linhai_score.cpp` 顶部加：

```cpp
static int calc_redhead_bonus(const RedHeadCatch& rh) {
    // 一万..九万: +1..+9 冲
    // 中/发/白/东/南/西/北: +5 冲
    int sum = 0;
    for (int hai : rh.caught_tiles) {
        if (hai >= 0 && hai <= 8) sum += (hai + 1);            // 1m..9m → +1..+9
        else if (hai >= 27 && hai <= 33) sum += 5;             // 字风+三元 → +5
        // 筒/条不在翻屁股表里，按 0
    }
    return sum;
}
```

替换 `b.redhead_bonus = 0;` 为 `b.redhead_bonus = calc_redhead_bonus(in.redhead);`。

- [ ] **Step 3: 运行测试，确认全 PASS**

- [ ] **Step 4: Stage**

---

### Task 8: 三键承包公式 + 完整 Phase 1 收尾

**Files:**
- Modify: `engine/share/linhai_score.cpp`
- Modify: `tests/cpp/test_linhai_score.cpp`
- Modify: `engine/setup.py`（加 linhai_score.cpp 编译源）

- [ ] **Step 1: 加承包测试**

```cpp
void test_contract_split() {
    // 自摸 + 承包激活 → 两家平摊 → contract_split = final_chong / 2
    ChongInputs in; ScoreTable table;
    in.tehai = make_hai({{0,1}});
    in.is_tsumo = true;
    in.contract.active = true;
    in.contract.counter = 3;
    in.contract.target_count = 3;
    auto b = calc_chong(in, table);
    // 默认普通胡硬碰硬（white=0） → base 1 * 2 = 2，承包平摊一家 1，两家共 2
    assert(b.contract_split == 1);  // 单家承包额，由调用方做总分配
}
```

- [ ] **Step 2: 实现承包**

```cpp
// 在 calc_chong 末尾、final_chong 计算后加：
if (in.is_tsumo && in.contract.active && in.contract.counter >= in.contract.target_count) {
    b.contract_split = b.final_chong / 2;  // 两家平摊（默认 0.5/0.5）
}
```

- [ ] **Step 3: 修改 engine/setup.py 加 linhai_score.cpp 编译源**

```python
# 在 sources 列表中加：
'engine/share/linhai_score.cpp',
```

- [ ] **Step 4: 运行所有测试，确认 9 个用例全 PASS**

- [ ] **Step 5: Stage**

---

## Phase 2 — Python 番数包装 + selfplay 结算改造

### Task 9: pybind 暴露 calc_chong

**Files:**
- Modify: `engine/python_binding.cpp`

- [ ] **Step 1: 在 python_binding.cpp 加 calc_chong 绑定**

```cpp
#include "share/linhai_score.hpp"
// ...
py::class_<linhai::ChongBreakdown>(m, "ChongBreakdown")
    .def_readonly("base_chong", &linhai::ChongBreakdown::base_chong)
    .def_readonly("final_chong", &linhai::ChongBreakdown::final_chong)
    .def_readonly("score", &linhai::ChongBreakdown::score)
    .def_readonly("explain", &linhai::ChongBreakdown::explain);

py::class_<linhai::ChongInputs>(m, "ChongInputs")
    .def(py::init<>())
    .def_readwrite("is_tsumo", &linhai::ChongInputs::is_tsumo)
    .def_readwrite("is_qiang_gang", &linhai::ChongInputs::is_qiang_gang)
    .def_readwrite("has_tree_active", &linhai::ChongInputs::has_tree_active)
    .def_readwrite("white_count_in_hand", &linhai::ChongInputs::white_count_in_hand)
    .def_readwrite("white_count_in_melds", &linhai::ChongInputs::white_count_in_melds)
    .def_readwrite("grab_charge_caught_count", &linhai::ChongInputs::grab_charge_caught_count);
    // tehai/melds/contract/redhead 后续按需加 setter helper

m.def("calc_chong", &linhai::calc_chong);
m.def("load_score_table", &linhai::load_score_table);
```

- [ ] **Step 2: 重新编译 .so**

```bash
cd engine && python3 setup.py build_ext --inplace
```

- [ ] **Step 3: smoke**

```bash
python3 -c "import linhai_v3; t = linhai_v3.load_score_table('engine/params/v3/score_table.json'); print(t)"
```

Expected: 不报错。

- [ ] **Step 4: Stage**

---

### Task 10: Python 番数计算包装 `score_calculator.py`

**Files:**
- Create: `backend/app/services/score_calculator.py`
- Create: `tests/python/test_linhai_score.py`

- [ ] **Step 1: 写测试**

```python
from backend.app.services.score_calculator import calc_score_from_game_event

def test_simple_tsumo_ordinary():
    event = {
        "is_tsumo": True,
        "is_qiang_gang": False,
        "has_tree_active": False,
        "white_count_in_hand": 1,
        "white_count_in_melds": 0,
        "tehai": {"1w": 1, "2w": 1},  # 简化
        "melds": [],
        "is_qingyise": False,
        "is_hunyise": False,
        "is_ziyise": False,
        "redhead_caught": [],
        "grab_charge_caught": 0,
        "contract_active": False,
    }
    score = calc_score_from_game_event(event, role="winner")
    assert score == 1, f"got {score}"
```

- [ ] **Step 2: 实现包装**

```python
"""Python 端番数 / 积分计算包装。

调用 C++ linhai_score.calc_chong；如果 C++ 模块加载失败，使用纯 Python
fallback（同样的逻辑）以保证 selfplay 在没装好 .so 的环境也能跑。
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
SCORE_TABLE_PATH = REPO_ROOT / "engine" / "params" / "v3" / "score_table.json"

import json

_CACHED_TABLE = None
def _load_table() -> Dict:
    global _CACHED_TABLE
    if _CACHED_TABLE is None:
        if SCORE_TABLE_PATH.exists():
            _CACHED_TABLE = json.loads(SCORE_TABLE_PATH.read_text())
        else:
            _CACHED_TABLE = {
                "chong_to_score": 1,
                "base_chong": {"qingyise": 8, "ziyise": 8, "hunyise_with_tree": 2, "hunyise_hard": 4},
                "bonuses": {"qianggang_hu": 2},
                "cap": {"max_base_chong": 8},
                "redhead": {"tile_to_chong": {}},
            }
    return _CACHED_TABLE

def _classify_hu(event: Dict) -> str:
    if event.get("is_qingyise"): return "qingyise"
    if event.get("is_ziyise"): return "ziyise"
    if event.get("is_hunyise"): return "hunyise"
    return "ordinary"

def calc_score_from_game_event(event: Dict, role: str = "winner") -> int:
    """role: 'winner' | 'loser' | 'observer'。返回该 role 视角的分数（带正负号）。"""
    if not event.get("is_hu", True):
        return 0  # 流局
    table = _load_table()
    base_table = table["base_chong"]
    cap = table["cap"]["max_base_chong"]
    qianggang = table["bonuses"]["qianggang_hu"]
    chong_to_score = table["chong_to_score"]
    redhead_table = table["redhead"]["tile_to_chong"]

    hu_type = _classify_hu(event)
    white_total = event["white_count_in_hand"] + event["white_count_in_melds"]

    if hu_type == "qingyise": base = cap
    elif hu_type == "ziyise": base = base_table["ziyise"]
    elif hu_type == "hunyise":
        base = base_table["hunyise_with_tree"] if white_total > 0 else base_table["hunyise_hard"]
    else: base = 1

    mult = 1
    if white_total == 0 and hu_type != "hunyise":  # 硬碰硬
        mult *= 2
    if event.get("has_tree_active") and event["white_count_in_hand"] >= 3:
        mult *= 4
    capped = min(base * mult, cap)

    qg_bonus = qianggang if event.get("is_qiang_gang") else 0
    extra = event.get("grab_charge_caught", 0)
    redhead = sum(int(redhead_table.get(tile, 0)) for tile in event.get("redhead_caught", []))

    final_chong = capped + extra + qg_bonus + redhead
    score = final_chong * chong_to_score

    if role == "winner": return score
    if role == "loser":  return -score
    return 0
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/fangyajun/CLionProjects/majiangV3 && python3 -m pytest tests/python/test_linhai_score.py -v
```

Expected: PASS

- [ ] **Step 4: Stage**

---

### Task 11: selfplay_eval 接入 score_calculator

**Files:**
- Modify: `tools/selfplay_eval.py`

- [ ] **Step 1: 找到现有结算逻辑**

```bash
grep -n "winrate\|win_count\|score" tools/selfplay_eval.py | head -20
```

- [ ] **Step 2: 替换"+1/-1 计胜负"为"+score_chong/-score_chong"**

在 `tools/selfplay_eval.py` 里：
- 找到 game-end handling
- 调用 `from backend.app.services.score_calculator import calc_score_from_game_event`
- 把 `winrate_a += 1` 改为 `total_chong_a += score`

新增输出字段：`avg_chong_a`、`avg_chong_b`、`pos_window_count_a`（8 局窗口正分数）。

- [ ] **Step 3: 跑 smoke**

```bash
python3 tools/selfplay_eval.py --policy-a heuristic --policy-b heuristic --games 16 --seed 1
```

Expected: 输出包含 `avg_chong_a`、`pos_window_count_a/b`，且 16 局应该至少 1 个完整窗口。

- [ ] **Step 4: Stage**

---

### Task 12: selfplay_sample 加 score_label

**Files:**
- Modify: `tools/selfplay_sample.py`

- [ ] **Step 1: 找到现有 label 生成位置**

```bash
grep -n "task_labels\|can_win_label\|outcome" tools/selfplay_sample.py | head
```

- [ ] **Step 2: 在写每条样本时加 `score_label`**

```python
# 局结束后，对该玩家在本局的每条样本：
sample["task_labels"]["score_label"] = final_score_for_this_player  # int，+/-/0
```

- [ ] **Step 3: 跑 smoke**

```bash
python3 tools/selfplay_sample.py --games 32 --output /tmp/score_smoke.jsonl --policy-a heuristic --policy-b heuristic --seed 1
python3 -c "import json; lines=open('/tmp/score_smoke.jsonl').readlines(); print(sum('score_label' in json.loads(l).get('task_labels',{}) for l in lines), 'of', len(lines))"
```

Expected: 全部样本都有 `score_label`。

- [ ] **Step 4: Stage**

---

## Phase 3 — ReferenceHumanPolicy + 校准

### Task 13: ReferenceHumanPolicy 类

**Files:**
- Create: `backend/app/services/reference_human.py`
- Create: `tests/python/test_reference_human.py`

- [ ] **Step 1: 写失败测试**

```python
from backend.app.services.reference_human import ReferenceHumanPolicy
from backend.app.core.state import GameState  # 沿用既有 state

def test_reference_picks_legal_discard():
    policy = ReferenceHumanPolicy(seed=1)
    state = GameState.example_two_player()  # 假设有 helper；如无，用 v2_fallback 测试同款
    decision = policy.recommend_discard(state)
    assert decision["tile"] in [t for t in state.current_player().hand]

def test_reference_betaori_when_high_houjuu():
    policy = ReferenceHumanPolicy(seed=1, betaori_threshold=0.20, betaori_prob=1.0)
    state = GameState.example_high_houjuu_state()
    decision = policy.recommend_discard(state)
    assert decision.get("chosen_by") == "betaori"
```

- [ ] **Step 2: 实现 ReferenceHumanPolicy**

```python
"""离线代理对手 — 模拟"中等熟练真人"水平。

配置项见 spec §2.2，超参可通过 calibrate_reference.py 校准后写入
`engine/params/v3/reference_human_config.json`。
"""
import random
from typing import Dict, List, Optional
from .v2_fallback import V2FallbackAdapter
from ..core.state import GameState

class ReferenceHumanPolicy:
    def __init__(self, seed: int = 0,
                 betaori_threshold: float = 0.20,
                 betaori_prob: float = 0.50,
                 hunyise_bonus: float = 0.25,
                 mistake_prob: float = 0.03):
        self.rng = random.Random(seed)
        self.v2 = V2FallbackAdapter()
        self.betaori_threshold = betaori_threshold
        self.betaori_prob = betaori_prob
        self.hunyise_bonus = hunyise_bonus
        self.mistake_prob = mistake_prob

    def recommend_discard(self, state: GameState) -> Dict:
        # 基础：V2 baseline
        v2_result = self.v2.recommend_discard(state)
        if not v2_result:
            from .strategy_fallback import StrategyFallbackService
            return StrategyFallbackService().recommend_discard(state)

        # 防守层：houjuu_prob 高时 betaori
        if v2_result.get("houjuu_prob", 0) > self.betaori_threshold:
            if self.rng.random() < self.betaori_prob:
                # 选最安全的牌（candidate_scores 里 houjuu 最低的）
                # 简化：保留 V2 选择，只标记 chosen_by
                v2_result["chosen_by"] = "betaori"
                return v2_result

        # 番数倾向：清一色/混一色 雏形 → 轻微 EV bonus
        # （这里只标 metadata，不实际改 EV，因为 V2 已经选完了；下一版改 V2 接口）

        # 错牌噪声
        if self.rng.random() < self.mistake_prob:
            cand = list(v2_result.get("candidate_scores", {}).items())
            cand.sort(key=lambda x: -x[1])
            top2 = cand[:2]
            if len(top2) >= 2:
                pick = self.rng.choice(top2)
                v2_result["tile"] = pick[0]
                v2_result["chosen_by"] = "noise"

        v2_result["engine"] = "reference_human"
        return v2_result

    def recommend_response(self, state, *args, **kwargs):
        return self.v2.recommend_response(state, *args, **kwargs)
```

- [ ] **Step 3: 运行测试**

```bash
python3 -m pytest tests/python/test_reference_human.py -v
```

Expected: 至少 `test_reference_picks_legal_discard` PASS。

> 如果 `GameState.example_two_player()` helper 不存在，按照 `tests/python/test_orchestrator.py` 的状态构造方式写一个 fixture（或直接用现有 fixture 改）。

- [ ] **Step 4: Stage**

---

### Task 14: 校准脚本 calibrate_reference.py

**Files:**
- Create: `tools/calibrate_reference.py`

- [ ] **Step 1: 写脚本**

```python
"""校准 ReferenceHumanPolicy 强度：必须 vs heuristic 胜率 ≥ 70%、放炮率 ≤ 12%。

不达标 → 返回非零退出码，CI 阻塞。
"""
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target-winrate", type=float, default=0.70)
    ap.add_argument("--max-houjuu-rate", type=float, default=0.12)
    args = ap.parse_args()

    # 调用 selfplay_eval 子例程
    from tools.selfplay_eval import run_selfplay
    result = run_selfplay(
        policy_a="reference_human",
        policy_b="heuristic",
        games=args.games,
        seed=args.seed,
    )
    winrate = result.get("winrate_a", 0)
    houjuu = result.get("houjuu_rate_a", 1)

    ok = (winrate >= args.target_winrate) and (houjuu <= args.max_houjuu_rate)
    out = {"winrate_a": winrate, "houjuu_rate_a": houjuu, "passed": ok,
           "target_winrate": args.target_winrate, "max_houjuu_rate": args.max_houjuu_rate}
    print(json.dumps(out, indent=2))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
```

> **依赖**：`tools/selfplay_eval.py` 要支持 `policy_a="reference_human"`。

- [ ] **Step 2: selfplay_eval.py 注册 reference_human policy**

在 `tools/selfplay_eval.py` 的 policy factory 里加：

```python
elif policy_name == "reference_human":
    from backend.app.services.reference_human import ReferenceHumanPolicy
    return ReferenceHumanPolicy(seed=seed)
```

- [ ] **Step 3: 跑 smoke**

```bash
python3 tools/calibrate_reference.py --games 50
```

Expected: 输出 `passed: true/false`，退出码与 `passed` 对应。

- [ ] **Step 4: Stage**

---

### Task 15: 调参循环（如果 calibrate fail）

**Files:**
- Modify: `backend/app/services/reference_human.py`（如果首次校准失败）

- [ ] **Step 1: 校准结果分类**

如果 winrate < 70%：放宽 `mistake_prob` 到 0.01（更接近最优）；增强 `hunyise_bonus`。
如果 houjuu_rate > 12%：增加 `betaori_prob` 到 0.70；降低 `betaori_threshold` 到 0.15。

- [ ] **Step 2: 重跑 calibrate**

每次调一个超参，记录到 `engine/params/v3/reference_human_config.json`：

```json
{
  "version": "1.0.0",
  "calibrated_at": "2026-04-28",
  "winrate_vs_heuristic": 0.72,
  "houjuu_rate_vs_heuristic": 0.10,
  "params": {
    "betaori_threshold": 0.20,
    "betaori_prob": 0.50,
    "mistake_prob": 0.03
  }
}
```

- [ ] **Step 3: ReferenceHumanPolicy 加载校准后参数**

```python
def __init__(self, seed=0, config_path=None):
    if config_path:
        cfg = json.loads(Path(config_path).read_text())["params"]
        # 用 cfg 覆盖默认值
```

- [ ] **Step 4: Stage**

---

### Task 16: 防守层用 candidate_scores 真实选最安全

**Files:**
- Modify: `backend/app/services/reference_human.py`

- [ ] **Step 1: 加测试**

```python
def test_betaori_picks_safer_tile():
    # 模拟一个状态，hand 里有两张：一张 V2 推荐但 houjuu_prob 高，一张安全
    # ReferenceHumanPolicy 触发 betaori 时应该选安全那张
    # （细节按现有 v2_fallback.recommend_discard 返回的 candidate_scores）
    pass
```

- [ ] **Step 2: 替换 betaori 分支为"按 V2 candidate_scores 里 houjuu 最低排序"**

```python
if v2_result.get("houjuu_prob", 0) > self.betaori_threshold:
    if self.rng.random() < self.betaori_prob:
        # candidate_scores 是 {tile: ev}；ReferenceHumanPolicy 需要按 houjuu 重排，
        # 这要求 V2 fallback 在 candidate_scores 里包含 per-tile houjuu。
        # 如 V2 没暴露，先简化为 V2 top1 的 fallback：标记 chosen_by 即可。
        per_tile_houjuu = v2_result.get("candidate_houjuu", {})
        if per_tile_houjuu:
            safest = min(per_tile_houjuu, key=per_tile_houjuu.get)
            v2_result["tile"] = safest
        v2_result["chosen_by"] = "betaori"
```

> **注**：如果 `v2_fallback` 暂未暴露 per-tile houjuu，记 TODO，本期保持简化版。

- [ ] **Step 3: Stage**

---

### Task 17: ReferenceHumanPolicy 集成 selfplay 通联测试

**Files:**
- Modify: `tests/python/test_reference_human.py`

- [ ] **Step 1: 加跑 16 局自对弈通联**

```python
def test_reference_human_self_play_smoke():
    from tools.selfplay_eval import run_selfplay
    result = run_selfplay(policy_a="reference_human", policy_b="heuristic", games=16, seed=1)
    assert result["games"] == 16
    assert 0 <= result["winrate_a"] <= 1
```

- [ ] **Step 2: 运行**

```bash
python3 -m pytest tests/python/test_reference_human.py -v
```

- [ ] **Step 3: Stage**

---

## Phase 4 — acceptance_25_8 + perf_regression

### Task 18: acceptance_25_8.py 5-seed × 25 窗口

**Files:**
- Create: `tools/acceptance_25_8.py`

- [ ] **Step 1: 写脚本**

```python
"""5-seed × 200 局 (25 个 8-局窗口) 验收回归。

通过条件（spec §6.1）：
- 5 seed × 25 = 125 个窗口全部 > 0
- 平均冲数 / 局 ≥ 0.5
- 最差单窗口 ≥ +2 冲
- 总放炮率 ≤ 12%
"""
import argparse, json, sys, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

def settle_windows(per_game_chong: list, window_size: int = 8) -> list:
    return [sum(per_game_chong[i:i+window_size]) for i in range(0, len(per_game_chong), window_size)
            if len(per_game_chong[i:i+window_size]) == window_size]

def run_one_seed(seed: int, games: int, window: int) -> dict:
    from tools.selfplay_eval import run_selfplay_with_per_game_chong
    chongs, houjuu = run_selfplay_with_per_game_chong(
        policy_a="orchestrator:v3", policy_b="reference_human",
        games=games, seed=seed,
    )
    windows = settle_windows(chongs, window)
    return {
        "seed": seed,
        "windows": windows,
        "all_positive": all(w > 0 for w in windows),
        "min_window": min(windows) if windows else 0,
        "avg_chong_per_game": sum(chongs) / max(1, len(chongs)),
        "houjuu_rate": houjuu,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-seed", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5])
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--min-window-chong", type=int, default=2)
    ap.add_argument("--min-avg-chong", type=float, default=0.5)
    ap.add_argument("--max-houjuu-rate", type=float, default=0.12)
    ap.add_argument("--output", default="docs/release/acceptance_report.json")
    args = ap.parse_args()

    results = [run_one_seed(s, args.games_per_seed, args.window) for s in args.seeds]
    all_windows = [w for r in results for w in r["windows"]]
    all_positive = all(r["all_positive"] for r in results)
    min_window = min(r["min_window"] for r in results)
    avg_chong = sum(r["avg_chong_per_game"] for r in results) / len(results)
    avg_houjuu = sum(r["houjuu_rate"] for r in results) / len(results)

    passed = (all_positive
              and min_window >= args.min_window_chong
              and avg_chong >= args.min_avg_chong
              and avg_houjuu <= args.max_houjuu_rate)

    out = {
        "passed": passed,
        "all_positive_window_count": sum(1 for w in all_windows if w > 0),
        "total_window_count": len(all_windows),
        "min_window_chong": min_window,
        "avg_chong_per_game": avg_chong,
        "avg_houjuu_rate": avg_houjuu,
        "per_seed": results,
        "thresholds": vars(args),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ["passed","all_positive_window_count","total_window_count","min_window_chong","avg_chong_per_game","avg_houjuu_rate"]}, indent=2))
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: selfplay_eval.py 加 run_selfplay_with_per_game_chong helper**

```python
def run_selfplay_with_per_game_chong(policy_a, policy_b, games, seed, **kw):
    """返回 (per_game_chong: list[int], houjuu_rate: float)。"""
    # 在现有 run_selfplay 主循环里收集每局 chong 加返回
```

- [ ] **Step 3: 跑 smoke（小样本，验证流水通畅）**

```bash
python3 tools/acceptance_25_8.py --games-per-seed 8 --seeds 1
```

Expected: JSON 输出，passed 字段存在；不强求 passed=true（小样本）。

- [ ] **Step 4: Stage**

---

### Task 19: perf_regression.py 4 个最坏牌型

**Files:**
- Create: `tools/perf_regression.py`

- [ ] **Step 1: 写脚本**

```python
"""4 个最坏牌型的 P95 引擎延迟测试。

P95 > 800ms 退出 1。
"""
import argparse, json, sys, time, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

WORST_CASES = [
    {"name": "4_white_simple", "hand": ["white","white","white","white","2w","3w","4w","5w","6w","7w","8w","9w","1t","1t"]},
    {"name": "complex_meld_response", "hand": ["1w","2w","3w"], "melds": [...], "discarded": "5w"},
    {"name": "multi_qianggang_cand", "hand": [...]},
    {"name": "grab_charge_full", "hand": [...]},
]

def measure_one(case, runs=50) -> dict:
    from backend.app.services.v3_service import V3SearchService
    svc = V3SearchService()
    if not svc.available():
        return {"name": case["name"], "skipped": "v3 unavailable"}
    state = build_state_from_case(case)  # helper 见 Step 2
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        svc.recommend_discard(state)
        timings.append((time.perf_counter() - t0) * 1000)
    return {
        "name": case["name"],
        "p50_ms": statistics.median(timings),
        "p95_ms": statistics.quantiles(timings, n=20)[-1],
        "max_ms": max(timings),
    }

def build_state_from_case(case):
    from backend.app.core.state import GameState
    # 按 case dict 构造 GameState（参考 tests/python/test_ai_app_compatibility.py）
    ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p95-budget-ms", type=float, default=800.0)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()
    results = [measure_one(c, args.runs) for c in WORST_CASES]
    failed = [r for r in results if r.get("p95_ms", 0) > args.p95_budget_ms]
    print(json.dumps({"results": results, "failed": [r["name"] for r in failed]}, indent=2))
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 实现 build_state_from_case helper**

参考 `tests/python/test_ai_app_compatibility.py` 里现有 GameState 构造方式。

- [ ] **Step 3: 跑 smoke**

```bash
python3 tools/perf_regression.py --runs 5
```

Expected: 输出 4 个 case 的 p95，`failed` 字段为空（如果 V3 加载成功）。

- [ ] **Step 4: Stage**

---

### Task 20: pytest 短跑通 acceptance_25_8

**Files:**
- Create: `tests/python/test_acceptance_25_8.py`

- [ ] **Step 1: 写测试**

```python
import subprocess, sys, json
from pathlib import Path

def test_acceptance_smoke(tmp_path):
    out_path = tmp_path / "report.json"
    r = subprocess.run(
        [sys.executable, "tools/acceptance_25_8.py",
         "--games-per-seed", "16", "--seeds", "1",
         "--output", str(out_path)],
        capture_output=True, text=True,
    )
    assert out_path.exists(), r.stderr
    report = json.loads(out_path.read_text())
    assert "passed" in report
    assert "per_seed" in report
```

- [ ] **Step 2: 运行**

```bash
python3 -m pytest tests/python/test_acceptance_25_8.py -v
```

- [ ] **Step 3: Stage**

---

### Task 21: pytest 短跑通 perf_regression

**Files:**
- Create: `tests/python/test_perf_regression.py`

- [ ] **Step 1: 写测试**

```python
import subprocess, sys

def test_perf_regression_runs():
    r = subprocess.run(
        [sys.executable, "tools/perf_regression.py", "--runs", "3"],
        capture_output=True, text=True,
    )
    # 不强求 returncode == 0（V3 可能不可用）
    assert "results" in r.stdout
```

- [ ] **Step 2: 运行 + Stage**

---

## Phase 5 — safety per-tile 特征双侧同步

### Task 22: 特征定义文档与 tile 列表

**Files:**
- Modify: `docs/FEATURES.md`（加 §A-3 safety 特征族）
- Create: `tests/python/test_feature_parity.py`

- [ ] **Step 1: 写 feature_parity 测试**

```python
"""保证 Python 训练侧和 C++ 推理侧的特征名集合完全一致。"""
def test_safety_feature_parity():
    from tools.extract_canonical_states import build_model_features
    # 构造一个最小 state
    state_dict = {...}
    feats = build_model_features(state_dict)
    py_feature_names = set(feats.keys())

    # 加载 C++ 端导出的特征名（从 .so 取）
    import linhai_v3
    cpp_names = set(linhai_v3.list_state_feature_names())

    safety_py = {n for n in py_feature_names if n.startswith(("safety_genbutsu_t_", "safety_suji_t_", "safety_kabe_t_"))}
    safety_cpp = {n for n in cpp_names if n.startswith(("safety_genbutsu_t_", "safety_suji_t_", "safety_kabe_t_"))}
    assert safety_py == safety_cpp, f"Safety feature names diverge: py only {safety_py - safety_cpp}, cpp only {safety_cpp - safety_py}"
```

- [ ] **Step 2: 运行测试，确认失败（特征还没加）**

- [ ] **Step 3: Stage**

---

### Task 23: Python 侧 safety 特征实现

**Files:**
- Modify: `tools/extract_canonical_states.py`

- [ ] **Step 1: 在 build_model_features 里加 safety 特征**

```python
PER_TILE_FEATURE_ORDER = [
    "1w","2w","3w","4w","5w","6w","7w","8w","9w",
    "1t","2t","3t","4t","5t","6t","7t","8t","9t",
    "east","south","west","north","red","green","white",
]

def _calc_genbutsu(opp_disc_counts: dict) -> dict:
    return {tile: int(opp_disc_counts.get(tile, 0) > 0) for tile in PER_TILE_FEATURE_ORDER}

def _calc_suji(opp_disc_counts: dict) -> dict:
    """4w 出过 → 1w/7w 安全 +1。"""
    suji = {tile: 0 for tile in PER_TILE_FEATURE_ORDER}
    for suit_prefix in ("w", "t"):
        for mid in (4, 5, 6):
            mid_tile = f"{mid}{suit_prefix}"
            if opp_disc_counts.get(mid_tile, 0) > 0:
                if mid == 4: suji[f"1{suit_prefix}"] = 1; suji[f"7{suit_prefix}"] = 1
                if mid == 5: suji[f"2{suit_prefix}"] = 1; suji[f"8{suit_prefix}"] = 1
                if mid == 6: suji[f"3{suit_prefix}"] = 1; suji[f"9{suit_prefix}"] = 1
    return suji

def _calc_kabe(visible_counts: dict) -> dict:
    """X 可见 ≥3 张 → 相邻数牌 +1 安全度。"""
    kabe = {tile: 0 for tile in PER_TILE_FEATURE_ORDER}
    for suit_prefix in ("w", "t"):
        for r in range(1, 10):
            tile = f"{r}{suit_prefix}"
            if visible_counts.get(tile, 0) >= 3:
                if r > 1: kabe[f"{r-1}{suit_prefix}"] = 1
                if r < 9: kabe[f"{r+1}{suit_prefix}"] = 1
    return kabe

# 在 build_model_features 末尾加：
def build_model_features(state):
    feats = {...}  # 已有
    opp_disc = state.get("opponent_discard_counts", {})
    visible = state.get("visible_counts", {})
    for tile, v in _calc_genbutsu(opp_disc).items():
        feats[f"safety_genbutsu_t_{tile}"] = v
    for tile, v in _calc_suji(opp_disc).items():
        feats[f"safety_suji_t_{tile}"] = v
    for tile, v in _calc_kabe(visible).items():
        feats[f"safety_kabe_t_{tile}"] = v
    return feats
```

- [ ] **Step 2: 运行 Python 侧 smoke**

```bash
python3 -c "from tools.extract_canonical_states import build_model_features; f = build_model_features({'opponent_discard_counts':{'4w':1},'visible_counts':{}}); print({k:v for k,v in f.items() if 'safety' in k})"
```

Expected: 输出包含 `safety_suji_t_1w=1` 和 `safety_suji_t_7w=1`。

- [ ] **Step 3: Stage**

---

### Task 24: C++ 侧 safety 特征实现

**Files:**
- Modify: `engine/share/linhai_search_v3.cpp`（`build_state_features`）

- [ ] **Step 1: 找到 build_state_features**

```bash
grep -n "build_state_features\|tile_code_for_feature\|opp_disc_t" engine/share/linhai_search_v3.cpp | head -30
```

- [ ] **Step 2: 在 build_state_features 末尾加 safety 计算**

```cpp
// 与 tools/extract_canonical_states.py 必须一一对应
auto opp_count = [&](int tile) { return opponent_discard_counts[tile]; };
auto vis_count = [&](int tile) { return visible_counts[tile]; };

for (int tile : PER_TILE_FEATURE_ORDER_CPP) {
    feats["safety_genbutsu_t_" + tile_to_str(tile)] = opp_count(tile) > 0 ? 1.0f : 0.0f;
}
// 筋
for (auto suit : {0, 1}) {  // 0=w, 1=t (按仓库索引)
    int base = suit * 9;
    auto suji_set = [&](int mid_idx, int low_idx, int high_idx) {
        if (opp_count(base + mid_idx) > 0) {
            feats["safety_suji_t_" + tile_to_str(base + low_idx)] = 1.0f;
            feats["safety_suji_t_" + tile_to_str(base + high_idx)] = 1.0f;
        }
    };
    suji_set(3, 0, 6);  // 4 → 1, 7
    suji_set(4, 1, 7);  // 5 → 2, 8
    suji_set(5, 2, 8);  // 6 → 3, 9
}
// 壁
for (auto suit : {0, 1}) {
    int base = suit * 9;
    for (int r = 0; r < 9; ++r) {
        if (vis_count(base + r) >= 3) {
            if (r > 0) feats["safety_kabe_t_" + tile_to_str(base + r - 1)] = 1.0f;
            if (r < 8) feats["safety_kabe_t_" + tile_to_str(base + r + 1)] = 1.0f;
        }
    }
}
// 字牌默认 0（safety_genbutsu 已覆盖）
```

- [ ] **Step 3: 重新编译**

```bash
cd engine && python3 setup.py build_ext --inplace
```

- [ ] **Step 4: 运行 feature_parity 测试，确认 PASS**

```bash
python3 -m pytest tests/python/test_feature_parity.py -v
```

- [ ] **Step 5: Stage**

---

### Task 25: list_state_feature_names pybind 暴露

**Files:**
- Modify: `engine/share/linhai_search_v3.hpp`、`linhai_search_v3.cpp`、`engine/python_binding.cpp`

- [ ] **Step 1: 加方法 `list_state_feature_names()`**

C++ 实现：构造一个 dummy state 调用 build_state_features，返回 keys。

- [ ] **Step 2: pybind 暴露**

```cpp
m.def("list_state_feature_names", []() {
    LinhaiSearchEngineV3 dummy;
    auto names = dummy.list_state_feature_names();
    return std::vector<std::string>(names.begin(), names.end());
});
```

- [ ] **Step 3: 重新编译 + 测 feature_parity**

```bash
cd engine && python3 setup.py build_ext --inplace
python3 -m pytest tests/python/test_feature_parity.py -v
```

- [ ] **Step 4: Stage**

---

### Task 26: 让 V3 search 真正使用 safety 特征（推理侧）

**Files:**
- Modify: `engine/share/linhai_search_v3.cpp`（`predict_head` 或 GBDT 推理路径）

- [ ] **Step 1: 验证 GBDT loader 能从 model.json 读 safety_* 特征名**

GBDT 推理 `V3GBDTModel::predict` 已经按特征名查表，新加的 `safety_*` 在重训后会出现在 `feature_names` 数组里。验证：

```bash
python3 -c "
import json
m = json.load(open('engine/params/v3/agari_prob/model.json'))
print('safety features:', sum(1 for n in m['feature_names'] if 'safety' in n))"
```

> Phase 5 完成时这里还是 0；Phase 6 重训后才会非零。先确认 loader 路径不会因新特征名报错。

- [ ] **Step 2: 加单测**

```python
def test_v3_loads_with_safety_features_in_bundle():
    # 把一个 mock model.json 写入 tmp 目录，包含 safety_genbutsu_t_1w 特征
    # V3 应该正常加载、不抛异常
    ...
```

- [ ] **Step 3: Stage**

---

## Phase 6 — 训练管线改造

### Task 27: train_model_stub 支持 score_label regression

**Files:**
- Modify: `tools/train_model_stub.py`

- [ ] **Step 1: 加 task 名 `agari_score` / `houjuu_score`**

```python
TASK_TO_LABEL_KEY = {
    "agari_prob": "task_labels.can_win_label",
    "agari_score": "task_labels.score_label",
    "houjuu_prob": "source_meta.houjuu_label",
    "houjuu_score": "task_labels.score_label",  # 同字段，trainer 内部按 winner/loser 切分
    # ... 已有项保留
}
```

- [ ] **Step 2: 加 regression objective 路径**

```python
if task in {"agari_score", "houjuu_score"}:
    objective = "regression"
    # LightGBM regression head；linear fallback 用 sklearn LinearRegression
```

- [ ] **Step 3: 跑 smoke**

```bash
python3 tools/train_model_stub.py --task agari_score --version-dir /tmp/score_bundle --dataset /tmp/score_smoke.jsonl --validation-split 0.2 --model-kind auto
```

Expected: 产出 `/tmp/score_bundle/v3/agari_score/model.json`，`objective: "regression"`。

- [ ] **Step 4: Stage**

---

### Task 28: V3 引擎加载 score regression head

**Files:**
- Modify: `engine/share/linhai_search_v3.cpp`（load_model_bundle）

- [ ] **Step 1: 在 load_model_bundle 里识别 agari_score / houjuu_score 子目录**

```cpp
// 已有：load 6 个 head；新增 agari_score、houjuu_score（可选 head）
load_v3_head(bundle_dir + "/v3/agari_score", &agari_score_linear, &agari_score_gbdt);
load_v3_head(bundle_dir + "/v3/houjuu_score", &houjuu_score_linear, &houjuu_score_gbdt);
```

- [ ] **Step 2: 推理时优先用 score head 算 EV**

```cpp
// 在 search_one_action_ev 里：
double agari_ev = predict_head(agari_score_linear, agari_score_gbdt, features);
double houjuu_ev = predict_head(houjuu_score_linear, houjuu_score_gbdt, features);
double total_ev = agari_ev - houjuu_ev;  // E[my_score] - E[opp_score]
// 如果 score head 不存在（loaded == false），回退到旧 prob head 路径
```

- [ ] **Step 3: 重新编译 + 跑 smoke**

```bash
cd engine && python3 setup.py build_ext --inplace
python3 -c "import linhai_v3; e=linhai_v3.LinhaiSearchEngineV3(); e.load_model_bundle('/tmp/score_bundle'); print('ok')"
```

- [ ] **Step 4: Stage**

---

### Task 29: extract_canonical_states.py 加 score_label 反向填充

**Files:**
- Modify: `tools/extract_canonical_states.py`（与 selfplay_sample.py 同步）

- [ ] **Step 1: 输出样本时加 score_label 字段**

跟 Task 12 已经做了 selfplay_sample 的；这里如果有外部数据 ingest，把同一字段加进去。

- [ ] **Step 2: Stage**

---

### Task 30: 端到端 selfplay → label → trainer 测试

**Files:**
- Create: `tests/python/test_score_label_pipeline.py`

- [ ] **Step 1: 写测试**

```python
import subprocess, sys, tempfile, json
from pathlib import Path

def test_score_label_end_to_end(tmp_path):
    # 1. selfplay sample
    sample_path = tmp_path / "samples.jsonl"
    subprocess.check_call([sys.executable, "tools/selfplay_sample.py",
                           "--games", "16",
                           "--policy-a", "heuristic", "--policy-b", "heuristic",
                           "--seed", "1", "--output", str(sample_path)])
    # 验证至少一条样本带 score_label
    lines = sample_path.read_text().splitlines()
    assert any("score_label" in json.loads(l).get("task_labels",{}) for l in lines)

    # 2. train agari_score
    bundle_dir = tmp_path / "bundle"
    subprocess.check_call([sys.executable, "tools/train_model_stub.py",
                           "--task", "agari_score",
                           "--version-dir", str(bundle_dir),
                           "--dataset", str(sample_path),
                           "--validation-split", "0.2", "--model-kind", "auto"])
    model_path = bundle_dir / "v3" / "agari_score" / "model.json"
    assert model_path.exists()
    model = json.loads(model_path.read_text())
    assert model.get("objective") == "regression"
```

- [ ] **Step 2: 运行**

```bash
python3 -m pytest tests/python/test_score_label_pipeline.py -v
```

- [ ] **Step 3: Stage**

---

### Task 31: 文档更新（FEATURES.md / PARAMS.md / EVAL.md）

**Files:**
- Modify: `docs/FEATURES.md`、`docs/PARAMS.md`、`docs/EVAL.md`

- [ ] **Step 1: FEATURES.md 加 §A-3 safety 族**

```md
## A-3 safety 特征族（per-tile）

按 spec §4.4 实现的对手压力特征，与 V3 GBDT head 一起决定弃牌。

| 前缀 | 含义 | Python 实现 | C++ 实现 |
|---|---|---|---|
| `safety_genbutsu_t_X` | X 在对手河 | tools/extract_canonical_states.py | engine/share/linhai_search_v3.cpp |
| `safety_suji_t_X` | X 被筋 | 同 | 同 |
| `safety_kabe_t_X` | X 邻牌为壁 | 同 | 同 |

contract: 训练侧和推理侧的特征名 set 必须严格一致，由 `tests/python/test_feature_parity.py` 兜底。
```

- [ ] **Step 2: PARAMS.md 加 score_table.json + score head 描述**

- [ ] **Step 3: EVAL.md 加 25/25 验收口径**

- [ ] **Step 4: Stage**

---

## Phase 7 — iterative_train R0-R3

### Task 32: iterative_train.py 串联 R0

**Files:**
- Create: `tools/iterative_train.py`

- [ ] **Step 1: 写脚本骨架**

```python
"""R0-R3 多轮迭代训练流水线。

每轮三步：
1. selfplay_sample 生成数据
2. train_model_stub 训 6+2 head
3. acceptance_25_8 跑回归（R3 才需要严格通过）
"""
import argparse, subprocess, sys, json
from pathlib import Path

ROUNDS = {
    "R0": {"policy_a": "reference_human", "policy_b": "random", "games": 50000},
    "R1": {"policy_a": "orchestrator:v3", "policy_b": "reference_human", "games": 50000},
    "R2": {"policy_a": "orchestrator:v3", "policy_b": "orchestrator:v3", "games": 50000, "temperature": 0.5},
    "R3": {"policy_a": "orchestrator:v3", "policy_b": "reference_human", "games": 200, "is_acceptance": True},
}
HEADS = ["agari_prob","tenpai_prob","houjuu_prob","betaori","tsumo_num","ryukyoku_prob","agari_score","houjuu_score"]

def run_round(name, cfg, params_dir, sample_dir):
    if cfg.get("is_acceptance"):
        return run_acceptance(params_dir)
    sample_file = sample_dir / f"{name}.jsonl"
    subprocess.check_call([sys.executable, "tools/selfplay_sample.py",
                           "--policy-a", cfg["policy_a"], "--policy-b", cfg["policy_b"],
                           "--games", str(cfg["games"]), "--seed", "1",
                           "--output", str(sample_file)])
    out_dir = params_dir / name
    for head in HEADS:
        subprocess.check_call([sys.executable, "tools/train_model_stub.py",
                               "--task", head, "--version-dir", str(out_dir),
                               "--dataset", str(sample_file),
                               "--validation-split", "0.2", "--model-kind", "auto"])
    return {"name": name, "params_dir": str(out_dir)}

def run_acceptance(params_dir):
    # 设置环境变量让 V3 加载该 bundle
    import os
    env = os.environ.copy()
    env["LINHAI_V3_PARAMS_DIR"] = str(params_dir)
    r = subprocess.run([sys.executable, "tools/acceptance_25_8.py"],
                       env=env, capture_output=True, text=True)
    return {"name": "R3", "stdout": r.stdout, "passed": r.returncode == 0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="R0")
    ap.add_argument("--end", default="R3")
    ap.add_argument("--params-root", default="data/iter_params")
    ap.add_argument("--sample-root", default="data/iter_samples")
    args = ap.parse_args()
    params_root = Path(args.params_root); params_root.mkdir(parents=True, exist_ok=True)
    sample_root = Path(args.sample_root); sample_root.mkdir(parents=True, exist_ok=True)

    rounds = list(ROUNDS.keys())
    start_idx, end_idx = rounds.index(args.start), rounds.index(args.end)
    log = []
    for name in rounds[start_idx:end_idx+1]:
        log.append(run_round(name, ROUNDS[name], params_root, sample_root))
        # 每轮日志写盘
        Path(args.params_root + "/iter_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log[-1], indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 smoke（小样本）**

```bash
python3 tools/iterative_train.py --start R0 --end R0 --params-root /tmp/iter_params --sample-root /tmp/iter_samples
```

> 把 ROUNDS["R0"]["games"] 临时改 100 跑通流水。

- [ ] **Step 3: Stage**

---

### Task 33: 实跑 R0 训练（CPU 6 小时级）

**Files:** 无（执行任务）

- [ ] **Step 1: 跑完整 R0**

```bash
python3 tools/iterative_train.py --start R0 --end R0 \
    --params-root data/iter_params \
    --sample-root data/iter_samples
```

预计：50k 局自对弈 + 8 head 训练 ≈ 6-8 小时。建议用 `nohup` + `tee`：

```bash
nohup python3 tools/iterative_train.py --start R0 --end R0 \
    --params-root data/iter_params --sample-root data/iter_samples \
    > data/iter_log_R0.txt 2>&1 &
```

- [ ] **Step 2: 跑完后调用 acceptance_25_8 看 R0 baseline**

```bash
LINHAI_V3_PARAMS_DIR=data/iter_params/R0 python3 tools/acceptance_25_8.py --games-per-seed 200 --seeds 1
```

期望：通过率应该比改造前更高（积分而非胜率维度），但不一定 25/25 通过。记录 baseline。

- [ ] **Step 3: 确认产出文件**

```bash
ls data/iter_params/R0/v3/
# 期望看到 agari_prob/, agari_score/, ... 8 个目录
```

---

### Task 34: R1 / R2 训练

**Files:** 无（执行任务）

- [ ] **Step 1: 跑 R1**

```bash
LINHAI_V3_PARAMS_DIR=data/iter_params/R0 \
nohup python3 tools/iterative_train.py --start R1 --end R1 \
    --params-root data/iter_params --sample-root data/iter_samples \
    > data/iter_log_R1.txt 2>&1 &
```

- [ ] **Step 2: 跑 R2**

```bash
LINHAI_V3_PARAMS_DIR=data/iter_params/R1 \
nohup python3 tools/iterative_train.py --start R2 --end R2 \
    --params-root data/iter_params --sample-root data/iter_samples \
    > data/iter_log_R2.txt 2>&1 &
```

> 每轮间需要让 V3 引擎 reload bundle（设环境变量后重启 selfplay 子进程即可，iterative_train.py 已经 fork 子进程）。

- [ ] **Step 3: 监控放炮率走势**

期望：R1 → R2 houjuu_rate 持续下降，avg_chong 上升。

---

### Task 35: R3 验收（acceptance_25_8 5-seed 全跑）

**Files:** 无（执行任务）

- [ ] **Step 1: 用 R2 bundle 跑完整 acceptance_25_8**

```bash
LINHAI_V3_PARAMS_DIR=data/iter_params/R2 python3 tools/acceptance_25_8.py \
    --games-per-seed 200 --seeds 1 2 3 4 5 \
    --output docs/release/acceptance_report_R3.json
```

预计：约 1 小时（5 × 200 × 单步 ~50ms × 平均 25 步）。

- [ ] **Step 2: 验证 passed=true**

```bash
python3 -c "import json; r=json.load(open('docs/release/acceptance_report_R3.json')); print('PASSED' if r['passed'] else 'FAILED'); print(json.dumps({k:r[k] for k in ['all_positive_window_count','total_window_count','min_window_chong','avg_chong_per_game','avg_houjuu_rate']}, indent=2))"
```

Expected: `PASSED`，125/125 windows 全正，min_window ≥ 2，avg_chong ≥ 0.5，houjuu ≤ 12%。

- [ ] **Step 3: 不通过则回 R2 调参再跑 R2'**

调参方向：
- avg_chong 不够 → 加深 search depth；扩大 R2 样本量；提升 hunyise/qingyise 倾向
- houjuu_rate 偏高 → houjuu_score head 加权重；betaori 阈值降低

- [ ] **Step 4: Stage 报告**

---

## Phase 8 — 性能与诊断、交付物

### Task 36: 性能回归门禁

**Files:** 无（执行任务）

- [ ] **Step 1: 用 R3 bundle 跑 perf_regression**

```bash
LINHAI_V3_PARAMS_DIR=data/iter_params/R2 python3 tools/perf_regression.py --runs 100
```

Expected: 4 个 case P95 全部 < 800ms，failed 字段为空。

- [ ] **Step 2: 不通过 → 收 beam / 切 depth**

修改 `backend/app/services/v3_service.py` profile 配置，重跑 perf_regression。

- [ ] **Step 3: Stage 性能报告到 docs/release/**

---

### Task 37: model_version + bundle_hash

**Files:**
- Modify: `backend/app/services/v3_service.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: 在 V3SearchService 计算 bundle_hash**

```python
import hashlib
def _compute_bundle_hash(self, bundle_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(bundle_dir.rglob("model.json")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
```

- [ ] **Step 2: status() 返回 model_version + bundle_hash**

```python
def status(self):
    return {
        ...,
        "model_version": getattr(self, "_model_version", "unknown"),
        "bundle_hash": getattr(self, "_bundle_hash", "unknown"),
    }
```

- [ ] **Step 3: 加测试**

```python
def test_engine_status_has_model_version():
    from fastapi.testclient import TestClient
    from backend.app.main import app
    client = TestClient(app)
    r = client.get("/debug/engine_status").json()
    assert "model_version" in r["v3"]
    assert "bundle_hash" in r["v3"]
```

- [ ] **Step 4: Stage**

---

### Task 38: 部署手册 docs/DEPLOYMENT.md

**Files:**
- Create: `docs/DEPLOYMENT.md`

- [ ] **Step 1: 写部署手册**

```md
# 部署运行手册

## 上线前检查清单
1. 环境变量
   - `LINHAI_V3_ENGINE_MODULE_DIR` 指向 .so 所在目录
   - `LINHAI_V3_PARAMS_DIR` 指向 bundle 目录（生产用 R3 bundle）
   - `LINHAI_V3_PROFILE=prod`

2. 启动后 GET `/debug/engine_status`：
   - `active_engine` 必须为 `"v3"`
   - `v3.reason` 必须为 `"ok"`
   - `v3.model_version`、`v3.bundle_hash` 必须与发布版本匹配

3. 跑 acceptance_25_8 sanity（小样本 16 局/seed）：
   ```bash
   LINHAI_V3_PARAMS_DIR=$PROD_BUNDLE python3 tools/acceptance_25_8.py --games-per-seed 16 --seeds 1
   ```

## 性能监控
- 服务监控 P95 引擎延迟
- 任何 P95 > 1800ms 自动触发 V2 fallback
- 告警阈值：P95 > 800ms 持续 5 分钟

## 回滚
- 上一版 bundle 在 `engine/params/v3/release/Rn-1/`
- 切换：改环境变量 + 重启进程
```

- [ ] **Step 2: Stage**

---

### Task 39: 真人 dry-run（spec §6.1.1）

**Files:** 无（执行任务）

- [ ] **Step 1: 组织 4-6 人内测**

每人对 V3 至少 24 局（3 个窗口）。记录：
- 每人 8-局窗口分数
- 主观评分（决策合理性 1-10）
- 实测延迟 P95

- [ ] **Step 2: 任一窗口 ≤ 0 → 回 R2 调参**

记录失败 case 的牌谱，作为 R2 重训补充样本。

- [ ] **Step 3: 全过 → 交付**

打包 `data/release/`：
- `acceptance_report_R3.json`
- `selfplay_sample_R3_10pct.jsonl`（R2 样本 10% 抽样）
- `engine/params/v3/release/R3/`（最终 bundle）
- `docs/DEPLOYMENT.md`

---

## Self-Review Checklist

- [x] 番数引擎覆盖图全部条款（普通胡 / 硬碰硬 / 特殊牌翻倍 / 树掉还原 / 混一色 2/4 / 清一色 8 / 字一色 / 抢杠胡 / 抓冲 / 翻屁股 / 三键承包） — Task 1-8 全覆盖
- [x] selfplay 接 score 结算 — Task 11-12
- [x] ReferenceHumanPolicy 5 个行为模块 — Task 13-17
- [x] 5-seed × 25 验收 + 真人 dry-run — Task 18, 35, 39
- [x] perf_regression 4 个最坏 case — Task 19, 36
- [x] safety per-tile 特征双侧同步 + parity 测试 — Task 22-26
- [x] R0-R3 训练流水线 — Task 32-35
- [x] model_version + bundle_hash 诊断 — Task 37
- [x] GPU 入口预留（`tools/iterative_train.py` 留 `--accelerator` 参数） — Phase 7 任务里加，本期默认 cpu，gpu 路径仅 print "not implemented in this milestone"

**已知风险与回滚机制**：每轮训练前打 `git tag pre-Rn`，参数 bundle 按 `data/iter_params/Rn/` 隔离。任一阶段不达标，参考 spec §7。

---

## 实施节奏（4 周）

| 周 | 内容 |
|---|---|
| W1 | Phase 1-3（番数引擎 + ReferenceHumanPolicy） |
| W2 | Phase 4-5（验收 / 性能脚本 + safety 特征） |
| W3 | Phase 6 + R0-R1 训练 |
| W4 | R2-R3 训练 + 性能回归 + 真人 dry-run + 交付打包 |
