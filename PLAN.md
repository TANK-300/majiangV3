# 新建 `/Users/wf/Documents/wb/linhai-majiang-v3` 的完整最强方案

## 摘要
新项目名称固定为：`/Users/wf/Documents/wb/linhai-majiang-v3`。

方案选择固定为：**以 `Akochan` 范式为主骨架，彻底重构，不沿用当前 V2 的主决策架构**。  
实现策略采用“**Akochan 基座重起 + 临海规则/服务层择优迁入**”，不做简单复制，不做就地升级，不做平行脏分叉。

这个新项目的目标是形成一套完整闭环：
- 临海专用 canonical state
- Akochan 风格在线 EV/Search 引擎
- 临海专用离线特征/训练/参数体系
- Python 服务层与现有调用兼容
- V2 仅作为基线与回退，不再是主线

## 关键实现
### 1. 项目结构固定
新项目目录结构固定为：

- `/Users/wf/Documents/wb/linhai-majiang-v3/engine`
  - C++ 主引擎
  - `share/` 放核心状态、规则、搜索、特征、概率模型、绑定代码
  - `params/` 放版本化参数
- `/Users/wf/Documents/wb/linhai-majiang-v3/backend`
  - Python 服务层
  - 只保留状态构造、接口编排、降级与调试输出
- `/Users/wf/Documents/wb/linhai-majiang-v3/tools`
  - 数据抽取、样本生成、训练、评估、回归脚本
- `/Users/wf/Documents/wb/linhai-majiang-v3/tests`
  - C++ 单测
  - Python 集成测试
  - 固定牌例回归集
- `/Users/wf/Documents/wb/linhai-majiang-v3/docs`
  - 状态定义
  - 特征定义
  - 参数版本说明
  - 评估标准

固定迁移来源：
- `Akochan`：只迁移架构范式、搜索/参数组织方式、必要底层组件
- `linhai-majiang-v2`：迁移已验证过的临海规则实现、白板/抓冲/承包/风险相关逻辑
- `majiang-linhai-ai`：迁移现有 Python 状态、接口层、服务对接经验

明确不迁移的内容：
- 当前 V2 的“伪多巡 DP”主逻辑不进入新项目主干
- 当前 `UnifiedAI` 的主决策排序逻辑不作为新项目核心
- 当前 MCTS 不作为新项目主架构

### 2. 在线主引擎固定为 Akochan 风格 Linhai Search Engine
C++ 核心必须重建为新的主引擎，不沿用现有 `recommend_discard_ev` 作为中心。

固定新增主接口：
- `recommend_discard_v3(const GameState& state)`
- `recommend_response_v3(const GameState& state, int target_hai, int from_player, const std::vector<std::string>& available_actions)`
- `set_search_config(const SearchConfig& cfg)`
- `load_model_bundle(const std::string& version_dir)`

固定核心类型：
- `CanonicalGameState`
- `SearchConfig`
- `SearchContext`
- `SearchResult`
- `ModelBundle`

搜索形态固定为：
- 根节点枚举合法动作
- `Chance Node` 按精确剩余牌池枚举摸牌
- `Decision Node` 做最优动作选择
- 叶子节点调用训练好的临海参数模型估值
- 默认深度 2，自摸/响应关键局面升到 3
- 支持缓存与束搜索剪枝

在线估值统一整合：
- `agari_prob`
- `tenpai_prob`
- `houjuu_prob`
- `betaori`
- `ryukyoku_prob`
- 白板价值
- 抓冲价值
- 承包风险
- `pass_hu / can_win`
- 真实对手弃牌与安全信息

新引擎必须自带 fallback：
- V3 超时/异常/参数缺失时，回退到 V2 baseline
- V2 不可用时，再回退到 Strategy/heuristic baseline

### 3. 离线训练与参数体系必须作为正式子系统一起建设
这是最强方案的核心，不可后置。

固定要做的离线任务：
- 原始局面转 `CanonicalGameState`
- 样本提取
- 特征编码
- 标签生成
- 参数训练
- 离线评估
- 参数版本化发布

固定参数任务：
- `agari_prob`
- `tenpai_prob`
- `houjuu_prob`
- `betaori`
- `tsumo_num`
- `ryukyoku_prob`

特征必须按临海重建，不能照抄标准四麻：
- 双人局巡目与剩余牌池特征
- 白板万能牌相关特征
- 抓冲与承包相关特征
- 过胡不胡约束
- 对手弃牌、现物、壁、筋、防守压力
- 副露、杠、门前状态
- 手牌结构、向听、受入、待牌质量

固定产物：
- `tools/extract_*`
- `tools/train_*`
- `tools/eval_*`
- `params/v3/<task>/...`

### 4. Python 服务层只做编排，不再做主决策
`backend` 的职责固定为：
- 把外部请求转成 `CanonicalGameState`
- 调用 V3 引擎
- 处理降级逻辑
- 输出候选分数、解释和调试信息
- 保持与现有调用接口尽量兼容

固定调度顺序：
1. `V3 Search Engine`
2. `V2 baseline`
3. `Strategy baseline`

固定返回字段：
- `engine`
- `chosen_by`
- `candidate_scores`
- `total_ev`
- `agari_prob`
- `houjuu_prob`
- `defense_score`
- `search_depth`
- `nodes_expanded`
- `cache_hits`
- `fallback_reason`

### 5. MCTS 定位固定为辅助，不进入主干
新项目默认不把 MCTS 接入主链。

MCTS 的固定用途：
- 作为离线对比基线
- 作为 debug / ablation 工具
- 可选地在 top-2/top-3 分差极小时做线上 tie-breaker

默认不开启原因固定：
- 当前 MCTS rollout 仍是简化模拟
- 主上限来自 Akochan 风格参数与搜索
- 将 MCTS 作为主链会拉高延迟并引入额外噪声

如果未来要保留线上 MCTS，只允许：
- 作为 `SearchResult` 的后处理器
- 不允许改写底层参数模型
- 不允许替代主排序

## 测试与验收
### 1. 规则与状态
必须覆盖：
- 白板万能牌
- 抓冲与承包
- 过胡不胡
- 双人局对手识别
- 实际弃牌传递
- 响应动作合法性

### 2. 搜索正确性
必须覆盖：
- toy case 下 chance/decision 递推与手算一致
- 精确剩余牌池会改变决策排序
- 弃牌与响应都能输出候选 EV 排序
- 超时/截断时仍只返回合法动作

### 3. 参数与回归
必须建立固定回归集，比较：
- V3 vs V2
- V3 vs 简化 rollout
- 不同参数版本之间的策略稳定性

固定验收指标：
- 离线 top-1/top-k 命中率提升
- 自对弈胜率提升
- 关键牌例决策符合预期
- P95 延迟达标

### 4. 性能门槛
固定目标：
- 弃牌 P95 < 100ms
- 响应 P95 < 80ms
- 调试模式单独开关
- 超预算优先降深度和收窄 beam，不直接禁用 V3

## 假设与默认选择
- 新项目名固定为 `linhai-majiang-v3`
- 采用“必须重构”的原则，不做当前仓库上的就地升级
- 采用 `Akochan` 范式重起，但临海规则和服务层择优迁入
- V2 只作为 baseline/fallback，不再是新项目核心
- 训练管线默认纳入第一阶段，不后置
- MCTS 默认不进入主决策链
- 实施顺序固定为：
  1. 建新项目骨架与 canonical state
  2. 建离线样本与参数体系
  3. 建在线 C++ 搜索引擎
  4. 接 Python backend
  5. 建回归/评估体系
