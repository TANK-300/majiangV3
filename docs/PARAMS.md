# Params

参数目录采用版本化结构：

- `params/agari_prob/linhai`
- `params/tenpai_prob/linhai`
- `params/houjuu_prob/linhai`
- `params/betaori/linhai`
- `params/tsumo_num/linhai`
- `params/ryukyoku_prob/linhai`
- `params/v3/<task>/model.json`
- `params/v3/<task>/model_meta.json`
- `params/v3/score_table.json` — 番数与积分参数表（验收新增，spec §3.4）

## 验收新增任务（spec §3.3）

`task` 取值新增：

- `agari_score`：regression head，学 E[my_chong]（带号）
- `houjuu_score`：regression head，学 E[loss_chong]（带号）

label_key 都是 `task_labels.score_label`（同字段，trainer 内部按数值正负学习不同模式）。

## score_table.json schema（spec §3.4 默认值）

字段对应图条款：

| 字段 | 含义 |
|---|---|
| `chong_to_score` | 1 冲 = 多少积分（默认 1） |
| `base_chong.ordinary` | 普通胡基础冲（默认 1） |
| `base_chong.qingyise` | 清一色基础（默认 8） |
| `base_chong.ziyise` | 字一色基础（默认 8，spec T1） |
| `base_chong.hunyise_with_tree` | 混一色 + 有白板（默认 2） |
| `base_chong.hunyise_hard` | 混一色 + 无白板（默认 4） |
| `multipliers.hard_collision` | 硬碰硬倍数（默认 2） |
| `multipliers.tree_restore_white_anko` | 树掉还原（默认 4） |
| `bonuses.qianggang_hu` | 抢杠胡 +冲（默认 2，spec T2） |
| `bonuses.grab_charge_per_tile` | 每张抓冲 +冲（默认 1） |
| `redhead.tile_to_chong` | 翻屁股每张红头的冲数 |
| `contract.split_ratio` | 三键承包平摊比例（默认 [0.5, 0.5]） |
| `cap.max_base_chong` | 基础冲封顶（默认 8） |
| `cap.extra_chong_uncapped` | 抓冲是否计入封顶（默认 false / 不计入） |

当前 `params/v3/<task>/model.json` 允许记录：

- `task / label_key`
- `model_type`
- `feature_names`
- `feature_means / feature_stds`
- `intercept / weights`
- `training_examples`
- `metrics`

当前 `model_meta.json` 允许记录：

- `task / created_at / status`
- 可选 `label_key / model_file / model_type`
- 可选 `trained_examples / skipped_records / metrics`
- 可选 `dataset` 摘要：
  - `total_records`
  - `records_with_task_labels`
  - `label_key_counts`
