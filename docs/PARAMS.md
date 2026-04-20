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
