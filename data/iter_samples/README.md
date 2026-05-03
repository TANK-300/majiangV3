# iter_samples — 自对弈训练样本

由于单文件超过 GitHub 100MB 限制（R0.jsonl 达 8GB），训练样本不在本仓库中存储，
托管在 Hugging Face Datasets：

**🤗 https://huggingface.co/datasets/<YOUR_HF_USERNAME>/majiangv3-iter-samples**

> 首次上传后，请把 `<YOUR_HF_USERNAME>` 替换成你的真实 HuggingFace 用户名。

---

## 下载样本

```bash
python3 -m pip install -U "huggingface_hub[cli]"

# 登录（一次即可，token 在 https://huggingface.co/settings/tokens 申请，read 权限够用）
hf auth login

# 下载到当前位置
hf download <YOUR_HF_USERNAME>/majiangv3-iter-samples \
  --repo-type dataset \
  --local-dir data/iter_samples
```

> 注：`huggingface_hub>=1.0` 起 CLI 命令从 `huggingface-cli` 改为 `hf`。

## 文件清单

| 文件 | 大小 | 说明 |
|---|---|---|
| `R0.jsonl` | 8.0 GB | 第 0 轮自对弈样本（完整） |
| `R0_10k.jsonl` | 2.4 GB | R0 采样 1 万局快照 |
| `R1.jsonl` | 1.0 GB | 第 1 轮自对弈样本 |
| `R2.jsonl` | 817 MB | 第 2 轮自对弈样本 |

每行一条 JSON 记录，字段约定见 `backend/app/services/score_calculator.py` 与
`docs/power/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md`。

## 上传新一轮样本（Rn.jsonl）

```bash
bash tools/upload_samples_to_hf.sh <YOUR_HF_USERNAME>
```

脚本会自动：
- 检查/安装 `huggingface_hub[cli]`
- 创建（或复用）数据集仓库
- 用 `upload-large-folder` 分块、并发、断点续传上传整个 `data/iter_samples/` 目录
