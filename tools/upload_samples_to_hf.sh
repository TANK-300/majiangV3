#!/usr/bin/env bash
# upload_samples_to_hf.sh — 把 data/iter_samples/ 上传到 Hugging Face Datasets
#
# Usage:
#   bash tools/upload_samples_to_hf.sh <hf_username> [repo_name]
#
# Example:
#   bash tools/upload_samples_to_hf.sh fangyajun666
#   bash tools/upload_samples_to_hf.sh fangyajun666 majiangv3-iter-samples-v2
#
# Prereqs:
#   1. HuggingFace 账号 + access token (write 权限)
#      https://huggingface.co/settings/tokens
#   2. 首次运行前执行 `hf auth login` 输入 token
#      (新版 huggingface_hub>=1.0 把 CLI 命令从 huggingface-cli 改成了 hf)
set -euo pipefail

# 把 user-site 的 hf 命令加入 PATH (macOS pip --user 默认安装到此目录)
export PATH="${HOME}/Library/Python/3.9/bin:${HOME}/.local/bin:${PATH}"

HF_USER="${1:-}"
REPO_NAME="${2:-majiangv3-iter-samples}"

if [[ -z "${HF_USER}" ]]; then
  echo "[!] Usage: bash $0 <hf_username> [repo_name]" >&2
  exit 1
fi

REPO_ID="${HF_USER}/${REPO_NAME}"
SAMPLES_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/iter_samples"

if [[ ! -d "${SAMPLES_DIR}" ]]; then
  echo "[!] ${SAMPLES_DIR} 不存在" >&2
  exit 1
fi

echo "[*] 上传源目录: ${SAMPLES_DIR}"
echo "[*] 目标仓库:   https://huggingface.co/datasets/${REPO_ID}"
echo

# 1) 工具自检
if ! command -v hf &>/dev/null; then
  echo "[*] huggingface_hub[cli] 未安装，pip 安装中..."
  python3 -m pip install -U "huggingface_hub[cli]"
fi

# 2) 登录态校验
if ! hf auth whoami &>/dev/null; then
  echo "[!] 未登录 HuggingFace。请先执行：" >&2
  echo "      hf auth login" >&2
  echo "    token 申请地址: https://huggingface.co/settings/tokens (需 write 权限)" >&2
  exit 1
fi

CURRENT_USER="$(hf auth whoami | head -n1)"
echo "[✓] 已登录 HuggingFace 账号: ${CURRENT_USER}"

# 3) 创建数据集仓库（已存在则忽略报错）
echo "[*] 准备数据集仓库 ${REPO_ID} ..."
hf repo create "${REPO_ID}" --repo-type dataset -y 2>/dev/null \
  || echo "    (仓库可能已存在，跳过创建)"

# 4) 大目录上传：分块 + 并发 + 断点续传
echo "[*] 开始上传 (此过程可能耗时数十分钟到数小时，可随时 Ctrl+C 后重跑续传)..."
hf upload-large-folder \
  "${REPO_ID}" \
  "${SAMPLES_DIR}" \
  --repo-type=dataset \
  --num-workers=4

echo
echo "[✓] 上传完成"
echo "    浏览数据集: https://huggingface.co/datasets/${REPO_ID}"
echo "    更新本地文档: 把 data/iter_samples/README.md 里的 <YOUR_HF_USERNAME>"
echo "                  替换为 ${HF_USER}"
