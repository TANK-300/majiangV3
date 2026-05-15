#!/usr/bin/env bash
# 停止 majiangV3 FastAPI 服务（与 start.sh 端口约定一致）
#
# 用法:
#   ./stop.sh            # 默认 9000
#   ./stop.sh 8002       # 指定端口
#   PORT=8001 ./stop.sh  # 环境变量指定端口
#
# 行为：先 SIGTERM，1.5s 内未退则 SIGKILL；端口本来就空时直接返回 0。

set -euo pipefail

PORT="${1:-${PORT:-9000}}"

PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
if [[ -z "${PIDS// /}" ]]; then
  echo "[stop.sh] 端口 ${PORT} 当前没有 LISTEN 进程，无需停止。"
  exit 0
fi

echo "[stop.sh] 端口 ${PORT} -> kill ${PIDS}"
kill ${PIDS} 2>/dev/null || true
sleep 1.5

REMAIN="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
if [[ -n "${REMAIN// /}" ]]; then
  echo "[stop.sh] SIGTERM 后仍存活，发送 SIGKILL: ${REMAIN}"
  kill -9 ${REMAIN} 2>/dev/null || true
  sleep 0.5
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[stop.sh] ⚠ 端口 ${PORT} 仍被占用：" >&2
  lsof -iTCP:"$PORT" -sTCP:LISTEN >&2
  exit 1
fi
echo "[stop.sh] ✓ 端口 ${PORT} 已释放"
