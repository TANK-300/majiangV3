#!/usr/bin/env bash
# 启动 majiangV3 FastAPI 服务（V3 引擎 + heuristic 兜底）
#
# 用法:
#   ./start.sh            # 默认 0.0.0.0:8000
#   ./start.sh 8002       # 指定端口
#   PORT=8001 ./start.sh  # 环境变量指定端口
#   KILL_PORT=1 ./start.sh 8000   # 端口被占用时自动 kill 占用进程

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT="${1:-${PORT:-8000}}"
HOST="${HOST:-0.0.0.0}"

VENV_PY="$ROOT/.venv/bin/python"
VENV_UVICORN="$ROOT/.venv/bin/uvicorn"

if [[ ! -x "$VENV_UVICORN" ]]; then
  echo "[start.sh] .venv 不存在或缺少 uvicorn — 请先创建虚拟环境并 pip install -r backend/requirements.txt" >&2
  exit 1
fi

# 确认 V3 .so 与当前 venv Python ABI 匹配；不匹配则自动重编
PY_TAG="$("$VENV_PY" -c 'import sys; print(f"cpython-{sys.version_info.major}{sys.version_info.minor}")')"
if ! ls "$ROOT/engine/linhai_v3.${PY_TAG}-"*.so >/dev/null 2>&1; then
  echo "[start.sh] 没有匹配 ${PY_TAG} 的 linhai_v3 扩展，开始编译..."
  "$VENV_PY" -m pip install --quiet pybind11
  ( cd "$ROOT/engine" && "$VENV_PY" setup.py build_ext --inplace )
fi

# 端口占用检测；KILL_PORT=1 时自动结束占用进程
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  if [[ "${KILL_PORT:-0}" == "1" ]]; then
    PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN | tr '\n' ' ')"
    echo "[start.sh] 端口 ${PORT} 被占用，KILL_PORT=1 — kill ${PIDS}"
    kill ${PIDS} 2>/dev/null || true
    sleep 1
    if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "[start.sh] kill 后仍被占用，尝试 SIGKILL"
      kill -9 ${PIDS} 2>/dev/null || true
      sleep 1
    fi
  fi
  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[start.sh] 端口 ${PORT} 仍被占用：" >&2
    lsof -iTCP:"$PORT" -sTCP:LISTEN >&2
    echo "[start.sh] 提示：换端口 ./start.sh 8002，或 KILL_PORT=1 ./start.sh ${PORT}" >&2
    exit 1
  fi
fi

echo "[start.sh] starting uvicorn on ${HOST}:${PORT}"
exec "$VENV_UVICORN" backend.app.main:app \
  --host "$HOST" --port "$PORT" \
  --app-dir "$ROOT"
