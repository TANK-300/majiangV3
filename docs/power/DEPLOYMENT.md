# 部署运行手册

> 临海麻将 V3 算法验收版本上线指引。  
> 对应 spec：`docs/superpowers/specs/2026-04-28-linhai-v3-acceptance-design.md`

---

## 1. 上线前检查清单

### 1.1 环境

| 项 | 必须 |
|---|---|
| Python | ≥ 3.10 |
| 系统 | Linux x86_64（macOS 仅推荐开发；生产用 Linux） |
| 编译依赖 | boost ≥ 1.74、libomp（macOS）/ gomp（Linux）、pybind11 |
| 运行依赖 | 见 `backend/requirements.txt`（lightgbm 仅训练侧需要，推理侧不需要） |

### 1.2 必备环境变量

```bash
# C++ 引擎 .so 路径
export LINHAI_V3_ENGINE_MODULE_DIR=/path/to/engine

# 参数 bundle 目录（生产用 R3 bundle）
export LINHAI_V3_PARAMS_DIR=/path/to/engine/params/v3/release/R3

# Profile（spec §5）
export LINHAI_V3_PROFILE=prod  # 仅生产用 prod；fast 是开发/回归
```

### 1.3 编译 .so

```bash
cd engine
python3 setup.py build_ext --inplace
# 验证
python3 -c "import linhai_v3; print(linhai_v3.__file__)"
```

如果 boost 路径非默认，用 `LINHAI_EXTRA_INCLUDE_DIRS` / `LINHAI_EXTRA_LIBRARY_DIRS` 覆盖（见 `engine/setup.py`）。

### 1.4 启动后必检 (`/debug/engine_status`)

```bash
curl -s http://localhost:8000/debug/engine_status | python3 -m json.tool
```

**发布前 release gate**（spec §6.2）：
- `active_engine` **必须** 为 `"v3"`，否则上线阻塞
- `v3.reason` 必须为 `"ok"`
- `v3.model_version` 与发布版本一致（`R3` / `3.0.0-r3` 之类）
- `v3.bundle_hash` 与发布说明里的哈希一致（首 16 hex chars）
- `v3.profile` 必须为 `"prod"`

如果 `active_engine == "heuristic"`，看 `v3.reason` 找原因：
| reason | 含义 | 怎么改 |
|---|---|---|
| `import_failed:...` | .so 没编译 / 路径不对 | 重新 `setup.py build_ext --inplace` 或检查 `LINHAI_V3_ENGINE_MODULE_DIR` |
| `params_not_found` | bundle 目录不存在 | 检查 `LINHAI_V3_PARAMS_DIR` |
| `engine_init_failed:...` | C++ 异常 | 看后端日志 |

### 1.5 上线 sanity（小样本回归）

```bash
LINHAI_V3_PARAMS_DIR=$PROD_BUNDLE \
python3 tools/acceptance_25_8.py \
    --games-per-seed 16 --seeds 1 \
    --output /tmp/acceptance_sanity.json
```

不要求 `passed=true`（16 局太短），但必须无异常完成。

---

## 2. 性能监控

### 2.1 P95 阈值（spec §5.1）

- 引擎计算 P95 ≤ 800ms
- 端到端（含网络）P95 ≤ 2000ms

### 2.2 自动降级

服务实时监控 P95：

- P95 > 1800ms 持续 5 分钟 → 自动切 V2 fallback 并告警
- 单次请求 > 60% 预算 → 内部收 beam
- 单次请求 > 80% 预算 → 内部切 depth
- 单次请求 > 95% 预算 → 切 V2

### 2.3 性能回归（CI 强制）

```bash
LINHAI_V3_PARAMS_DIR=$PROD_BUNDLE \
python3 tools/perf_regression.py --runs 100
```

4 个最坏 case（4 白板 / 复杂副露 / 多抢杠 / 抓冲全开）任一 P95 > 800ms → CI 失败。

---

## 3. 回滚

每轮训练前打 git tag（`pre-Rn`），bundle 按 `data/iter_params/Rn/` 隔离。

回滚步骤：
1. `export LINHAI_V3_PARAMS_DIR=/path/to/previous/Rn-1`
2. 重启 `uvicorn`
3. 检查 `/debug/engine_status` 的 `bundle_hash` 与上一版本一致

---

## 4. 真人 dry-run（spec §6.1.1）

R3 完成后、交付甲方前必做：

1. 内部组织 4-6 人内测，每人 ≥ 24 局（3 个窗口）
2. 记录每人 8-局窗口分数 + 决策合理性主观评分（1-10）
3. 实测引擎延迟 P95
4. 任一窗口 ≤ 0 → 回 R2 重训

通过后产出交付包：
- `data/release/acceptance_report_R3.json`
- `data/release/selfplay_sample_R3_10pct.jsonl`（10% 抽样）
- `engine/params/v3/release/R3/`
- `docs/DEPLOYMENT.md`（本文）

---

## 5. 服务对接接口

按现状（无变化）：

- `POST /ai/recommend` — 弃牌建议
- `POST /ai/recommend-response` — 吃碰杠胡 / pass
- `POST /ai/recommend-all` — 所有候选弃牌评分
- `GET  /debug/engine_status` — 部署诊断
- `GET  /health` / `GET /health/ping` — 存活检查

详见 `backend/app/main.py` 与 `backend/app/routers/ai.py`。
