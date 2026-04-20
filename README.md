# majiangV3 / linhai-majiang-v3

临海麻将 V3 引擎与服务层。C++ 搜索引擎 + Python 编排 + 离线训练/评估工具。

## 路径/环境变量

生产环境通过以下环境变量覆盖默认搜索路径，避免再依赖 macOS 的绝对路径：

- `LINHAI_V3_ENGINE_MODULE_DIR`：V3 pybind 扩展目录（含 `linhai_v3*.so`）。多路径用 `:` 分隔。
- `LINHAI_V3_PARAMS_DIR`：V3 参数目录（含 `agari_prob/linhai/*.txt` 或 `v3/...`）。
- `LINHAI_V2_MODULE_DIR`：V2 旧 `linhai_ai*.so` 所在目录。
- `LINHAI_V2_PARAMS_DIR`：V2/akochan 参数目录。
- `LINHAI_EXTRA_INCLUDE_DIRS` / `LINHAI_EXTRA_LIBRARY_DIRS` / `LINHAI_EXTRA_LIBRARIES` / `LINHAI_EXTRA_COMPILE_ARGS` / `LINHAI_EXTRA_LINK_ARGS`：`engine/setup.py` 构建时的额外参数（用于非默认的 boost / libomp 安装位置）。

如果都不设置，会依次尝试仓库内相对路径（`engine/`, `engine/params/`, `backend/`）以及已知的开发机路径。

## 运维诊断

服务启动后 GET `/debug/engine_status`，返回实际生效的引擎：

```json
{
  "active_engine": "v3",
  "v3": {"available": true, "reason": "ok", "module_dir": "...", "params_dir": "..."},
  "v2": {"available": true, "reason": "ok", "module_dir": "...", "params_dir": "..."},
  "heuristic_available": true
}
```

如果 `active_engine == "heuristic"`，服务处于降级模式；看 `v3.reason` / `v2.reason` 找原因。

## 胜率回归

```
python3 tools/selfplay_eval.py --policy-a heuristic --policy-b heuristic --games 200 --seed 1
python3 tools/selfplay_eval.py --policy-a orchestrator:v3 --policy-b heuristic --games 500 --seed 1
```

详见 `docs/EVAL.md`。

## 目录

- `engine/` — C++ 引擎 + pybind 绑定 + 版本化参数
- `backend/` — FastAPI 服务、orchestrator、fallback 链、core state
- `tools/` — 数据抽取、训练 stub、离线评估、自对弈胜率回归
- `tests/cpp/`、`tests/python/` — 单测与集成测试
- `docs/` — 状态/特征/参数/评估约定
