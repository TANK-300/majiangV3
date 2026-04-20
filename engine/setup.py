from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import List

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def _split_env(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    return [item for item in raw.split(os.pathsep) if item]


def _first_existing(paths: List[Path]) -> List[Path]:
    return [p for p in paths if p.is_dir()]


def _default_include_dirs() -> List[str]:
    env = _split_env("LINHAI_EXTRA_INCLUDE_DIRS")
    if env:
        return env
    system = platform.system()
    if system == "Darwin":
        candidates = [
            Path("/opt/homebrew/opt/boost/include"),
            Path("/opt/homebrew/opt/libomp/include"),
            Path("/usr/local/opt/boost/include"),
            Path("/usr/local/opt/libomp/include"),
            Path("/usr/local/include"),
        ]
    else:
        candidates = [
            Path("/usr/include"),
            Path("/usr/local/include"),
        ]
    return [str(p) for p in _first_existing(candidates)]


def _default_library_dirs() -> List[str]:
    env = _split_env("LINHAI_EXTRA_LIBRARY_DIRS")
    if env:
        return env
    system = platform.system()
    if system == "Darwin":
        candidates = [
            Path("/opt/homebrew/opt/boost/lib"),
            Path("/opt/homebrew/opt/libomp/lib"),
            Path("/usr/local/opt/boost/lib"),
            Path("/usr/local/opt/libomp/lib"),
            Path("/usr/local/lib"),
        ]
    else:
        candidates = [
            Path("/usr/lib"),
            Path("/usr/local/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib/aarch64-linux-gnu"),
        ]
    return [str(p) for p in _first_existing(candidates)]


def _default_libraries() -> List[str]:
    env = _split_env("LINHAI_EXTRA_LIBRARIES")
    if env:
        return env
    libs = ["boost_serialization"]
    if platform.system() == "Darwin":
        libs.append("omp")
    else:
        libs.append("gomp")
    return libs


def _default_compile_args() -> List[str]:
    env = _split_env("LINHAI_EXTRA_COMPILE_ARGS")
    if env:
        return env
    args = ["-std=c++11", "-O2"]
    if platform.system() == "Darwin":
        args.extend(["-Xpreprocessor", "-fopenmp"])
    else:
        args.append("-fopenmp")
    return args


def _default_link_args() -> List[str]:
    env = _split_env("LINHAI_EXTRA_LINK_ARGS")
    if env:
        return env
    args: List[str] = []
    if platform.system() == "Darwin":
        for path in ("/opt/homebrew/opt/boost/lib", "/opt/homebrew/opt/libomp/lib"):
            if Path(path).is_dir():
                args.append(f"-Wl,-rpath,{path}")
    else:
        args.append("-fopenmp")
    return args


INCLUDE_DIRS = ["."] + _default_include_dirs()
LIBRARY_DIRS = _default_library_dirs()
LIBRARIES = _default_libraries()
COMPILE_ARGS = _default_compile_args()
LINK_ARGS = _default_link_args()


ext_modules = [
    Pybind11Extension(
        "linhai_v3",
        [
            "python_binding.cpp",
            "share/linhai_ai_engine.cpp",
            "share/linhai_ev_engine.cpp",
            "share/linhai_search_v3.cpp",
            "share/linhai_game_adapter.cpp",
            "share/linhai_risk.cpp",
            "share/linhai_bonus.cpp",
            "share/linhai_grab_charge.cpp",
            "share/linhai_rules.cpp",
            "share/linhai_shanten_v4.cpp",
            "share/calc_shanten.cpp",
            "share/calc_agari.cpp",
            "share/calc_yaku.cpp",
            "share/types.cpp",
            "share/json11.cpp",
            "share/include.cpp",
        ],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES,
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINK_ARGS,
    ),
]

setup(
    name="linhai_v3",
    version="3.0.0",
    author="LinHai Mahjong Team",
    description="\u4e34\u6d77\u9ebb\u5c06V3\u641c\u7d22\u5f15\u64ceePython\u7ed1\u5b9a",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
