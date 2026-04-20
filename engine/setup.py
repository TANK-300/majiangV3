from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

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
        include_dirs=[
            ".",
            "/opt/homebrew/opt/boost/include",
            "/opt/homebrew/opt/libomp/include",
        ],
        library_dirs=[
            "/opt/homebrew/opt/boost/lib",
            "/opt/homebrew/opt/libomp/lib",
        ],
        libraries=["boost_serialization", "omp"],
        extra_compile_args=["-std=c++11", "-Xpreprocessor", "-fopenmp"],
        extra_link_args=["-Wl,-rpath,/opt/homebrew/opt/boost/lib", "-Wl,-rpath,/opt/homebrew/opt/libomp/lib"],
    ),
]

setup(
    name="linhai_v3",
    version="3.0.0",
    author="LinHai Mahjong Team",
    description="临海麻将V3搜索引擎Python绑定",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
