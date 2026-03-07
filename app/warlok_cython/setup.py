# setup.py  —  build WARL0K Cython GRU inference extension
#
# Usage
# ─────
#   pip install cython numpy
#   python setup.py build_ext --inplace
#
# Produces:
#   warlok_gw/gru_infer.cpython-3XX-linux-gnu.so    (Linux/sidecar)
#   warlok_gw/gru_infer.cpython-3XX-darwin.so        (macOS dev)
#   warlok_gw/gru_infer.cpython-3XX-win_amd64.pyd   (Windows)
#
# Deploy the .so + warlok_gw/__init__.py + model_io.py + gateway.py
# alongside model.npz to any gateway / sidecar host.
#
# Cross-compile for arm64 gateway appliance
# ─────────────────────────────────────────
#   CC=aarch64-linux-gnu-gcc \
#   CFLAGS="-O3 -march=armv8-a+simd -ffast-math" \
#   python setup.py build_ext --inplace
#
# Cython annotation (bottleneck analysis)
# ───────────────────────────────────────
#   python setup.py build_ext --inplace --annotate
#   open warlok_gw/gru_infer.html   # yellow lines = Python overhead remaining

from setuptools import setup, Extension
import numpy as np
import sys

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("[setup.py] WARNING: Cython not found — trying pre-generated .c file")

source = "warlok_gw/gru_infer.pyx" if USE_CYTHON else "warlok_gw/gru_infer.c"

# ── Compiler flags per platform ───────────────────────────────────────────────
if sys.platform == "win32":
    # MSVC
    compile_args = ["/O2", "/fp:fast", "/arch:AVX2"]
    link_args    = []
elif sys.platform == "darwin":
    compile_args = ["-O3", "-ffast-math", "-funroll-loops", "-march=native"]
    link_args    = []
else:
    # Linux / sidecar / container
    compile_args = [
        "-O3",
        "-ffast-math",        # allow FP reassociation (safe for bounded inputs)
        "-funroll-loops",     # unroll the H_DIM=64 inner loops
        "-march=native",      # AVX2/AVX512 on server CPUs
    ]
    link_args = []

ext = Extension(
    name               = "warlok_gw.gru_infer",
    sources            = [source],
    include_dirs       = [np.get_include()],
    extra_compile_args = compile_args,
    extra_link_args    = link_args,
    define_macros      = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

cython_directives = {
    "language_level":    "3",
    "boundscheck":       False,
    "wraparound":        False,
    "cdivision":         True,
    "nonecheck":         False,
    "initializedcheck":  False,
    "profile":           False,
    "linetrace":         False,
}

setup(
    name             = "warlok_gw",
    version          = "1.0.0",
    description      = "WARL0K GRU gateway inference engine",
    packages         = ["warlok_gw"],
    ext_modules      = cythonize([ext],
                           compiler_directives=cython_directives,
                           annotate=False,
                       ) if USE_CYTHON else [ext],
    install_requires = ["numpy>=1.21"],
    python_requires  = ">=3.8",
)
