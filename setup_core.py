"""Build the compiled core in place:  python setup_core.py build_ext --inplace
(needs Cython and a C++ compiler; eulsim works without it, see eulsim/core.py)."""
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    name="eulsim-core",
    ext_modules=cythonize(
        [Extension("eulsim._core", ["eulsim/_core.pyx"], language="c++",
                   extra_compile_args=["-O3", "-std=c++17"])],
        compiler_directives={"language_level": "3"},
    ),
)
