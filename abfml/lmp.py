# SPDX-License-Identifier: LGPL-3.0-or-later
"""Register entry points for ABFML with LAMMPS."""

import os
import platform
from importlib import import_module
from pathlib import Path
from typing import Optional

if platform.system() == "Linux":
    lib_env = "LD_LIBRARY_PATH"
    preload_env = "LD_PRELOAD"
elif platform.system() == "Darwin":
    lib_env = "DYLD_FALLBACK_LIBRARY_PATH"
    preload_env = "DYLD_INSERT_LIBRARIES"
else:
    raise RuntimeError("Unsupported platform")


def get_env(paths: list[Optional[str]]) -> str:
    """Get the environment variable from given paths."""
    return ":".join(p for p in paths if p is not None)


def get_library_path(module: str, filename: str) -> list[str]:
    """Get library path from a module.

    Parameters
    ----------
    module : str
        The module name.
    filename : str
        The library filename pattern.

    Returns
    -------
    list[str]
        The library path.
    """
    try:
        m = import_module(module)
    except ModuleNotFoundError:
        return []
    else:
        libs = sorted(Path(m.__path__[0]).glob(filename))
        return [str(lib) for lib in libs]


# Get PyTorch C++ extension directory
try:
    import torch

    torch_dir = torch.utils.cmake_prefix_path
except ImportError:
    raise RuntimeError("PyTorch is not installed!")

# Set paths for ABFML
abfml_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))


# Set environment variables
os.environ[preload_env] = get_env([
    os.environ.get(preload_env)
])

os.environ[lib_env] = get_env([
    os.environ.get(lib_env),
    torch_dir,
    abfml_lib_dir,
])


def get_abfml_lib_dir() -> str:
    """Get the directory of the ABFML library."""
    return abfml_lib_dir
