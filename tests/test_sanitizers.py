"""Compile and run the C++20 ASan+UBSan native stress harness inside pytest."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cpp_asan_ubsan_stress_harness(tmp_path: Path):
    is_macos = platform.system().lower() == "darwin"
    cxx = (
        (shutil.which("clang++") or shutil.which("g++"))
        if is_macos
        else (shutil.which("g++") or shutil.which("clang++"))
    )
    if cxx is None:
        pytest.skip("No C++ compiler found")

    bin_path = tmp_path / "cpp_asan_harness"
    cmd = [
        cxx,
        "-O1",
        "-g",
        "-std=c++20",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(ROOT / "src" / "flashmp" / "_csrc" / "flashmp.cpp"),
        str(ROOT / "tests" / "cpp_asan_harness.cpp"),
        "-o",
        str(bin_path),
    ]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f"ASan build failed:\n{build.stdout}\n{build.stderr}"

    env = dict(os.environ)
    if is_macos:
        env["MallocNanoZone"] = "0"
    proc = subprocess.run([str(bin_path)], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"ASan harness failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    assert "ALL C++ ASAN/UBSAN HARNESS CHECKS PASSED!" in proc.stdout
