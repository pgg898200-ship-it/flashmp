"""Compile and run the C++20 ASan+UBSan native stress harness inside pytest."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cpp_asan_ubsan_stress_harness(tmp_path: Path):
    cxx = shutil.which("g++") or shutil.which("clang++")
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
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    proc = subprocess.run([str(bin_path)], check=True, capture_output=True, text=True)
    assert "ALL C++ ASAN/UBSAN HARNESS CHECKS PASSED!" in proc.stdout
