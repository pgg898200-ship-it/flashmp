#!/usr/bin/env python3
"""Drop-in STUMPY migration example — identical API, zero Numba JIT cold start."""

from __future__ import annotations

import math
import time

# Swap `import stumpy` -> `import flashmp as stumpy`
import flashmp as stumpy


def main():
    # Generate a 4,096-tick periodic telemetry stream with a shape anomaly
    N = 4096
    m = 48
    x = [math.sin(2.0 * math.pi * i / m) + 0.3 * math.cos(4.0 * math.pi * i / m) for i in range(N)]
    # Inject a subtle same-amplitude shape discord at index 2400
    for k in range(m):
        x[2400 + k] = (1.0 if (k % 6) < 3 else -1.0) * 0.9

    t0 = time.perf_counter()
    mp = stumpy.stump(x, m)
    dt_ms = (time.perf_counter() - t0) * 1e3

    # Exact STUMPY column layout: col 0 = distance, col 1 = nearest neighbor index
    dists = [float(row[0]) for row in mp]
    best_discord = max(range(len(dists)), key=lambda i: dists[i] if math.isfinite(dists[i]) else -1.0)
    print(f"Computed exact Matrix Profile (N={N}, m={m}) in {dt_ms:.2f} ms (ISA: {stumpy.simd_isa()})")
    print(f"Top shape discord detected at index {best_discord} (planted at 2400), distance={dists[best_discord]:.3f}")


if __name__ == "__main__":
    main()
