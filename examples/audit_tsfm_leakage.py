#!/usr/bin/env python3
"""Audit a train/test split for both verbatim AND affine-rescaled/shifted data leakage."""

from __future__ import annotations

import numpy as np
import flashmp


def main():
    rng = np.random.default_rng(42)
    train = np.cumsum(rng.standard_normal(50_000))
    test = np.cumsum(rng.standard_normal(4_000))

    # Plant hidden leakage:
    # 1. Verbatim window copy
    test[400:464] = train[12_345 : 12_345 + 64]
    # 2. Affine-rescaled & shifted copy (e.g. unit conversion / currency / normalization)
    test[1800:1864] = 2.75 * train[33_000 : 33_000 + 64] - 512.0
    # 3. Sign-flipped copy
    test[3100:3164] = -1.5 * train[41_000 : 41_000 + 64] + 100.0

    report = flashmp.scan_leakage(train, test, m=64, corr_thresh=0.999, check_neg=True)
    print(report.summary())


if __name__ == "__main__":
    main()
