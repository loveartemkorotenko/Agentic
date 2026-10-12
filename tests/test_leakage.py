"""Features at origin t must be identical when all data after t is deleted."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import RES, grid, load_alerts  # noqa: E402
from src.features import build_dataset  # noqa: E402

H = 6


def test_no_future_information():
    alerts = load_alerts(since="2025-06-01")
    idx, units, uo, state, starts = grid(alerts)
    full = build_dataset(idx, units, uo, state, starts, H)

    cut_ts = pd.Timestamp("2026-03-01 12:00", tz="UTC")
    cut = int((cut_ts - idx[0]).total_seconds() // (60 * RES))
    # origins up to the cut are kept (require_target=False), so a feature peeking
    # even one step past t sees truncated data and differs from the full run
    trunc = build_dataset(idx[:cut], units, uo, state[:cut].copy(), starts[:cut].copy(), H,
                          require_target=False)

    feat_cols = [c for c in full.columns if c not in ("y", "time", "unit", "oblast")]
    f = full[full["time"].isin(trunc["time"])].reset_index(drop=True)
    t = trunc[trunc["time"].isin(f["time"])].reset_index(drop=True)
    assert len(f) == len(t) > 1000
    assert (f["time"].to_numpy() == t["time"].to_numpy()).all()
    for c in feat_cols:
        np.testing.assert_allclose(f[c].astype(float), t[c].astype(float), equal_nan=True,
                                   err_msg=f"future leakage in feature {c}")
