"""Features at origin t must not change when every observation at/after t is erased.

The grid keeps its length (data after the cut is zeroed, not dropped) and the cut sits exactly
on an origin, so even a one-step look-ahead at that origin shows up as a difference.
"""
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
    full = build_dataset(idx, units, uo, state, starts, H, require_target=False)

    cut_ts = pd.Timestamp("2026-03-01 12:00", tz="UTC")
    cut = int((cut_ts - idx[0]).total_seconds() // (60 * RES))
    st, sf = state.copy(), starts.copy()
    st[cut:] = False
    sf[cut:] = False
    erased = build_dataset(idx, units, uo, st, sf, H, require_target=False)

    feat_cols = [c for c in full.columns if c not in ("y", "rem_min", "time", "unit", "oblast")]
    f = full[full["time"] <= cut_ts].reset_index(drop=True)
    t = erased[erased["time"] <= cut_ts].reset_index(drop=True)
    assert len(f) == len(t) > 1000 and (f["time"] == t["time"]).all()
    assert (f["time"] == cut_ts).any()
    for c in feat_cols:
        np.testing.assert_allclose(f[c].astype(float), t[c].astype(float), equal_nan=True,
                                   err_msg=f"future leakage in feature {c}")
