"""Leakage-safe feature/target construction on the raion grid.

Forecast origin t is an hour boundary. Everything at origin i uses only:
  * grid steps strictly before t (state[t-1], run lengths, last-alert timing)
  * fully elapsed hours j < i     (occupancy, episode-start counts)
The target looks only forward: any alert step in [t, t + horizon).
"""
import numpy as np
import pandas as pd

from src.data import RES

BPH = 60 // RES                       # grid steps per hour
WARMUP_H = 24 * 7
ROLL_WINDOWS = (3, 6, 12, 24, 72, 168)
SINCE_CAP_MIN = 7 * 24 * 60


def neighbor_weights(idx, state, cutoff, k: int = 5) -> np.ndarray:
    """Data-driven 'neighbours': top-k raions whose hourly alert occupancy correlates most with
    each raion. Estimated ONLY on data before `cutoff` (end of the training period)."""
    n = int((cutoff - idx[0]).total_seconds() // (60 * RES)) // BPH * BPH
    occ = state[:n].reshape(-1, BPH, state.shape[1]).mean(1)
    c = np.nan_to_num(np.corrcoef(occ.T))
    np.fill_diagonal(c, -np.inf)
    W = np.zeros_like(c, dtype=np.float32)
    top = np.argsort(-c, axis=1)[:, :k]
    for i, js in enumerate(top):
        W[i, js] = 1.0 / k
    return W


def build_dataset(idx, units, unit_oblast, state, starts, horizon_h: int,
                  require_target: bool = True, W=None, extra=None) -> pd.DataFrame:
    T, K = state.shape
    Hn = T // BPH
    b = np.arange(Hn) * BPH                                 # grid step of each origin t
    hour_ts = idx[b]

    occ = state[: Hn * BPH].reshape(Hn, BPH, K).mean(1, dtype=np.float32)
    starts_h = starts[: Hn * BPH].reshape(Hn, BPH, K).sum(1, dtype=np.float32)

    # --- step-level lookbacks, evaluated at step t-1 ------------------------
    ar = np.arange(T, dtype=np.int32)[:, None]
    last_active = np.maximum.accumulate(np.where(state, ar, np.int32(-1)), axis=0)
    last_inactive = np.maximum.accumulate(np.where(~state, ar, np.int32(-1)), axis=0)
    prev = np.clip(b - 1, 0, None)
    act_now = state[prev]
    since_end = np.where(last_active[prev] < 0, SINCE_CAP_MIN, (prev[:, None] - last_active[prev]) * RES)
    since_end = np.minimum(since_end, SINCE_CAP_MIN).astype(np.float32)
    run_len = np.where(act_now, (prev[:, None] - last_inactive[prev]) * RES, 0).astype(np.float32)
    del last_active, last_inactive

    # --- forward target -----------------------------------------------------
    cs = np.vstack([np.zeros((1, K), dtype=np.int32), np.cumsum(state, axis=0, dtype=np.int32)])
    end = np.minimum(b + horizon_h * BPH, T)
    target = (cs[end] - cs[b]) > 0
    target_complete = (b + horizon_h * BPH) <= T
    del cs

    # --- hourly rolling features (strictly past hours) ----------------------
    def past_sum(a, w):  # sum of a[i-w .. i-1]
        c = np.vstack([np.zeros((1, K), dtype=np.float32), np.cumsum(a, axis=0, dtype=np.float64)]).astype(np.float32)
        out = np.full(a.shape, np.nan, dtype=np.float32)
        i = np.arange(w, Hn)
        out[i] = c[i] - c[i - w]
        return out

    feats = {"act_now": act_now.astype(np.float32), "since_end_min": since_end, "run_len_min": run_len}
    lag1 = np.full(occ.shape, np.nan, dtype=np.float32); lag1[1:] = occ[:-1]
    feats["occ_lag1"] = lag1
    for w in ROLL_WINDOWS:
        feats[f"occ_mean_{w}h"] = past_sum(occ, w) / w
        feats[f"starts_{w}h"] = past_sum(starts_h, w)
    prof = np.zeros(occ.shape, dtype=np.float32); cnt = np.zeros(occ.shape, dtype=np.float32)
    for d in range(1, 8):                                    # same hour-of-day, previous 7 days
        sh = np.full(occ.shape, np.nan, dtype=np.float32); sh[24 * d:] = occ[: Hn - 24 * d]
        ok = ~np.isnan(sh)
        prof[ok] += sh[ok]; cnt[ok] += 1
    feats["occ_same_hour_7d"] = np.where(cnt > 0, prof / np.maximum(cnt, 1), np.nan)

    # context: share of raions in the same oblast / whole country, past info only
    same = (unit_oblast[:, None] == unit_oblast[None, :]).astype(np.float32)
    same /= same.sum(1, keepdims=True)
    occ3 = past_sum(occ, 3) / 3
    feats["oblast_active_frac"] = act_now.astype(np.float32) @ same
    feats["oblast_occ_3h"] = np.nan_to_num(occ3, nan=0.0) @ same
    if W is not None:                                       # W[i, j]: weight of neighbour j for raion i
        feats["nb_active_frac"] = act_now.astype(np.float32) @ W.T
        feats["nb_occ_3h"] = np.nan_to_num(occ3, nan=0.0) @ W.T
        feats["nb_starts_3h"] = np.nan_to_num(past_sum(starts_h, 3), nan=0.0) @ W.T
        feats["nb_occ_24h"] = np.nan_to_num(past_sum(occ, 24) / 24, nan=0.0) @ W.T
    feats["nat_active_frac"] = np.repeat(act_now.mean(1, keepdims=True, dtype=np.float32), K, 1)
    feats["nat_occ_24h"] = np.repeat(np.nan_to_num(past_sum(occ, 24) / 24).mean(1, keepdims=True), K, 1)

    if extra:                                               # e.g. text features, already (Hn, K)
        feats.update(extra)

    local = hour_ts.tz_convert("Europe/Kyiv")
    cal = {"hour": local.hour.to_numpy(), "dow": local.dayofweek.to_numpy()}

    valid = np.arange(Hn) >= WARMUP_H
    if require_target:
        valid &= target_complete
    frames = []
    for k, u in enumerate(units):
        d = {name: arr[valid, k] for name, arr in feats.items()}
        d.update({c: v[valid] for c, v in cal.items()})
        d["unit"] = u
        d["oblast"] = unit_oblast[k]
        d["time"] = hour_ts[valid]
        d["y"] = target[valid, k].astype(np.int8)
        frames.append(pd.DataFrame(d))
    out = pd.concat(frames, ignore_index=True)
    out["unit"] = out["unit"].astype("category")
    out["oblast"] = out["oblast"].astype("category")
    return out.sort_values(["time", "unit"]).reset_index(drop=True)
