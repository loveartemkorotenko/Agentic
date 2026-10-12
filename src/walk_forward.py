"""Rolling-origin evaluation: N consecutive test windows, each with its own train/val/embargo."""
import argparse
import contextlib
import io
import json

import pandas as pd

import numpy as np
from sklearn.metrics import average_precision_score

import src.evaluate as ev
from src.evaluate import ROOT, run


def main(horizon, windows, test_days, val_days):
    rows, pool = [], []
    for w in range(windows):
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(horizon, test_days, val_days, end_offset_days=w * test_days, tag=f"_wf{w}")
        pool.append({k: np.asarray(v) for k, v in ev.LAST.items()})
        m = r["macro_onset"]
        base = {k: v for k, v in m["mean_pr_auc"].items() if k != "lightgbm"}
        best = max(base, key=base.get)
        rows.append({"test_start": r["test_period"][0][:10], "test_end": r["test_period"][1][:10],
                     "n_units": m["n_units"], "lightgbm": m["mean_pr_auc"]["lightgbm"],
                     "best_baseline": best, "best_baseline_pr": base[best],
                     "lift": round(m["mean_pr_auc"]["lightgbm"] - base[best], 4),
                     "beats_share": m["lightgbm_beats_best_baseline_share"],
                     "pooled_pr_auc": r["all"]["lightgbm"]["pr_auc"]})
    out = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    print(out.to_string())
    print(f"\nmean lift {out['lift'].mean():.4f} | windows with lift>0: {(out['lift'] > 0).sum()}/{len(out)}")
    out.to_csv(ROOT / "reports" / f"walk_forward_h{horizon}.csv", index=False)
    bootstrap(pool, horizon)


def bootstrap(pool, horizon, n_boot=300, seed=0):
    """Block bootstrap over calendar weeks, pooled across windows (onset rows only).
    Reports LightGBM minus occ_mean_24h baseline for Brier (positive = LightGBM better) and PR-AUC."""
    cat = {k: np.concatenate([p[k] for p in pool]) for k in pool[0]}
    wk = pd.Series(pd.to_datetime(cat["time"])).dt.tz_localize(None).dt.to_period("W").astype(str).to_numpy()
    weeks, inv = np.unique(wk, return_inverse=True)
    rows_of = [np.where(inv == i)[0] for i in range(len(weeks))]
    rng = np.random.default_rng(seed)
    d_brier, d_ap = [], []
    for _ in range(n_boot):
        ix = np.concatenate([rows_of[i] for i in rng.integers(0, len(weeks), len(weeks))])
        y, m, b = cat["y"][ix], cat["model"][ix], cat["base"][ix]
        d_brier.append(np.mean((b - y) ** 2) - np.mean((m - y) ** 2))
        d_ap.append(average_precision_score(y, m) - average_precision_score(y, b))
    res = {"n_weeks": int(len(weeks)), "n_boot": n_boot}
    for name, d in (("brier_improvement", d_brier), ("pr_auc_improvement", d_ap)):
        res[name] = {"mean": round(float(np.mean(d)), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                     "p_better": round(float(np.mean(np.array(d) > 0)), 3)}
    print("\nweek-block bootstrap (pooled onset rows):", json.dumps(res))
    (ROOT / "reports" / f"walk_forward_bootstrap_h{horizon}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--val-days", type=int, default=45)
    a = ap.parse_args()
    main(a.horizon, a.windows, a.test_days, a.val_days)
