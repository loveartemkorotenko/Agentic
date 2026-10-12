"""Rolling-origin evaluation: N consecutive test windows, each with its own train/val/embargo."""
import argparse
import contextlib
import io
import json

import pandas as pd

from src.evaluate import ROOT, run


def main(horizon, windows, test_days, val_days):
    rows = []
    for w in range(windows):
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(horizon, test_days, val_days, end_offset_days=w * test_days, tag=f"_wf{w}")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--val-days", type=int, default=45)
    a = ap.parse_args()
    main(a.horizon, a.windows, a.test_days, a.val_days)
