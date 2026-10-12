"""Task 2: an alert is active in a raion at origin t -- how much longer will it last?

Targets (at origins where an alert is active, i.e. the state at t-1 is 'alert'):
  * remaining minutes until the alert ends, capped at 24 h (a handful of frontline raions are
    'in alert' for months; the cap keeps them from dominating without selecting rows by label)
  * P(alert ends within 60 / 120 min)
Models: LightGBM quantile regression (q10/q50/q90 of log1p remaining) + binary classifiers.
Baselines use only elapsed time of the alert, estimated on the training period.
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data import grid, load_alerts
from src.evaluate import FOCUS, ROOT, feature_cols, split
from src.features import build_dataset
from src.telegram import load as load_posts
from src.text_features import text_features

CAP = 1440.0
ELAPSED_BINS = [0, 15, 30, 60, 120, 240, 480, 1e9]
PARAMS = dict(learning_rate=0.05, num_leaves=31, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7)


def make_dataset(since, text=None, horizon_h=6):
    alerts = load_alerts(since=since)
    idx, units, uo, state, starts = grid(alerts)
    extra, posts = None, None
    if text:
        posts = pd.concat([load_posts(c) for c in text.split(",")], ignore_index=True)
        extra = text_features(posts, idx, units, uo)
    df = build_dataset(idx, units, uo, state, starts, horizon_h, extra=extra, require_target=False)
    if posts is not None:
        df = df[df["time"] >= posts["published_at"].min().ceil("D") + pd.Timedelta(days=1)]
    d = df[(df["act_now"] == 1) & df["rem_min"].notna()].copy()
    d["rem_cap"] = d["rem_min"].clip(upper=CAP)
    d["bucket"] = pd.cut(d["run_len_min"], ELAPSED_BINS, right=False, labels=False)
    return d.reset_index(drop=True), alerts["finished_at"].max()


def baseline_tables(tr):
    g = tr.groupby("bucket")["rem_cap"]
    glob = pd.DataFrame({"med": g.median(), "p10": g.quantile(.1), "p90": g.quantile(.9),
                         "p60": g.apply(lambda s: (s <= 60).mean()), "p120": g.apply(lambda s: (s <= 120).mean())})
    gu = tr.groupby(["unit", "bucket"], observed=True)["rem_cap"]
    cnt = gu.size()
    uni = pd.DataFrame({"n": cnt, "med": gu.median(), "p10": gu.quantile(.1), "p90": gu.quantile(.9),
                        "p60": gu.apply(lambda s: (s <= 60).mean()), "p120": gu.apply(lambda s: (s <= 120).mean())})
    return glob, uni[uni["n"] >= 30]


def baseline_predict(df, glob, uni, per_unit):
    out = glob.reindex(df["bucket"].to_numpy()).reset_index(drop=True)
    out.index = df.index
    if per_unit:
        j = df[["unit", "bucket"]].astype({"unit": str}).merge(
            uni.reset_index().astype({"unit": str}), on=["unit", "bucket"], how="left")
        j.index = df.index
        for c in ("med", "p10", "p90", "p60", "p120"):
            out[c] = j[c].fillna(out[c])
    return out


def fit(tr, va, cols, objective, y_tr, y_va, **kw):
    p = dict(PARAMS, objective=objective, **kw)
    d_tr = lgb.Dataset(tr[cols], y_tr); d_va = lgb.Dataset(va[cols], y_va, reference=d_tr)
    return lgb.train(p, d_tr, 2000, valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])


def week_bootstrap(time, err_a, err_b, n_boot=2000, seed=0):
    """Paired block bootstrap over calendar weeks: mean(err_b - err_a) with 95% CI (>0 => a better)."""
    wk = pd.Series(time.dt.tz_localize(None).dt.to_period("W").astype(str).to_numpy())
    df = pd.DataFrame({"w": wk, "d": np.asarray(err_b) - np.asarray(err_a)})
    s = df.groupby("w")["d"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(s), (n_boot, len(s)))
    means = s["sum"].to_numpy()[pick].sum(1) / s["count"].to_numpy()[pick].sum(1)
    return {"mean": float(df["d"].mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n_weeks": int(len(s)), "p_better": float((means > 0).mean())}


def run(since="2024-01-01", test_days=60, val_days=60, text=None, tag="", tune=None):
    d, data_end = make_dataset(since, text)
    tr, va, te = split(d, 24, test_days, val_days)
    print(f"origins with active alert: train/val/test = {len(tr):,}/{len(va):,}/{len(te):,} | data end {data_end}")
    cols = feature_cols(d, False)
    cols = [c for c in cols if c not in ("rem_cap", "bucket")]
    kw = tune or {}

    ytr, yva = np.log1p(tr["rem_cap"]), np.log1p(va["rem_cap"])
    q = {a: fit(tr, va, cols, "quantile", ytr, yva, alpha=a, **kw) for a in (0.1, 0.5, 0.9)}
    pred = {a: np.expm1(m.predict(te[cols], num_iteration=m.best_iteration)) for a, m in q.items()}
    lo, med, hi = np.minimum(pred[0.1], pred[0.5]), pred[0.5], np.maximum(pred[0.9], pred[0.5])
    # conformalised quantile regression (CQR): widen [q10, q90] in log space by the 80% quantile of
    # validation non-conformity scores, so that the interval is calibrated on held-out data
    vlo, vhi, vy = (q[0.1].predict(va[cols], num_iteration=q[0.1].best_iteration),
                    q[0.9].predict(va[cols], num_iteration=q[0.9].best_iteration), np.log1p(va["rem_cap"]).to_numpy())
    score = np.maximum(vlo - vy, vy - vhi)
    corr = float(np.quantile(score, min(1.0, 0.8 * (1 + 1 / len(score)))))
    llo = np.log1p(lo) - corr
    lo_c, hi_c = np.expm1(np.maximum(llo, 0)), np.minimum(np.expm1(np.log1p(hi) + corr), CAP)
    clf = {h: fit(tr, va, cols, "binary", (tr["rem_cap"] <= h).astype(int), (va["rem_cap"] <= h).astype(int), **kw)
           for h in (60, 120)}
    p_end = {h: m.predict(te[cols], num_iteration=m.best_iteration) for h, m in clf.items()}

    glob, uni = baseline_tables(tr)
    bases = {"bucket": baseline_predict(te, glob, uni, False), "unit_bucket": baseline_predict(te, glob, uni, True)}

    y = te["rem_cap"].to_numpy()
    res = {"test_period": [str(te.time.min()), str(te.time.max())], "n_test": int(len(te)), "regression": {}, "ends_within": {},
           "interval_80": {}, "kyiv": {}, "bootstrap_week": {}}
    res["regression"]["lightgbm"] = {"mae_min": round(float(np.abs(med - y).mean()), 1),
                                    "mae_log": round(float(np.abs(np.log1p(med) - np.log1p(y)).mean()), 4),
                                    "median_ae_min": round(float(np.median(np.abs(med - y))), 1)}
    for n, b in bases.items():
        m = b["med"].to_numpy()
        res["regression"][n] = {"mae_min": round(float(np.abs(m - y).mean()), 1),
                                "mae_log": round(float(np.abs(np.log1p(m) - np.log1p(y)).mean()), 4),
                                "median_ae_min": round(float(np.median(np.abs(m - y))), 1)}
    res["interval_80"]["lightgbm"] = {"coverage": round(float(((y >= lo) & (y <= hi)).mean()), 3),
                                      "median_width_min": round(float(np.median(hi - lo)), 1)}
    res["interval_80"]["lightgbm_conformal"] = {"coverage": round(float(((y >= lo_c) & (y <= hi_c)).mean()), 3),
                                                "median_width_min": round(float(np.median(hi_c - lo_c)), 1),
                                                "log_correction": round(corr, 3)}
    for n, b in bases.items():
        l, h_ = b["p10"].to_numpy(), b["p90"].to_numpy()
        res["interval_80"][n] = {"coverage": round(float(((y >= l) & (y <= h_)).mean()), 3),
                                 "median_width_min": round(float(np.median(h_ - l)), 1)}
    for h in (60, 120):
        yy = (y <= h).astype(int)
        row = {"pos_rate": round(float(yy.mean()), 3)}
        for n, p in [("lightgbm", p_end[h])] + [(n, b[f"p{h}"].to_numpy()) for n, b in bases.items()]:
            row[n] = {"pr_auc": round(average_precision_score(yy, p), 4), "roc_auc": round(roc_auc_score(yy, p), 4),
                      "brier": round(brier_score_loss(yy, np.clip(p, 0, 1)), 4)}
        res["ends_within"][f"{h}min"] = row

    # paired week-block bootstrap: LightGBM vs the strongest baseline on MAE
    best = min(bases, key=lambda n: res["regression"][n]["mae_min"])
    res["bootstrap_week"]["mae_min_improvement_vs_" + best] = week_bootstrap(
        te["time"], np.abs(med - y), np.abs(bases[best]["med"].to_numpy() - y))
    res["bootstrap_week"]["brier60_improvement_vs_" + best] = week_bootstrap(
        te["time"], (p_end[60] - (y <= 60)) ** 2, (bases[best]["p60"].to_numpy() - (y <= 60)) ** 2)

    for u in FOCUS:
        m = (te["unit"].astype(str) == u).to_numpy()
        if m.sum() >= 30:
            bm = bases[best]["med"].to_numpy()
            res["kyiv" if u == "Kyiv City" else u] = {
                "n": int(m.sum()), "lgbm_mae": round(float(np.abs(med[m] - y[m]).mean()), 1),
                "baseline_mae": round(float(np.abs(bm[m] - y[m]).mean()), 1),
                "coverage80": round(float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean()), 3)}
    imp = pd.Series(q[0.5].feature_importance("gain"), index=cols).sort_values(ascending=False)
    res["top_features_gain_median"] = {k: round(float(v), 1) for k, v in imp.head(10).items()}
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    (out / f"duration{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("regression", "interval_80", "ends_within", "bootstrap_week", "kyiv")}, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2024-01-01")
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--val-days", type=int, default=60)
    ap.add_argument("--text", default=None, help="comma-separated scraped channels")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.since, a.test_days, a.val_days, a.text, a.tag)
