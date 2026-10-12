"""Train / evaluate: time-ordered split with an embargo, baselines vs LightGBM."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from src.data import grid, load_alerts
from src.features import build_dataset, neighbor_weights

ROOT = Path(__file__).resolve().parents[1]
FOCUS = ["Kyiv City", "Kharkivska oblast / Kharkivskyi raion", "Dnipropetrovska oblast / Dniprovskyi raion",
         "Odeska oblast / Odeskyi raion", "Lvivska oblast / Lvivskyi raion"]


def split(df, horizon_h, test_days, val_days):
    """train | embargo | val | embargo | test. Embargo = horizon so that no training
    target window overlaps the period it is evaluated on."""
    end = df["time"].max()
    emb = pd.Timedelta(hours=horizon_h)
    test_start = end - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    val_start = test_start - pd.Timedelta(days=val_days)
    tr = df[df["time"] < val_start - emb]
    va = df[(df["time"] >= val_start) & (df["time"] < test_start - emb)]
    te = df[df["time"] >= test_start]
    return tr, va, te


def best_f1_threshold(y, p):
    ts = np.linspace(0.02, 0.98, 97)
    return ts[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in ts]))]


def metrics(y, p, thr):
    if len(np.unique(y)) < 2:
        return {"n": len(y), "pos_rate": float(np.mean(y))}
    return {"n": int(len(y)), "pos_rate": round(float(np.mean(y)), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "roc_auc": round(roc_auc_score(y, p), 4),
            "brier": round(brier_score_loss(y, np.clip(p, 0, 1)), 4),
            "f1": round(f1_score(y, p >= thr, zero_division=0), 4)}


def feature_cols(df, drop_ids=False):
    skip = {"y", "time"} | ({"unit", "oblast"} if drop_ids else set())
    return [c for c in df.columns if c not in skip]


def baselines(tr, df):
    """Return dict name -> prediction array for rows of df. Only past-derived inputs."""
    out = {"persistence": df["act_now"].to_numpy(float)}
    prof = df["occ_same_hour_7d"].fillna(0).to_numpy()
    out["same_hour_last_7d"] = prof
    out["occ_mean_24h"] = df["occ_mean_24h"].fillna(0).to_numpy()
    # logistic-free "climatology": train positive rate per (oblast, hour)
    clim = tr.groupby(["unit", "hour"], observed=True)["y"].mean().rename("clim")
    out["climatology"] = df.join(clim, on=["unit", "hour"])["clim"].fillna(tr["y"].mean()).to_numpy()
    return out


def run(horizon_h=6, test_days=60, val_days=60, since="2024-01-01", drop_ids=False, tag="", neighbors=False, end_offset_days=0, out_dir=ROOT / "reports"):
    alerts = load_alerts(since=since)
    idx, units, uo, state, starts = grid(alerts)
    train_end = idx[-1] - pd.Timedelta(days=test_days + val_days) - pd.Timedelta(hours=horizon_h)
    W = neighbor_weights(idx, state, train_end) if neighbors else None
    df = build_dataset(idx, units, uo, state, starts, horizon_h, W=W)
    if end_offset_days:  # pretend the data ends earlier: rolling-origin evaluation
        df = df[df["time"] <= df["time"].max() - pd.Timedelta(days=end_offset_days)]
    tr, va, te = split(df, horizon_h, test_days, val_days)
    print(f"data end (UTC): {alerts['finished_at'].max()}  |  rows train/val/test = "
          f"{len(tr):,}/{len(va):,}/{len(te):,}")
    print(f"train {tr.time.min()} -> {tr.time.max()} | val {va.time.min()} -> {va.time.max()} | "
          f"test {te.time.min()} -> {te.time.max()}")

    cols = feature_cols(df, drop_ids)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, seed=7)
    d_tr = lgb.Dataset(tr[cols], tr["y"]); d_va = lgb.Dataset(va[cols], va["y"], reference=d_tr)
    model = lgb.train(params, d_tr, 2000, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    print("best iteration:", model.best_iteration)

    preds_va = {"lightgbm": model.predict(va[cols], num_iteration=model.best_iteration)}
    preds_va.update(baselines(tr, va))
    preds_te = {"lightgbm": model.predict(te[cols], num_iteration=model.best_iteration)}
    preds_te.update(baselines(tr, te))

    # thresholds are tuned on validation only, never on test
    thr = {k: best_f1_threshold(va["y"].to_numpy(), v) for k, v in preds_va.items()}
    onset = te["act_now"].to_numpy() == 0
    results = {"horizon_h": horizon_h, "test_period": [str(te.time.min()), str(te.time.max())],
               "all": {}, "onset_only": {}, "per_oblast_lightgbm": {}}
    for name, p in preds_te.items():
        results["all"][name] = metrics(te["y"].to_numpy(), p, thr[name])
        results["onset_only"][name] = metrics(te["y"].to_numpy()[onset], p[onset], thr[name])
    # headline: macro average over raions, onset rows only (no alert at t), raions with
    # enough of both classes. Pooled numbers are dominated by saturated frontline raions.
    yt = te["y"].to_numpy(); unit_arr = te["unit"].astype(str).to_numpy()
    rows = []
    for u in np.unique(unit_arr):
        m = (unit_arr == u) & onset
        if yt[m].sum() >= 20 and (1 - yt[m]).sum() >= 20:
            rows.append({"unit": u, "n_pos": int(yt[m].sum()),
                         **{n: average_precision_score(yt[m], p[m]) for n, p in preds_te.items()}})
    macro = pd.DataFrame(rows).set_index("unit")
    methods = list(preds_te)
    best_base = macro[[c for c in methods if c != "lightgbm"]].max(axis=1)
    results["macro_onset"] = {
        "n_units": len(macro),
        "mean_pr_auc": {n: round(float(macro[n].mean()), 4) for n in methods},
        "lightgbm_beats_best_baseline_share": round(float((macro["lightgbm"] > best_base).mean()), 3)}
    macro.round(4).to_csv(out_dir / f"per_unit_onset_h{horizon_h}{tag}.csv")

    for ob in FOCUS:
        m = (te["unit"] == ob).to_numpy()
        results["per_oblast_lightgbm"][ob] = {
            "all": metrics(te["y"].to_numpy()[m], preds_te["lightgbm"][m], thr["lightgbm"]),
            "persistence_pr_auc": metrics(te["y"].to_numpy()[m], preds_te["persistence"][m], .5).get("pr_auc"),
            "onset": metrics(te["y"].to_numpy()[m & onset], preds_te["lightgbm"][m & onset], thr["lightgbm"])}

    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    results["top_features_gain"] = {k: round(float(v), 1) for k, v in imp.head(12).items()}
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"results_h{horizon_h}{tag}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for part in ("all", "onset_only"):
        print(f"\n== {part} ==")
        print(pd.DataFrame(results[part]).T.to_string())
    print("\n== macro over raions, onset rows ==\n", json.dumps(results["macro_onset"], indent=1))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--val-days", type=int, default=60)
    ap.add_argument("--since", default="2024-01-01")
    ap.add_argument("--drop-ids", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--neighbors", action="store_true", help="add data-driven neighbour features (no gain, kept as ablation)")
    a = ap.parse_args()
    run(a.horizon, a.test_days, a.val_days, a.since, a.drop_ids, a.tag, a.neighbors)
