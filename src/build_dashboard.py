"""Build docs/index.html: a static, historical analytics page (NO live forecasts).

Everything shown is computed from past alerts and from the saved evaluation reports.
Run:  python -m src.build_dashboard
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import RES, RAW_PATH, grid, load_alerts

ROOT = Path(__file__).resolve().parents[1]
WINDOW_DAYS = 365
BINS = [0, 15, 30, 60, 120, 240, 480, 960, np.inf]          # minutes
BIN_LABELS = ["<15 хв", "15–30 хв", "30–60 хв", "1–2 год", "2–4 год", "4–8 год", "8–16 год", ">16 год"]


def ukrainian_names():
    """(en oblast, en raion) -> (uk oblast, uk raion); rows of the two dataset files are aligned."""
    en = pd.read_csv(RAW_PATH)
    uk = pd.read_csv(RAW_PATH.with_name("official_data_uk.csv"))
    m = pd.DataFrame({"oe": en["oblast"], "re": en["raion"], "ou": uk["oblast"], "ru": uk["raion"]}).drop_duplicates()
    names = {(r.oe, r.re): (r.ou, r.ru) for r in m.itertuples() if isinstance(r.re, str)}
    names[("Kyiv City", "Kyiv City")] = ("м. Київ", "м. Київ")
    return names


def unit_stats():
    alerts = load_alerts(since="2024-01-01")
    idx, units, uo, state, starts = grid(alerts)
    end = idx[-1]
    sel = idx >= end - pd.Timedelta(days=WINDOW_DAYS)
    st, idx_s = state[sel], idx[sel]
    local_h = idx_s.tz_convert("Europe/Kyiv").hour.to_numpy()
    week = ((idx_s - idx_s[0]) // pd.Timedelta(days=7)).to_numpy()
    n_weeks = len(idx_s) // (7 * 24 * 60 // RES)        # complete weeks only (a partial last week looks like a drop)
    names = ukrainian_names()
    out = []
    for k, u in enumerate(units):
        s = st[:, k]
        d = np.diff(np.concatenate([[0], s.astype(np.int8), [0]]))
        dur = (np.where(d == -1)[0] - np.where(d == 1)[0]) * RES
        oblast_en, _, raion_en = u.partition(" / ")
        raion_en = raion_en or oblast_en
        ou, ru = names.get((oblast_en, raion_en), (oblast_en, raion_en))
        hist = np.histogram(dur, bins=BINS)[0].tolist() if len(dur) else [0] * 8
        hour_share = [float(s[local_h == h].mean()) if (local_h == h).any() else 0.0 for h in range(24)]
        weekly = [round(float(s[week == w].sum() * RES / 60), 1) for w in range(n_weeks)]
        out.append({
            "oblast": ou, "raion": ru, "key": u, "n_episodes": int(len(dur)),
            "share_in_alert": round(float(s.mean()), 4),
            "median_min": float(np.median(dur)) if len(dur) else None,
            "p25_min": float(np.percentile(dur, 25)) if len(dur) else None,
            "p75_min": float(np.percentile(dur, 75)) if len(dur) else None,
            "p90_min": float(np.percentile(dur, 90)) if len(dur) else None,
            "share_le_60": round(float((dur <= 60).mean()), 3) if len(dur) else None,
            "hist": hist, "hour_share": [round(x, 4) for x in hour_share], "weekly_hours": weekly})
    return out, {"from": str(idx_s[0].date()), "to": str(end.date()), "weeks": n_weeks}


def model_summary():
    rep = ROOT / "reports"
    dur = json.loads((rep / "duration.json").read_text(encoding="utf-8"))
    wf = pd.read_csv(rep / "walk_forward_h6.csv")
    boot = json.loads((rep / "walk_forward_bootstrap_h6.json").read_text(encoding="utf-8"))
    horizons = {}
    for h in (1, 2, 3, 6):
        r = json.loads((rep / f"results_h{h}.json").read_text(encoding="utf-8"))
        m = r["macro_onset"]["mean_pr_auc"]
        horizons[h] = round(m["lightgbm"] - max(v for k, v in m.items() if k != "lightgbm"), 4)
    return {"duration": dur, "walk_forward": wf.to_dict("records"), "bootstrap": boot, "horizon_lift": horizons}


def main():
    units, meta = unit_stats()
    data = {"meta": meta, "units": sorted(units, key=lambda u: (u["oblast"], u["raion"])),
            "bins": BIN_LABELS, "model": model_summary()}
    tpl = (ROOT / "src" / "dashboard_template.html").read_text(encoding="utf-8")
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print("wrote", out, f"({out.stat().st_size / 1024:.0f} KB), {len(units)} units, window {meta}")


if __name__ == "__main__":
    main()
