"""Text features at origin t must not change when posts published at/after t are removed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import RES, grid, load_alerts  # noqa: E402
from src.text_features import text_features  # noqa: E402

TEMPLATES = ["БпЛА на Харківщину", "Ракети курсом на Київ", "КАБи на Суми", "Тихо", "Шахеди на Одещину"]


def test_text_features_use_only_past_posts():
    alerts = load_alerts(since="2026-01-01")
    idx, units, uo, _, _ = grid(alerts)
    rng = np.random.default_rng(0)
    n = 4000
    posts = pd.DataFrame({
        "published_at": idx[0] + pd.to_timedelta(rng.integers(0, int((idx[-1] - idx[0]).total_seconds()), n), unit="s"),
        "text": rng.choice(TEMPLATES, n)})
    cut_ts = pd.Timestamp("2026-06-01 12:00", tz="UTC")     # exactly an origin
    # guarantee posts for every template right after the cut, so a look-ahead cannot hide
    extra = pd.DataFrame({"published_at": [cut_ts + pd.Timedelta(minutes=5 * (k + 1)) for k in range(len(TEMPLATES))],
                          "text": TEMPLATES})
    posts = pd.concat([posts, extra], ignore_index=True)
    full = text_features(posts, idx, units, uo)

    erased = text_features(posts[posts["published_at"] < cut_ts], idx, units, uo)
    i_cut = int((cut_ts - idx[0]).total_seconds() // 3600)
    for name, a in erased.items():
        np.testing.assert_allclose(a[: i_cut + 1], full[name][: i_cut + 1], err_msg=f"text leakage in {name}")
    assert sum(float(np.abs(a).sum()) for a in full.values()) > 0
