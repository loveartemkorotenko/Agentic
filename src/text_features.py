"""Leakage-safe features from channel posts.

Posts are bucketed by hour of *publication*. A feature at origin hour i sums only hours
j < i, i.e. posts published strictly before t. Posts are attributed to oblasts by Ukrainian
place-name stems, and to a threat type by keywords. No model of language, just counts.
"""
import re

import numpy as np
import pandas as pd

from src.features import BPH

OBLAST_PATTERNS = {
    "Cherkaska oblast": r"Черкащин|Черкас",
    "Chernihivska oblast": r"Чернігівщин|Чернігів",
    "Chernivetska oblast": r"Буковин|Чернівц",
    "Dnipropetrovska oblast": r"Дніпропетровщин|Дніпр|Кривий Ріг|Кривому Розі|Павлоград",
    "Donetska oblast": r"Донеччин|Донец|Краматорськ|Слов'янськ",
    "Ivano-Frankivska oblast": r"Прикарпат|Івано-Франківщин|Франківськ",
    "Kharkivska oblast": r"Харківщин|Харків",
    "Khersonska oblast": r"Херсонщин|Херсон",
    "Khmelnytska oblast": r"Хмельниччин|Хмельницьк",
    "Kirovohradska oblast": r"Кіровоградщин|Кропивницьк",
    "Kyivska oblast": r"Київщин",
    "Kyiv City": r"Ки[їє]в(?!щ)",
    "Lvivska oblast": r"Львівщин|Львів",
    "Mykolaivska oblast": r"Миколаївщин|Миколаїв",
    "Odeska oblast": r"Одещин|Одес",
    "Poltavska oblast": r"Полтавщин|Полтав",
    "Rivnenska oblast": r"Рівненщин|Рівн",
    "Sumska oblast": r"Сумщин|Сум(?:и|ах|ам|ами)\b",
    "Ternopilska oblast": r"Тернопільщин|Тернопіл",
    "Vinnytska oblast": r"Вінниччин|Вінниц",
    "Volynska oblast": r"Волин|Луцьк",
    "Zakarpatska oblast": r"Закарпат|Ужгород",
    "Zaporizka oblast": r"Запоріжж|Запорізьк",
    "Zhytomyrska oblast": r"Житомирщин|Житомир",
}
THREATS = {
    "uav": r"БпЛА|дрон|шахед|Shahed|Герань",
    "missile": r"ракет|балістик|Кинджал|Калібр|Іскандер|Х-\d+|крилат",
    "kab": r"КАБ",
}
_OB = {k: re.compile(v, re.I) for k, v in OBLAST_PATTERNS.items()}
_TH = {k: re.compile(v, re.I) for k, v in THREATS.items()}


def hourly_counts(posts: pd.DataFrame, hour_ts: pd.DatetimeIndex, oblasts):
    """Return dict of (Hn, O) arrays: per-oblast mention counts per publication hour, overall and
    per threat type, plus (Hn,) national post counts."""
    Hn = len(hour_ts)
    hi = hour_ts.get_indexer(posts["published_at"].dt.floor("h"))
    ok = hi >= 0
    posts, hi = posts[ok], hi[ok]
    ob_idx = {o: i for i, o in enumerate(oblasts)}
    out = {"any": np.zeros((Hn, len(oblasts)), np.float32)}
    out.update({t: np.zeros((Hn, len(oblasts)), np.float32) for t in _TH})
    nat = np.zeros(Hn, np.float32)
    for text, h in zip(posts["text"].fillna("").to_numpy(), hi):
        nat[h] += 1
        types = [t for t, rx in _TH.items() if rx.search(text)]
        for o, rx in _OB.items():
            if o in ob_idx and rx.search(text):
                out["any"][h, ob_idx[o]] += 1
                for t in types:
                    out[t][h, ob_idx[o]] += 1
    out["national"] = nat
    return out


def text_features(posts: pd.DataFrame, idx, units, unit_oblast):
    """Per-(origin hour, unit) arrays aligned with features.build_dataset."""
    Hn = len(idx) // BPH
    hour_ts = idx[np.arange(Hn) * BPH]
    oblasts = sorted(set(unit_oblast))
    c = hourly_counts(posts, hour_ts, oblasts)
    col = {o: i for i, o in enumerate(oblasts)}
    umap = np.array([col[o] for o in unit_oblast])

    def past_sum(a, w):  # sum of a[i-w .. i-1]; rows < w are partial sums of what exists
        cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0, dtype=np.float64)], axis=0)
        i = np.arange(Hn)
        return (cs[i] - cs[np.maximum(i - w, 0)]).astype(np.float32)

    feats = {}
    for w in (1, 3, 6):
        feats[f"tg_any_{w}h"] = past_sum(c["any"], w)[:, umap]
    for t in _TH:
        feats[f"tg_{t}_3h"] = past_sum(c[t], 3)[:, umap]
    feats["tg_national_3h"] = np.repeat(past_sum(c["national"], 3)[:, None], len(units), 1)
    feats["tg_national_1h"] = np.repeat(past_sum(c["national"], 1)[:, None], len(units), 1)
    # hours since the last post about this oblast (capped at 24), over fully elapsed hours
    last = np.maximum.accumulate(np.where(c["any"] > 0, np.arange(Hn)[:, None], -1), axis=0)
    prev = np.vstack([np.full((1, last.shape[1]), -1), last[:-1]])
    since = np.where(prev < 0, 24, np.minimum(np.arange(Hn)[:, None] - prev, 24)).astype(np.float32)
    feats["tg_hours_since_mention"] = since[:, umap]
    return feats
