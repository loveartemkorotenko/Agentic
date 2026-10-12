"""Loading official alert history and rasterising it onto a 5-minute grid per raion.

Spatial unit = raion. Since late 2025 the official feed publishes almost only
raion-level alerts, while earlier history is mostly oblast-level. To keep one
consistent definition ("an alert covers this raion"), an oblast-level alert is
expanded to every raion of that oblast, a hromada alert is mapped to its raion.
Kyiv City has no raions and is its own unit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_URL = ("https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
           "main/datasets/official_data_en.csv")
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "official_data_en.csv"
RES = 5  # minutes per grid step; 60 % RES == 0
MIN_RECORDS_PER_UNIT = 20


def download(force: bool = False) -> Path:
    if force or not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(RAW_URL).to_csv(RAW_PATH, index=False)
    return RAW_PATH


def load_alerts(since=None) -> pd.DataFrame:
    """Return rows (unit, oblast, started_at, finished_at) in UTC, one per raion covered."""
    df = pd.read_csv(download(), parse_dates=["started_at", "finished_at"])
    df = df.dropna(subset=["started_at", "finished_at"])
    df = df[df["finished_at"] > df["started_at"]]
    if since is not None:
        df = df[df["finished_at"] >= pd.Timestamp(since, tz="UTC")]

    sub = df[df["level"] != "oblast"]
    counts = sub.groupby(["oblast", "raion"]).size()
    units = counts[counts >= MIN_RECORDS_PER_UNIT].index.to_frame(index=False)
    units["unit"] = units["oblast"] + " / " + units["raion"]
    kyiv = pd.DataFrame({"oblast": ["Kyiv City"], "raion": ["Kyiv City"], "unit": ["Kyiv City"]})
    units = pd.concat([units[units["oblast"] != "Kyiv City"], kyiv], ignore_index=True)

    fine = sub.merge(units, on=["oblast", "raion"])[["unit", "oblast", "started_at", "finished_at"]]
    coarse = df[df["level"] == "oblast"].drop(columns=["raion"]).merge(units, on="oblast")
    coarse = coarse[["unit", "oblast", "started_at", "finished_at"]]
    return pd.concat([fine, coarse], ignore_index=True)


def grid(alerts: pd.DataFrame):
    """Return (index, units, unit_oblast, state[T, K] bool, starts[T, K] bool).

    state[m, k]  -- an alert covers unit k during step m
    starts[m, k] -- a new (merged) alert episode began in step m
    """
    t0 = alerts["started_at"].min().floor("D")
    t1 = alerts["finished_at"].max().ceil("h")
    idx = pd.date_range(t0, t1, freq=f"{RES}min", tz="UTC")
    amap = alerts[["unit", "oblast"]].drop_duplicates().sort_values("unit")
    units = amap["unit"].tolist()
    unit_oblast = amap["oblast"].to_numpy()
    T, K = len(idx), len(units)
    k_of = {u: i for i, u in enumerate(units)}

    k = alerts["unit"].map(k_of).to_numpy()
    s = ((alerts["started_at"] - t0).dt.total_seconds() // (60 * RES)).astype(int).to_numpy()
    e = np.ceil((alerts["finished_at"] - t0).dt.total_seconds() / (60 * RES)).astype(int).to_numpy()
    delta = np.zeros((T + 1, K), dtype=np.int16)
    np.add.at(delta, (s, k), 1)
    np.add.at(delta, (e, k), -1)
    state = np.cumsum(delta[:-1], axis=0, dtype=np.int16) > 0
    prev = np.vstack([np.zeros((1, K), dtype=bool), state[:-1]])
    starts = state & ~prev          # episode starts after merging overlaps
    return idx, units, unit_oblast, state, starts
