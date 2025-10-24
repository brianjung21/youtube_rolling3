"""
Explore Rolling-1 / Rolling-2 outputs from kpop_rolling_gathering.py
- Reads:
    - data/yt_brand_daily_panel.csv (Rolling-1 daily, 7-day window stamped as_of_date_utc)
    - data/yt_brand_roll7_daily.csv (Rolling-2 rollup, one row/brand per report_date_utc)
    - data/yt_video_registry.csv (optional drilldown - not used in first pass)
    - data/yt_video_stats_daily.csv (optional drilldown - not used in first pass)
- Let's you pick an as_of_date_utc (T), a metric, and brands to plot
- Shows: 7-day daily line chart for [T-6..T], and roll-7 bar table for the same T.
- Optional expander: compare roll-7 across available T's for the selected brands
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

DATA_DIR = (Path(__file__).parent / "../data").resolve()
PANEL_FILE = "yt_brand_daily_panel.csv"
ROLL7_FILE = "yt_brand_roll7_daily.csv"

METRIC_LABELS = {
    "video_mentions": "Video mentions",
    "views": "Views",
    "likes": "Likes",
    "comments": "Comments"
}
DEFAULT_TOPN = 8

# Brands to hide temporarily (case-insensitive)
EXCLUDED_BRANDS = {"sempio"}


def find_file(fname: str) -> Path:
    p = DATA_DIR / fname
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find {fname} in: {DATA_DIR}")


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    p = find_file(PANEL_FILE)
    df = pd.read_csv(p, parse_dates=["as_of_date_utc", "date"])
    for c in ["video_mentions", "views", "likes", "comments"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["brand"] = df["brand"].astype(str)
    df = df.sort_values(["as_of_date_utc", "date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_roll7() -> pd.DataFrame:
    p = find_file(ROLL7_FILE)
    df = pd.read_csv(p, parse_dates=["report_date_utc"])
    for c in ["roll7_video_mentions", "roll7_views", "roll7_likes", "roll7_comments"]:
        if c  in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["brand"] = df["brand"].astype(str)
    df = df.sort_values(["report_date_utc", "brand"]).reset_index(drop=True)
    return df


def default_top_brands(panel_T: pd.DataFrame, metric: str, topn: int) -> List[str]:
    if panel_T.empty:
        return []
    totals = panel_T.groupby("brand", as_index=False)[metric].sum().sort_values(metric, ascending=False)
    return totals["brand"].head(topn).tolist()


def to_long(panel_T: pd.DataFrame, brands: List[str], metric: str) -> pd.DataFrame:
    sub = panel_T[panel_T['brand'].isin(brands)].copy()
    sub = sub[["date", "brand", metric]].rename(columns={metric: "value"})
    return sub


def apply_small_rolling(long_df: pd.DataFrame, win: int = 3) -> pd.DataFrame:
    if long_df.empty:
        return long_df.assign(smooth=long_df.get("value", 0))
    out = []
    for b, g in long_df.groupby("brand", as_index=False):
        g = g.sort_values("date").copy()
        g["smooth"] = g["value"].rolling(win, min_periods=1).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def previous_asof(all_asofs: List[pd.Timestamp], sel: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the most recent as_of_date_utc strictly before sel, else None."""
    prior = [d for d in all_asofs if d < sel]
    return max(prior) if prior else None


st.set_page_config(page_title="YouTube Brand Mentions - Rolling Panels", layout="wide")
st.title("YouTube Brand Mentions - Rolling (as-of) Panels")


try:
    panel = load_panel()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Exclude problematic brands globally
if not panel.empty:
    _excl = {b.lower() for b in EXCLUDED_BRANDS}
    panel = panel[~panel["brand"].str.lower().isin(_excl)].copy()

# Try to load roll7, but don’t fail hard if missing (common when <7 days collected)
try:
    roll7 = load_roll7()
except FileNotFoundError:
    roll7 = pd.DataFrame(columns=[
        "report_date_utc", "brand",
        "roll7_video_mentions", "roll7_views", "roll7_likes", "roll7_comments"
    ])

# Also exclude from roll7 if present
if not roll7.empty and "brand" in roll7.columns:
    _excl = {b.lower() for b in EXCLUDED_BRANDS}
    roll7 = roll7[~roll7["brand"].str.lower().isin(_excl)].copy()


with st.sidebar:
    st.header("Controls")
    asofs = sorted(panel["as_of_date_utc"].dt.date.unique())
    if not asofs:
        st.error("No as_of dates found in panel")
        st.stop()
    sel_asof = st.selectbox("As-of date (UTC)", options=asofs, index=len(asofs) - 1)

    metric_key = st.radio(
        "Metric",
        list(METRIC_LABELS.keys()),
        index=0,
        format_func=lambda k: METRIC_LABELS[k]
    )
    engage_metric = st.radio(
        "Engagement metric",
        ["views", "likes", "comments"],
        index=0,
        format_func=lambda k: METRIC_LABELS[k]
    )
    window_days = st.number_input("Window length (days)", min_value=2, max_value=7, value=3)

    panel_T = panel[panel["as_of_date_utc"].dt.date == sel_asof].copy()

    if not panel_T.empty:
        max_d = panel_T["date"].max()
        min_d = max_d - pd.Timedelta(days=window_days - 1)
        panel_T = panel_T[panel_T["date"] >= min_d]

    # Precompute today and past slices regardless of whether today has rows
    if not panel_T.empty:
        max_d = panel_T["date"].max()
        today_df = panel_T[(panel_T["date"] == max_d)].copy()
        past_df = panel_T[(panel_T["date"] < max_d)].copy()
    else:
        today_df = panel_T.copy()
        past_df = panel_T.copy()

    # Previous as_of date (for delta comparisons)
    asof_ts = pd.to_datetime(sel_asof)
    all_asof_ts = sorted(panel["as_of_date_utc"].dt.date.unique())
    prev_asof_date = previous_asof(all_asof_ts, sel_asof)

    # Show effective window for clarity
    if not panel_T.empty:
        _win_min = panel_T["date"].min().date()
        _win_max = panel_T["date"].max().date()
        st.caption(f"Window: {_win_min} → {_win_max} (UTC)")

    brands_all = sorted(panel_T["brand"].unique().tolist())

    # Default brands = Top 5 by Rolling-7 engagement on the chosen as-of date (T)
    defaults: List[str] = []
    panel_T_full = panel[panel["as_of_date_utc"].dt.date == sel_asof].copy()
    if not panel_T_full.empty:
        try:
            max_d_full = panel_T_full["date"].max().floor("D")
            start_d = max_d_full - pd.Timedelta(days=12)  # ensures earliest T-6 has a full 7-day window
            # per-day engagement per brand
            df_e = (
                panel_T_full.assign(eng=lambda x: x["views"] + x["likes"] + x["comments"])\
                            .groupby(["brand","date"], as_index=False)["eng"].sum()
            )
            rows = []
            for b, g in df_e.groupby("brand"):
                g = g.sort_values("date").copy()
                idx = pd.date_range(start=start_d, end=max_d_full, freq="D")
                s = g.set_index("date")["eng"].reindex(idx, fill_value=0)
                r7 = s.rolling(7, min_periods=7).sum()
                val = r7.loc[max_d_full]
                if pd.notna(val):
                    rows.append({"brand": b, "r7_eng": float(val)})
            if rows:
                top = (pd.DataFrame(rows)
                       .sort_values("r7_eng", ascending=False)
                       .loc[:, "brand"].tolist())
                # keep only those in current window’s brand list and take top 5
                defaults = [b for b in top if b in brands_all][:5]
        except Exception:
            defaults = []
    # Fallbacks if needed
    if not defaults:
        if not today_df.empty:
            # fall back to top by mentions for latest publish date in window
            _def_tbl = (
                today_df.groupby("brand", as_index=False)["video_mentions"].sum()
                        .sort_values("video_mentions", ascending=False)
                        .head(5)
            )
            defaults = _def_tbl["brand"].tolist()
        else:
            defaults = default_top_brands(panel_T, "video_mentions", 5)

    selected_brands = st.multiselect("Brands", options=brands_all, default=defaults)

    do_smooth = st.checkbox("Show 3-day smoothing", value=True)
    trim_zeros = st.checkbox("Trim leading all-zero days (current T window)", value=True)

if panel_T.empty:
    st.info("No rows for this as_of date. Try another T.")
    st.stop()
if not selected_brands:
    st.info("Pick at least one brand to plot.")
    st.stop()

# ---------- A) Rolling-7 engagement evolution (first plot) ----------
st.subheader("Rolling-7 engagement evolution (T-6 → T)")
# Compute cohort-correct Rolling-7 by AS-OF date: for each t in [T-6..T],
# sum engagement of videos with publish_date d ∈ [t-6..t], evaluated as-of t.
T_max = pd.to_datetime(sel_asof)
T_min = T_max - pd.Timedelta(days=6)

# Slice only the needed as_of snapshots once
base = panel[(panel["as_of_date_utc"].dt.date >= T_min.date()) &
             (panel["as_of_date_utc"].dt.date <= T_max.date())].copy()
if base.empty:
    st.info("No rows for these as_of snapshots.")
else:
    # Define engagement as views + likes + comments
    base["eng"] = (pd.to_numeric(base.get("views"), errors="coerce").fillna(0) +
                    pd.to_numeric(base.get("likes"), errors="coerce").fillna(0) +
                    pd.to_numeric(base.get("comments"), errors="coerce").fillna(0))

    # Ensure dates are day-level
    base["as_of_d"] = base["as_of_date_utc"].dt.floor("D")
    base["pub_d"] = base["date"].dt.floor("D")

    # Prepare container for long-form results
    rows = []
    t_range = pd.date_range(T_min, T_max, freq="D")

    # Compute, for each t, the sum over publish dates in [t-6..t], evaluated as-of t
    for t in t_range:
        t0 = (t - pd.Timedelta(days=6)).floor("D")
        bt = base[(base["as_of_d"] == t.floor("D")) &
                  (base["pub_d"] >= t0) & (base["pub_d"] <= t.floor("D"))]
        if bt.empty:
            # still emit zeros for selected brands to keep lines continuous
            for b in selected_brands:
                rows.append({"date": t, "rolling7_engagement": 0.0, "brand": b})
            continue
        agg = bt.groupby("brand", as_index=False)["eng"].sum().rename(columns={"eng": "rolling7_engagement"})
        # Keep only selected brands and emit zeros for missing ones
        present = set(agg["brand"].unique())
        for b in selected_brands:
            if b in present:
                val = float(agg.loc[agg["brand"] == b, "rolling7_engagement"].iloc[0])
            else:
                val = 0.0
            rows.append({"date": t, "rolling7_engagement": val, "brand": b})

    r7_long = pd.DataFrame(rows)

    if r7_long.empty or r7_long["rolling7_engagement"].isna().all():
        st.info("Not enough history to compute 7-day rolling engagement for this as_of.")
    else:
        r7_long = r7_long.dropna(subset=["rolling7_engagement"])
        fig_r7e = px.line(
            r7_long.sort_values(["brand","date"]),
            x="date", y="rolling7_engagement", color="brand", markers=True,
            labels={"date":"Publish date", "rolling7_engagement":"Rolling-7 engagement"},
            title=None
        )
        st.plotly_chart(fig_r7e, use_container_width=True)
        # Optional companion table for quick inspection
        tbl = r7_long.pivot_table(index="date", columns="brand", values="rolling7_engagement", aggfunc="first").astype(int)
        tbl_fmt = tbl.applymap(lambda x: f"{x:,}")
        st.dataframe(tbl_fmt, use_container_width=True)


st.caption("Data: yt_brand_daily_panel.csv (Rolling-1) and yt_brand_roll7_daily.csv (Rolling-2). All timestamps UTC.")
