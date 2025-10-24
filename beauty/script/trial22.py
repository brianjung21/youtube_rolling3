# trial1.py  — YouTube 7-day rolling engagement (brand)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="YouTube — 7-Day Rolling Engagement", layout="wide")

WINDOW_DAYS = 7  # fixed
# Dates to impute if missing
IMPUTE_ASOF_DATES = []
# Publish dates to force-impute (even if present with zero)
IMPUTE_PUBLISH_DATES = []

# ----------------------------- Data loading -----------------------------
@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    candidates = [
        Path("/mnt/data/yt_brand_daily_panel.csv"),
        Path("./data/yt_brand_daily_panel.csv"),
        Path("../data/yt_brand_daily_panel.csv"),
        Path("yt_brand_daily_panel.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        st.error("Could not find yt_brand_daily_panel.csv in ./data, ../data, or current directory.")
        st.stop()

    df = pd.read_csv(path)
    lc = {c.lower(): c for c in df.columns}

    asof_col  = next((lc[c] for c in ["as_of","asof","as_of_date","as_of_date_utc","report_date","report_date_utc"] if c in lc), None)
    date_col  = next((lc[c] for c in ["date","day","publish_date","publish_date_utc","window_date"] if c in lc), None)
    brand_col = next((lc[c] for c in ["brand","group","artist"] if c in lc), None)
    if asof_col is None or date_col is None or brand_col is None:
        st.error("Missing required columns. Need an as-of column, a publish-date column, and a brand column.")
        st.stop()

    df = df.rename(columns={asof_col: "as_of", date_col: "date", brand_col: "brand"})

    # Engagement columns
    def pick_and_rename(std, cands):
        for c in cands:
            if c in lc and lc[c] in df.columns:
                df.rename(columns={lc[c]: std}, inplace=True)
                return
        if std not in df.columns:
            df[std] = 0

    pick_and_rename("views",    ["views","viewcount","view_count"])
    pick_and_rename("likes",    ["likes","likecount","like_count"])
    pick_and_rename("comments", ["comments","commentcount","comment_count"])

    # Types
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df["date"]  = pd.to_datetime(df["date"],  errors="coerce")
    df["brand"] = df["brand"].astype(str)
    for m in ["views","likes","comments"]:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

    df = df.dropna(subset=["as_of","date"]).reset_index(drop=True)
    return df

# ----------------------------- Core calculations -----------------------------
def _brand_asof_publish_totals(panel: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Build per-(as_of_date, publish_date) totals for a brand.
    Returns columns:
      as_of_date (date), publish_date (date),
      views (float), likes (float), comments (float), eng (float)
    where eng = views + likes + comments.

    Also imputes missing as_of dates listed in IMPUTE_ASOF_DATES by generating synthetic
    publish-day totals close to the brand's average scale around the gap (stored in eng).
    """
    df = panel[panel["brand"] == brand].copy()
    if df.empty:
        return pd.DataFrame(columns=["as_of_date","publish_date","eng"])

    df["as_of_date"]   = df["as_of"].dt.date
    df["publish_date"] = df["date"].dt.date  # strip time
    df["eng"] = df[["likes","comments"]].sum(axis=1)

    # Aggregate to per-(as_of_date, publish_date)
    out = (
        df.groupby(["as_of_date","publish_date"], as_index=False)[["views","likes","comments","eng"]]
          .sum()
          .sort_values(["as_of_date","publish_date"])
    )

    # ---------- Impute / repair target as_of dates (e.g., 2025-09-14) ----------
    if not out.empty:
        uniq_asofs = set(out["as_of_date"].unique().tolist())
        for T0 in IMPUTE_ASOF_DATES:
            # Always build the 7 publish dates for this T0
            win_min = (pd.to_datetime(T0) - pd.Timedelta(days=WINDOW_DAYS-1)).date()
            window_days = pd.date_range(pd.to_datetime(win_min), pd.to_datetime(T0), freq="D").date

            need_impute = False
            if T0 not in uniq_asofs:
                need_impute = True
            else:
                # Exists, but check if this T0 has usable data (>=1 positive row within window)
                exist_rows = out[(out["as_of_date"] == T0) & (out["publish_date"].isin(window_days))]
                if exist_rows.empty or float(exist_rows["eng"].sum()) == 0.0:
                    need_impute = True

            if need_impute:
                # Remove any placeholder/zero rows for T0 to avoid double counting
                out = out[out["as_of_date"] != T0]

                # Find nearest available as_of to estimate scale
                if uniq_asofs:
                    uniq_sorted = sorted(uniq_asofs)
                    nearest_T = min(uniq_sorted, key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(T0)))
                    base_vals = out[(out["as_of_date"] == nearest_T) &
                                    (out["publish_date"].isin(window_days))]["eng"]
                    mu = float(base_vals.mean()) if not base_vals.empty else float(out["eng"].mean() if len(out) else 0.0)
                else:
                    mu = 0.0

                # Brand-stable randomness: seed from brand + date so it doesn't jitter every rerun
                rng = np.random.default_rng(abs(hash((brand, T0))) % (2**32))
                synth_rows = []
                for d in window_days:
                    noise = rng.uniform(0.9, 1.1)  # "close to average scale"
                    val = int(max(0, round(mu * noise)))
                    synth_rows.append({
                        "as_of_date": T0,
                        "publish_date": d,
                        "views": 0,
                        "likes": 0,
                        "comments": 0,
                        "eng": val
                    })
                out = pd.concat([out, pd.DataFrame(synth_rows)], ignore_index=True)
                uniq_asofs.add(T0)

        # Ensure sorted after possible appends
        out = out.sort_values(["as_of_date","publish_date"]).reset_index(drop=True)

    return out


# ---------------------- Helper: window contributions (shared logic) ----------------------
def _window_contributions(panel: pd.DataFrame, brand: str, T_date, metric: str = "eng") -> tuple[pd.DataFrame, int]:
    """Return (contrib_df, total) for brand at as_of T_date for the chosen metric.
    metric ∈ {"eng","views"}.
    contrib_df has exactly 7 rows with columns: publish_date (datetime64[ns]), engagement (int).
    Imputation logic and RNG seeding are unified here so tables and chart match exactly.
    """
    ap = _brand_asof_publish_totals(panel, brand)
    T_date = pd.to_datetime(T_date).date()
    metric_col = "eng" if metric == "eng" else "views"

    # Build empty 7-day shell if brand has no data
    win_min = (pd.to_datetime(T_date) - pd.Timedelta(days=WINDOW_DAYS-1)).date()
    base_days = pd.date_range(pd.to_datetime(win_min), pd.to_datetime(T_date), freq="D")

    if ap.empty:
        contrib = pd.DataFrame({"publish_date": base_days, "engagement": 0}).astype({"engagement": int})
        return contrib, 0

    # rows at this as_of and within window
    cur = ap[(ap["as_of_date"] == T_date) &
             (ap["publish_date"] >= win_min) & (ap["publish_date"] <= T_date)][["publish_date", metric_col]].copy()
    cur.rename(columns={metric_col: "eng"}, inplace=True)

    # Force specific publish dates to be treated as missing
    if IMPUTE_PUBLISH_DATES:
        cur.loc[cur["publish_date"].isin(IMPUTE_PUBLISH_DATES), "eng"] = np.nan

    # Aggregate per day
    agg = cur.groupby("publish_date", as_index=False)["eng"].sum()

    # Identify days to impute: missing, zero/NaN, or forced
    present_days = set(agg["publish_date"].tolist())
    zero_or_nan_days = set(agg.loc[(agg["eng"].isna()) | (agg["eng"] <= 0), "publish_date"].tolist())
    forced_days = set(IMPUTE_PUBLISH_DATES)
    window_days = list(pd.to_datetime(base_days).date)
    missing_days = [d for d in window_days if (d not in present_days) or (d in zero_or_nan_days) or (d in forced_days)]

    if missing_days:
        mu = float(agg["eng"].mean())
        if not np.isfinite(mu) or mu == 0.0:
            mu = float(ap[metric_col].mean()) if len(ap) else 0.0
        # Single RNG seed shared across usages so chart & tables match exactly
        rng = np.random.default_rng(abs(hash(("impute_shared", brand, T_date))) % (2**32))
        add_rows = []
        for d in missing_days:
            val = int(max(0, round(mu * rng.uniform(0.9, 1.1))))
            add_rows.append({"publish_date": d, "eng": val})
        if add_rows:
            cur = pd.concat([cur, pd.DataFrame(add_rows)], ignore_index=True)

    # Final per-day contributions in window order
    per_day = cur.groupby("publish_date", as_index=False)["eng"].sum()
    per_day["publish_date"] = pd.to_datetime(per_day["publish_date"])  # normalize dtype
    contrib = pd.DataFrame({"publish_date": base_days}).merge(per_day, on="publish_date", how="left").fillna({"eng": 0})
    contrib.rename(columns={"eng": "engagement"}, inplace=True)
    contrib["engagement"] = contrib["engagement"].round().astype(int)

    total = int(contrib["engagement"].sum())
    return contrib, total

def compute_rolling_series(panel: pd.DataFrame, brands: list[str], metric: str = "eng") -> pd.DataFrame:
    """
    For each brand and each as_of date T:
      1) take per-publish-date totals (as of T),
      2) sum those totals for publish_date in [T-6, T].
    Returns long frame with:
      - as_of_date (date)
      - brand (str)
      - rolling_sum (int): 7-day rolling total
      - rolling_avg (float): 7-day moving average (rolling_sum / 7)
    """
    frames = []
    for b in brands:
        ap = _brand_asof_publish_totals(panel, b)
        if ap.empty:
            continue
        Ts = sorted(ap["as_of_date"].unique())
        rows = []
        for T in Ts:
            _, total = _window_contributions(panel, b, T, metric=metric)
            rows.append({
                "as_of_date": T,
                "brand": b,
                "rolling_sum": int(total),
                "rolling_avg": float(total) / 7.0
            })
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["as_of_date","brand","rolling_sum","rolling_avg"])
    return pd.concat(frames, ignore_index=True)


def table_publish_window(panel: pd.DataFrame, brand: str, T_date, metric: str = "eng") -> pd.DataFrame:
    contrib, _ = _window_contributions(panel, brand, T_date, metric=metric)
    return contrib

# ----------------------------- UI: Sidebar -----------------------------

panel = load_panel()
# Filter: disregard all data before 2025-09-21 (methodology reset)
CUTOFF_DATE = pd.to_datetime("2025-09-21").date()
panel = panel[panel["as_of"].dt.date >= CUTOFF_DATE].reset_index(drop=True)

with st.sidebar:
    line_metric = st.radio(
        "Line metric / 선 그래프 지표",
        options=["Total view counts", "Total engagement (likes+comments)"],
        index=0,  # default to views
        help="Choose whether the line shows 7-day totals of views or engagement (likes+comments). / 7일 합계를 조회수 또는 참여(좋아요+댓글)로 표시합니다."
    )
    base_metric = "eng" if line_metric.startswith("Total engagement") else "views"
    st.header("Controls")
    lang = st.radio("Language / 언어", ["English", "한국어"], index=0)
    # As-of selector (hidden by default). Uses latest date unless shown.
    asof_dates = sorted(panel["as_of"].dt.date.unique())
    if not asof_dates:
        st.stop()
    show_asof = st.checkbox("Show as-of date picker / 기준일 선택 보이기", value=False)
    if show_asof:
        sel_asof = st.selectbox(
            "As-of date / 기준일",
            options=asof_dates,
            index=len(asof_dates)-1,
            format_func=lambda d: d.isoformat(),
        )
    else:
        sel_asof = asof_dates[-1]

    # All brands in dataset
    all_brands = sorted(panel["brand"].unique().tolist())

    # Default brands: Top 5 by TOTAL rolling 7-day engagement (rolling_sum) on the selected as-of date
    # Compute rolling series for all brands once, then snapshot at sel_asof
    series_all_for_defaults = compute_rolling_series(panel, all_brands, metric=base_metric)
    if series_all_for_defaults.empty:
        default_brands = all_brands[:5]
    else:
        target_date = pd.to_datetime(sel_asof).date()
        snap = series_all_for_defaults[
            series_all_for_defaults["as_of_date"] == target_date
        ].sort_values("rolling_sum", ascending=False)
        # Fallbacks in case of ties/empties
        if snap.empty:
            default_brands = all_brands[:5]
        else:
            default_brands = snap["brand"].head(5).tolist()
            if len(default_brands) < min(5, len(all_brands)):
                # pad deterministically with remaining brands alphabetically
                pad = [b for b in all_brands if b not in default_brands]
                default_brands += pad[: max(0, 5 - len(default_brands))]

    # Persist brand selection across reruns using session_state (robust/simple)
    if "sel_brands" not in st.session_state:
        st.session_state.sel_brands = default_brands
    if "confirmed_brands" not in st.session_state:
        st.session_state.confirmed_brands = st.session_state.sel_brands

    # Brand selection form: changes only applied on confirmation
    with st.form("brand_form", clear_on_submit=False):
        sel_brands = st.multiselect(
            "Brands / 브랜드",
            options=all_brands,
            key="sel_brands",
        )
        confirm_clicked = st.form_submit_button("Confirm brands / 브랜드 확정")
        if confirm_clicked:
            if st.session_state.sel_brands:
                st.session_state.confirmed_brands = st.session_state.sel_brands
            else:
                st.warning("Please select at least one brand before confirming. / 최소 한 개의 브랜드를 선택하세요.")

    # Show which brands are currently applied
    applied = st.session_state.get("confirmed_brands", [])
    if applied:
        st.caption("Applied brands: " + ", ".join(applied))

    # Caption explaining default brand selection
    st.caption(
        "**Default brands:** Top 5 by total rolling 7-day engagement on the selected as-of date.  \
        **기본 브랜드:** 선택한 기준일에서 총 7일 롤링 참여도가 가장 높은 상위 5개 브랜드입니다."
    )

    # Brand used for tables (verification + breakdown): auto-select brand with the most views today (as-of)
    sel_asof_date = pd.to_datetime(sel_asof).date()
    series_views_all = compute_rolling_series(panel, all_brands, metric="views")
    confirmed_brands = st.session_state.confirmed_brands
    if not series_views_all.empty:
        snap_views = series_views_all[series_views_all["as_of_date"] == sel_asof_date]
        if not snap_views.empty:
            # Pick the brand with highest 7-day total views (rolling_sum) as of selected date
            top_row = snap_views.sort_values(["rolling_sum", "brand"], ascending=[False, True]).iloc[0]
            table_brand = str(top_row["brand"])
        else:
            # Fallback: use first confirmed brand if available
            table_brand = confirmed_brands[0] if confirmed_brands else (all_brands[0] if all_brands else None)
    else:
        table_brand = confirmed_brands[0] if confirmed_brands else (all_brands[0] if all_brands else None)
    # Let the user know which brand was auto-selected
    st.caption(f"Tables auto-select the brand with the most views (7-day total) on {sel_asof_date}: **{table_brand}**" if table_brand else "No brand available for tables.")


    # Share-of-voice toggle
    sov_mode = st.checkbox(
        "View as share-of-voice (%) / 점유율 보기 (%)",
        value=False,
        help=(
            "Plot each brand's share of the 7-day total as a percentage of the selected brands for each day.\n"
            "선택한 지표(7일 합계)에 대해, 선택된 브랜드들의 합 대비 각 브랜드의 비율(%)을 일자별로 표시합니다."
        ),
    )
    # SoV explanation (bilingual)
    if sov_mode:
        if lang == "English":
            st.caption(
                "**Share of voice (SoV)** shows each brand's **percentage of total engagement** among the *selected brands* for each day. "
                "Values sum to **100% per day** across selected brands."
            )
        else:
            st.caption(
                "**점유율(SoV)**은 선택한 브랜드들 중에서 **하루 총 참여** 대비 각 브랜드가 차지한 **비율(%)**을 의미합니다. "
                "하루 기준 전체 합은 **항상 100%**입니다."
            )








    # Bar chart race toggle
    show_race_toggle = st.checkbox(
        "Show Rolling Rank (Bar Chart Race) / 순위 변동 차트",
        value=False,
        help=(
            "Animate top brands by the selected metric's 7-day total across days.\n"
            "선택한 지표의 7일 합계를 기준으로, 일자별 상위 브랜드 순위 변화를 애니메이션으로 보여줍니다."
        ),
    )
    topn_race = st.slider(
        "Top N for race / 순위 차트 표시 개수",
        min_value=5, max_value=20, value=10, step=1,
        help="Each day, show only the top-N brands in the race to keep it readable. / 매일 상위 N개 브랜드만 표시합니다.",
    )

    # Use all brands in race
    race_use_all = st.checkbox(
        "Use all brands for race / 전체 브랜드 사용",
        value=True,
        help="If checked, the bar chart race will rank all brands in the dataset, not just the selected ones. / 체크하면 선택 브랜드뿐 아니라 전체 브랜드를 대상으로 순위를 계산합니다."
    )

# ----------------------------- Headings -----------------------------
if lang == "English":
    st.title("YouTube — 7-Day Rolling Engagement")
    st.caption(f"Window fixed at 7 days. Current window: "
               f"{(pd.to_datetime(sel_asof) - pd.Timedelta(days=WINDOW_DAYS-1)):%Y-%m-%d} → {pd.to_datetime(sel_asof):%Y-%m-%d} (UTC)")
else:
    st.title("YouTube — 7일 롤링 참여 지표")
    st.caption(f"윈도우는 7일로 고정되어 있습니다. 현재 윈도우: "
               f"{(pd.to_datetime(sel_asof) - pd.Timedelta(days=WINDOW_DAYS-1)):%Y-%m-%d} → {pd.to_datetime(sel_asof):%Y-%m-%d} (UTC)")

# ----------------------------- Plot: multi-brand rolling -----------------------------
st.subheader("Rolling engagement over time (7-day total)" if lang == "English" else "시간에 따른 롤링 참여 (7일 합계)")

with st.expander("ℹ️ Details / 설명", expanded=False):
    if lang == "English":
        st.markdown("""
**What**  
For each as-of date **T**, we compute a brand’s **7-day rolling total engagement** by first aggregating **each publish day’s total (as of T)** and then summing those **seven daily totals** for **[T−6 … T]**.

**How**  
At every **T**, build a table of *(publish_date → total engagement at T)*, then take the sum over the last seven publish dates.

**Meaning**  
The line shows how a brand’s 7-day engagement stack (the seven daily windows) evolves day by day. It isn’t a running accumulation across time; it’s a fresh 7-day window at each T.

**Implications & Actions**  
Use this to spot momentum inflections, compare brands on an apples-to-apples rolling basis, and avoid distortions from irregular posting schedules.
""")
    else:
        st.markdown("""
**무엇을 보여주나요?**  
각 **기준일 T**마다 **게시일별(T 시점 기준) 참여 합**을 먼저 계산한 뒤, 그중 **최근 7일([T−6…T])**을 더해 **7일 롤링 참여**를 만듭니다.

**어떻게 계산하나요?**  
매 **T**에서 *(게시일 → T 시점 참여 합)* 표를 만든 뒤, 그 표에서 최근 7개 게시일만 합산합니다.

**어떻게 해석하나요?**  
이 선형 차트는 누적 합계가 아니라 **매일 새로 산출되는 7일 창의 합**입니다.

**활용 팁**  
업로드 패턴 차이에 따른 왜곡을 줄이고, 브랜드 간 **모멘텀**을 동일 기준으로 비교하며 변곡점을 쉽게 찾을 수 있습니다.
""")

if not confirmed_brands:
    st.info("Select at least one brand in the sidebar." if lang == "English" else "사이드바에서 최소 한 개의 브랜드를 선택하세요.")
else:
    series = compute_rolling_series(panel, confirmed_brands, metric=base_metric)
    # Restrict FIRST plot to the last 7 days ending at the selected as-of date
    latest_date = pd.to_datetime(sel_asof).date()
    plot_window_start = (pd.to_datetime(latest_date) - pd.Timedelta(days=WINDOW_DAYS-1)).date()
    # Filter series to only include as_of_date >= 2025-10-18 for the first line plot
    min_plot_date = pd.to_datetime("2025-10-18").date()
    series_plot = series[
        (series["as_of_date"] >= plot_window_start) &
        (series["as_of_date"] <= latest_date) &
        (series["as_of_date"] >= min_plot_date)
    ].copy()
    if series.empty:
        st.info("No data available for the selected brands." if lang == "English" else "선택한 브랜드의 데이터가 없습니다.")
    else:
        y_col = "rolling_sum"

        if sov_mode:
            # Compute share-of-voice (%) per day across the *selected brands*
            plot_series = series_plot.copy()
            plot_series["total_day"] = plot_series.groupby("as_of_date")[y_col].transform("sum")
            plot_series["sov_pct"] = np.where(
                plot_series["total_day"] > 0,
                (plot_series[y_col] / plot_series["total_day"]) * 100.0,
                np.nan,
            )
            # --- Ensure x-axis is datetime for date axis ---
            plot_series["asof_dt"] = pd.to_datetime(plot_series["as_of_date"])
            x_col = "asof_dt"
            y_plot = "sov_pct"
            y_label_en = "Share of voice (%)"
            y_label_ko = "점유율 (%)"
            fig = px.area(
                plot_series,
                x=x_col,
                y=y_plot,
                color="brand",
                labels={x_col: "As-of date / 기준일", y_plot: f"{y_label_en} / {y_label_ko}"},
                groupnorm=None,
            )
            fig.update_yaxes(range=[0, 100])
            fig.update_xaxes(type="date")
            # Force one tick per visible date (avoid duplicate labels)
            tick_vals = sorted(plot_series["asof_dt"].dropna().unique().tolist())
            tick_text = [pd.to_datetime(v).strftime("%Y-%m-%d") for v in tick_vals]
            fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text)
        else:
            # Absolute values only (normalization removed)
            plot_series = series_plot.copy()
            # --- Ensure x-axis is datetime for date axis ---
            plot_series["asof_dt"] = pd.to_datetime(plot_series["as_of_date"])
            x_col = "asof_dt"
            y_plot = y_col
            metric_label_en = "7-day total engagement" if base_metric == "eng" else "7-day total views"
            metric_label_ko = "7일 합계 참여" if base_metric == "eng" else "7일 합계 조회수"
            y_label_en = metric_label_en
            y_label_ko = metric_label_ko
            fig = px.line(
                plot_series, x=x_col, y=y_plot, color="brand", markers=True,
                labels={x_col: "As-of date / 기준일", y_plot: f"{y_label_en} / {y_label_ko}"},
            )
            fig.update_xaxes(type="date")
            # Force one tick per visible date (avoid duplicate labels)
            tick_vals = sorted(plot_series["asof_dt"].dropna().unique().tolist())
            tick_text = [pd.to_datetime(v).strftime("%Y-%m-%d") for v in tick_vals]
            fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text)
        fig.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig, use_container_width=True)

        if sov_mode:
            # Current-day share-of-voice breakdown at the selected as-of date
            cur = series[series["as_of_date"] == pd.to_datetime(sel_asof).date()].copy()
            if not cur.empty:
                cur_total = cur[y_col].sum()
                cur["share"] = np.where(cur_total > 0, (cur[y_col] / cur_total) * 100.0, np.nan)
                if lang == "English":
                    st.subheader(f"Share of voice — {sel_asof}")
                else:
                    st.subheader(f"점유율 — {sel_asof}")
                st.bar_chart(cur.set_index("brand")["share"])




        # ----------------------------- Bar chart race (toggle) -----------------------------
        if show_race_toggle:
            # Use the selected metric before SoV/normalization (y_col is rolling_sum or rolling_avg)
            if race_use_all:
                race_src = compute_rolling_series(panel, sorted(panel["brand"].unique().tolist()), metric=base_metric)
            else:
                race_src = series[series["brand"].isin(confirmed_brands)].copy()
            # Restrict race frames to start from 2025-10-18 onwards
            min_race_date = pd.to_datetime("2025-10-18").date()
            race_src = race_src[race_src["as_of_date"] >= min_race_date].copy()
            if race_src.empty:
                st.info("No data for the selected brands to render a race chart." if lang == "English" else "선택한 브랜드의 데이터가 없어 순위 차트를 표시할 수 없습니다.")
            else:
                # For each day, keep top N brands by the chosen metric
                race_src = race_src.sort_values(["as_of_date", y_col], ascending=[True, False])
                # Strictly enforce Top-N per frame regardless of prior traces
                race_df = (
                    race_src.groupby("as_of_date", as_index=False, group_keys=False)
                            .head(int(topn_race))
                            .copy()
                )
                # Optional: provide an explicit rank for ordering within each frame
                race_df["rank"] = race_df.groupby("as_of_date")[y_col].rank(method="first", ascending=False)

                # Stringify dates for animation frames
                race_df["asof_str"] = pd.to_datetime(race_df["as_of_date"]).dt.strftime("%Y-%m-%d")

                # Guard against all-zero frames
                vmax = max(1.0, float(race_df[y_col].max() or 1.0))

                if lang == "English":
                    st.subheader("Rolling rank — bar chart race (Top N)")
                    st.caption("Animated leaderboard by selected metric across days. Use the Play ▶ control under the chart.")
                else:
                    st.subheader("순위 변동 — 바 차트 레이스 (상위 N)")
                    st.caption("일자별 선택 지표 기준의 애니메이션 순위표입니다. 차트 하단의 ▶ 버튼으로 재생하세요.")

                race_fig = px.bar(
                    race_df,
                    x=y_col, y="brand",
                    color="brand",
                    orientation="h",
                    animation_frame="asof_str",
                    animation_group="brand",
                    range_x=[0, vmax * 1.1],
                    labels={
                        y_col: "7-day total / 7일 합계",
                        "brand": "Brand / 브랜드",
                    },
                )
                race_fig.update_layout(showlegend=False)
                race_fig.update_yaxes(autorange="reversed")  # largest on top
                # Slow down the animation playback for readability
                try:
                    # Extend the play button's frame/transition durations
                    race_fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200  # ms per frame
                    race_fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500  # ms transition
                    # Also slow slider transitions
                    if race_fig.layout.sliders and len(race_fig.layout.sliders) > 0:
                        race_fig.layout.sliders[0]["transition"]["duration"] = 500
                except Exception:
                    # If plotly structure changes, fail silently; the chart will still render
                    pass
                st.plotly_chart(race_fig, use_container_width=True)


        # ----------------------------- Momentum leaderboard (DoD change in 7DMA; entire brand group) -----------------------------
        # Build over **all brands** so movers reflect the whole universe, not just the current selection
        series_all = compute_rolling_series(panel, sorted(panel["brand"].unique().tolist()), metric=base_metric)
        series_sorted = series_all.sort_values(["brand", "as_of_date"]).copy()

        def _add_dod(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            # Day-over-day change: compare to the previous day (shift by 1)
            g["dod_pct"] = (g["rolling_sum"] / g["rolling_sum"].shift(1)) - 1.0
            return g

        mom = series_sorted.groupby("brand", as_index=False).apply(_add_dod).reset_index(drop=True)

        # Filter momentum leaderboard to only include rows where as_of_date >= 2025-10-18
        mom = mom[mom["as_of_date"] >= pd.to_datetime("2025-10-18").date()]

        # Pick the selected as-of date
        sel_asof_date = pd.to_datetime(sel_asof).date()
        latest_mom = mom[mom["as_of_date"] == sel_asof_date][["brand", "rolling_sum", "dod_pct"]].copy()
        # Sort by DoD change descending
        latest_mom = latest_mom.sort_values("dod_pct", ascending=False)

        # Format for display
        fmt = latest_mom.copy()
        fmt["rolling_sum"] = fmt["rolling_sum"].map(lambda x: "-" if pd.isna(x) else f"{int(round(x)):,}")

        def _fmt_pct(x):
            if pd.isna(x) or not np.isfinite(x):
                return "-"
            return f"{x*100:+.1f}%"

        fmt["dod_pct"] = fmt["dod_pct"].map(_fmt_pct)

        # === Momentum: diverging bar chart (DoD % of 7-day total), with summaries ===
        # Build enhanced snapshot with totals
        snap_raw = latest_mom.copy()
        snap_raw["today_total"] = snap_raw["rolling_sum"].astype(float)
        snap_raw["yday_total"] = np.where(
            (snap_raw["dod_pct"].notna()) & ((snap_raw["dod_pct"] > -1.0) | (snap_raw["today_total"] == 0)),
            snap_raw["today_total"] / (1.0 + snap_raw["dod_pct"]),
            np.nan,
        )
        snap_raw["abs_change"] = snap_raw["today_total"] - snap_raw["yday_total"]

        # Label brands; highlight user-selected brands with a star
        snap_raw["brand_label"] = snap_raw["brand"].astype(str)
        if confirmed_brands:
            snap_raw.loc[snap_raw["brand"].isin(confirmed_brands), "brand_label"] = snap_raw["brand_label"] + " ★"

        # Optional floor to suppress tiny-base noise (kept permissive by default)
        FLOOR = 0  # set to e.g. 10000 to hide micro brands
        snap_plot = snap_raw[(snap_raw["today_total"].fillna(0) >= FLOOR) & snap_raw["dod_pct"].notna()].copy()

        # Prepare values for plotting (% points)
        snap_plot["dod_pct_display"] = snap_plot["dod_pct"] * 100.0

        # One-line summaries
        T_date = pd.to_datetime(sel_asof).date()
        T_prev = (pd.to_datetime(sel_asof) - pd.Timedelta(days=1)).date()
        c1, c2, c3, c4 = st.columns(4)
        if not snap_plot.empty:
            idx_rise_pct = snap_plot["dod_pct"].idxmax()
            idx_fall_pct = snap_plot["dod_pct"].idxmin()
            idx_rise_abs = snap_plot["abs_change"].idxmax()
            idx_biggest  = snap_plot["today_total"].idxmax()

            def _fmt_int(x):
                return "-" if pd.isna(x) else f"{int(round(x)):,}"

            with c1:
                b = snap_plot.loc[idx_rise_pct]
                st.metric("Biggest % riser", b["brand"], f"{b['dod_pct']*100:+.1f}%")
            with c2:
                b = snap_plot.loc[idx_fall_pct]
                st.metric("Biggest % decliner", b["brand"], f"{b['dod_pct']*100:+.1f}%")
            with c3:
                b = snap_plot.loc[idx_rise_abs]
                st.metric("Largest ↑ by volume", b["brand"], _fmt_int(b["abs_change"]))
            with c4:
                b = snap_plot.loc[idx_biggest]
                st.metric("Largest 7-day total (T)", b["brand"], _fmt_int(b["today_total"]))

        # Diverging bar chart centered at 0
        if lang == "English":
            st.subheader(f"Momentum — DoD change in 7-day total (T={T_date} vs T−1={T_prev})")
        else:
            st.subheader(f"모멘텀 — 7일 합계 일간 변화 (T={T_date} vs T−1={T_prev})")

        if snap_plot.empty:
            st.info("No momentum data for this date.")
        else:
            vmax = float(np.nanmax(np.abs(snap_plot["dod_pct_display"])) or 1.0)
            vmax = max(5.0, vmax)  # ensure some span
            fig_m = px.bar(
                snap_plot.sort_values("dod_pct_display"),
                x="dod_pct_display",
                y="brand_label",
                orientation="h",
                color="dod_pct_display",
                color_continuous_scale="RdBu_r",
                range_color=[-vmax, vmax],
                labels={
                    "dod_pct_display": "DoD % (7-day total)",
                    "brand_label": "Brand",
                },
                hover_data={
                    "brand": True,
                    "dod_pct_display": ":.1f",
                    "today_total": ":,.0f",
                    "yday_total": ":,.0f",
                    "abs_change": ":+,.0f",
                },
            )
            fig_m.update_layout(coloraxis_colorbar_title="DoD %")
            fig_m.add_vline(x=0, line_width=1, line_dash="dash", line_color="#888")
            st.plotly_chart(fig_m, use_container_width=True)

        # Optional: keep the raw table inside an expander for detail-oriented users
        with st.expander("Details table / 상세 표", expanded=False):
            tbl = snap_raw[["brand","today_total","yday_total","abs_change","dod_pct"]].copy()
            tbl = tbl.rename(columns={
                "brand": "Brand / 브랜드",
                "today_total": "T total (7-day) / T 합계",
                "yday_total": "T−1 total (7-day) / 전일 합계",
                "abs_change": "Δ abs / 절대 변화",
                "dod_pct": "DoD % / 일간 변화(%)",
            })
            # Format percentages for display in dataframe
            tbl["DoD % / 일간 변화(%)"] = (tbl["DoD % / 일간 변화(%)"] * 100.0).map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
            st.dataframe(tbl, use_container_width=True)



        # ----------------------------- Verification table: 7-day sums per as-of (from 2025-09-11) -----------------------------
        if lang == "English":
            st.subheader("Verification: 7-day sum by as-of date")
            st.caption("Brand-specific check: at each T, we sum publish-day totals for [T−6 … T]. This table verifies the 7-day total as used in the chart.")
        else:
            st.subheader("검증: 기준일별 7일 합계")
            st.caption("브랜드별 확인용 표입니다. 각 기준일 T에서 [T−6…T] 게시일 합계를 보여줍니다. 차트에서 사용하는 7일 합계를 그대로 검증합니다.")

        # --- Brand selector for verification table (limited to user's selected brands) ---
        if confirmed_brands:
            # Initialize session-backed selection on first render only
            if "verify_brand" not in st.session_state:
                initial_verify = table_brand if (table_brand in confirmed_brands) else confirmed_brands[0]
                st.session_state.verify_brand = initial_verify
            else:
                # Ensure the stored brand is still part of current selection; if not, reset gracefully
                if st.session_state.verify_brand not in confirmed_brands:
                    st.session_state.verify_brand = table_brand if (table_brand in confirmed_brands) else confirmed_brands[0]

            # Render selectbox without a default (pulls from session_state via key)
            verify_brand = st.selectbox(
                "Brand to verify / 검증할 브랜드",
                options=confirmed_brands,
                key="verify_brand",
            )
        else:
            verify_brand = table_brand

        def _build_debug_table(panel_df: pd.DataFrame, brand: str, start_date_str: str = "2025-09-21") -> pd.DataFrame:
            """Build a per-as_of row verifying the rolling sum, using the SAME imputation
            logic as the line series (compute_rolling_series). This ensures the
            verification table matches the plotted values.
            """
            ap = _brand_asof_publish_totals(panel_df, brand)
            if ap.empty:
                return pd.DataFrame(columns=["as_of_date","window_start","window_end","rolling_sum"])

            start_date = pd.to_datetime(start_date_str).date()
            rows = []
            for T in sorted(ap["as_of_date"].unique()):
                if T < start_date:
                    continue
                win_min = (pd.to_datetime(T) - pd.Timedelta(days=WINDOW_DAYS-1)).date()
                _, total = _window_contributions(panel_df, brand, T, metric=base_metric)
                rows.append({
                    "as_of_date": pd.to_datetime(T),
                    "window_start": pd.to_datetime(win_min),
                    "window_end": pd.to_datetime(T),
                    "rolling_sum": int(total),
                })
            df_out = pd.DataFrame(rows)
            df_out = df_out[df_out["as_of_date"].dt.date >= pd.to_datetime("2025-10-18").date()]
            return df_out

        # Use the chosen brand from the dropdown for both tables
        if verify_brand:
            if lang == "English":
                st.subheader(f"Verification: 7-day sum by as-of date — {verify_brand}")
            else:
                st.subheader(f"검증: 기준일별 7일 합계 — {verify_brand}")
            dbg = _build_debug_table(panel, verify_brand, start_date_str="2025-09-21")
            if not dbg.empty:
                # Display dates without time
                dbg_display = dbg.copy()
                for c in ["as_of_date","window_start","window_end"]:
                    dbg_display[c] = dbg_display[c].dt.date
                # Format rolling_sum with commas
                dbg_display["rolling_sum"] = dbg_display["rolling_sum"].apply(lambda x: "{:,}".format(x))
                st.dataframe(dbg_display, use_container_width=True)
                st.caption("Tip: Hover value on the line = 7-day total. The breakdown table below shows the contributions of each of the seven publish days; their sum equals the hover value (after imputation).")
            else:
                st.info("No data to show in verification table." if lang == "English" else "검증 테이블에 표시할 데이터가 없습니다.")

# ----------------------------- Table: 7-day window breakdown at selected T -----------------------------
# Use the same brand as the verification selector when available
focus_brand = verify_brand if ("verify_brand" in locals() and verify_brand) else table_brand

if lang == "English":
    st.subheader(f"Publish-day breakdown — {focus_brand if focus_brand else '—'} (window [" \
                 f"{(pd.to_datetime(sel_asof)-pd.Timedelta(days=WINDOW_DAYS-1)):%Y-%m-%d} → {pd.to_datetime(sel_asof):%Y-%m-%d}])")
else:
    st.subheader(f"게시일별 분해 — {focus_brand if focus_brand else '—'} (윈도우 [" \
                 f"{(pd.to_datetime(sel_asof)-pd.Timedelta(days=WINDOW_DAYS-1)):%Y-%m-%d} → {pd.to_datetime(sel_asof):%Y-%m-%d}])")

if not confirmed_brands:
    st.stop()

with st.expander("ℹ️ Details / 설명", expanded=False):
    if lang == "English":
        st.markdown(f"""
**What**  
At **{pd.to_datetime(sel_asof).date()}**, show each **publish day D ∈ [{(pd.to_datetime(sel_asof)-pd.Timedelta(days=WINDOW_DAYS-1)).date()} … {pd.to_datetime(sel_asof).date()}]** and its **total engagement at T** for **{focus_brand}**.

**How**  
Fix **T** and aggregate *(likes + comments)* for videos whose **publish_date = D**. Repeat for the seven days in the window.

**Meaning**  
Reveals whether today’s rolling total is driven by very recent uploads or older days still compounding.

**Implications & Actions**  
Separate **fresh spikes** (recent D) from **long-tail compounding** (older D) to plan cadence and judge durability of attention.
""")
    else:
        st.markdown(f"""
**무엇을 보여주나요?**  
**{pd.to_datetime(sel_asof).date()}** 기준으로 **{focus_brand}**의 7일 창 **각 게시일(D)**이 만든 **기여분**을 보여줍니다.

**어떻게 계산하나요?**  
**T**를 고정한 뒤, **게시일=D**인 영상들의 *(좋아요+댓글)*을 합산합니다. 이를 창의 7개 날짜에 대해 반복합니다.

**어떻게 해석하나요?**  
오늘의 7일 합이 **최근 업로드**의 영향인지, **며칠 전 영상의 계속 누적**인지 한눈에 구분할 수 있습니다.

**활용 팁**  
**신규 스파이크(최근 D)**와 **롱테일(이전 D)**을 구분하여 업로드 타이밍과 캠페인 전략을 조정하세요.
""")

table = table_publish_window(panel, focus_brand, sel_asof, metric=base_metric)
table = table.sort_values("publish_date", ascending=True)

# Ensure the displayed publish date has no time component
table_display = table.copy()
table_display["publish_date"] = table_display["publish_date"].dt.date
# Format engagement with commas
table_display["engagement"] = table_display["engagement"].apply(lambda x: "{:,}".format(x))
st.dataframe(
    table_display.rename(columns={
        "publish_date": "Publish date / 게시일",
        "engagement": ("Engagement (as-of T) / 참여(기준일)" if base_metric == "eng" else "Total views (as-of T) / 총 조회수(기준일)")
    }),
    use_container_width=True,
)

# Footer
if lang == "English":
    st.caption("Data: yt_brand_daily_panel.csv. Engagement = likes + comments. Timestamps treated as UTC. Start date filtered at 2025-09-21.")
else:
    st.caption("데이터: yt_brand_daily_panel.csv. 참여 = 좋아요+댓글. 모든 시각은 UTC 기준입니다. 시작일은 2025-09-21로 필터링되었습니다.")