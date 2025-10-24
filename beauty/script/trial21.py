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
    Build per-(as_of_date, publish_date) totals for a brand:
      eng = sum(views+likes+comments) for videos published on publish_date, measured at as_of_date.
    Output columns: as_of_date (date), publish_date (date), eng (float)

    Also imputes missing as_of dates listed in IMPUTE_ASOF_DATES by generating synthetic
    publish-day totals close to the brand's average scale around the gap.
    """
    df = panel[panel["brand"] == brand].copy()
    if df.empty:
        return pd.DataFrame(columns=["as_of_date","publish_date","eng"])

    df["as_of_date"]   = df["as_of"].dt.date
    df["publish_date"] = df["date"].dt.date  # strip time
    df["eng"] = df[["views","likes","comments"]].sum(axis=1)

    # Aggregate to per-(as_of_date, publish_date)
    out = (df.groupby(["as_of_date","publish_date"], as_index=False)["eng"].sum()
             .sort_values(["as_of_date","publish_date"]))

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
                    synth_rows.append({"as_of_date": T0, "publish_date": d, "eng": val})

                out = pd.concat([out, pd.DataFrame(synth_rows)], ignore_index=True)
                uniq_asofs.add(T0)

        # Ensure sorted after possible appends
        out = out.sort_values(["as_of_date","publish_date"]).reset_index(drop=True)

    return out


# ---------------------- Helper: window contributions (shared logic) ----------------------
def _window_contributions(panel: pd.DataFrame, brand: str, T_date) -> tuple[pd.DataFrame, int]:
    """Return (contrib_df, total) for brand at as_of T_date.
    contrib_df has exactly 7 rows with columns: publish_date (datetime64[ns]), engagement (int).
    Imputation logic and RNG seeding are unified here so tables and chart match exactly.
    """
    ap = _brand_asof_publish_totals(panel, brand)
    T_date = pd.to_datetime(T_date).date()

    # Build empty 7-day shell if brand has no data
    win_min = (pd.to_datetime(T_date) - pd.Timedelta(days=WINDOW_DAYS-1)).date()
    base_days = pd.date_range(pd.to_datetime(win_min), pd.to_datetime(T_date), freq="D")

    if ap.empty:
        contrib = pd.DataFrame({"publish_date": base_days, "engagement": 0}).astype({"engagement": int})
        return contrib, 0

    # rows at this as_of and within window
    cur = ap[(ap["as_of_date"] == T_date) &
             (ap["publish_date"] >= win_min) & (ap["publish_date"] <= T_date)][["publish_date","eng"]].copy()

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
            mu = float(ap["eng"].mean()) if len(ap) else 0.0
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

def compute_rolling_series(panel: pd.DataFrame, brands: list[str]) -> pd.DataFrame:
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
            _, total = _window_contributions(panel, b, T)
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


def table_publish_window(panel: pd.DataFrame, brand: str, T_date) -> pd.DataFrame:
    """
    For selected as_of T, list each publish date D in [T-6, T] and show the total engagement
    of videos published on D, measured as_of T.
    Output: publish_date (datetime64[ns]), engagement (float)
    """
    contrib, _ = _window_contributions(panel, brand, T_date)
    return contrib

# ----------------------------- UI: Sidebar -----------------------------

panel = load_panel()
# Filter: disregard all data before 2025-09-21 (methodology reset)
CUTOFF_DATE = pd.to_datetime("2025-09-21").date()
panel = panel[panel["as_of"].dt.date >= CUTOFF_DATE].reset_index(drop=True)

with st.sidebar:
    st.header("Controls")
    lang = st.radio("Language / 언어", ["English", "한국어"], index=0)
    # as-of selector FIRST (defaults depend on the selected date)
    asof_dates = sorted(panel["as_of"].dt.date.unique())
    if not asof_dates:
        st.stop()
    sel_asof = st.selectbox(
        "As-of date / 기준일",
        options=asof_dates,
        index=len(asof_dates)-1,
        format_func=lambda d: d.isoformat(),
    )

    # All brands in dataset
    all_brands = sorted(panel["brand"].unique().tolist())

    # Default brands: Top 5 by TOTAL rolling 7-day engagement (rolling_sum) on the selected as-of date
    # Compute rolling series for all brands once, then snapshot at sel_asof
    series_all_for_defaults = compute_rolling_series(panel, all_brands)
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

    # Multiselect of brands uses the computed defaults
    sel_brands = st.multiselect(
        "Brands / 브랜드",
        options=all_brands,
        default=default_brands,
    )
    # Caption explaining default brand selection
    st.caption(
        "**Default brands:** Top 5 by total rolling 7-day engagement on the selected as-of date.  \
        **기본 브랜드:** 선택한 기준일에서 총 7일 롤링 참여도가 가장 높은 상위 5개 브랜드입니다."
    )

    # Brand used for tables (verification + breakdown)
    if sel_brands:
        table_brand = st.selectbox(
            "Brand for tables / 테이블 대상 브랜드", options=sel_brands, index=0
        )
    else:
        table_brand = None

    # Metric toggle: 7-day total vs 7-day moving average
    metric_mode = st.radio(
        "Metric / 지표",
        options=["7-day total", "7-day moving average"],
        index=0,
        horizontal=False,
        help="Choose between the 7-day rolling total or the 7-day moving average (total ÷ 7). / 7일 합계 또는 7일 이동평균(합계를 7로 나눈 값) 중 선택하세요."
    )

    # Brief explanation (bilingual)
    if lang == "English":
        st.caption(
            "**7-day total** sums engagement over the last 7 days. **7-day moving average (7DMA)** divides that sum by 7 to show the average per day over the same window."
        )
    else:
        st.caption(
            "**7일 합계**는 최근 7일의 참여를 모두 더한 값입니다. **7일 이동평균(7DMA)**은 그 합을 7로 나눈 **일별 평균**을 보여줍니다."
        )

    # Share-of-voice toggle
    sov_mode = st.checkbox(
        "View as share-of-voice (%) / 점유율 보기 (%)",
        value=False,
        help=(
            "Plot each brand's share of the selected metric (7-day total or 7DMA) as a percentage of the selected brands for each day.\n"
            "선택한 지표(7일 합계 또는 7DMA)에 대해, 선택된 브랜드들의 합 대비 각 브랜드의 비율(%)을 일자별로 표시합니다."
        ),
    )
    # SoV explanation (bilingual)
    if sov_mode:
        if lang == "English":
            st.caption(
                "**Share of voice (SoV)** shows each brand's **percentage of total engagement** among the *selected brands* for each day. "
                "The base metric follows your toggle (7-day total or 7DMA). Values sum to **100% per day** across selected brands."
            )
        else:
            st.caption(
                "**점유율(SoV)**은 선택한 브랜드들 중에서 **하루 총 참여** 대비 각 브랜드가 차지한 **비율(%)**을 의미합니다. "
                "기준 지표는 상단 토글(7일 합계 또는 7DMA)을 따르며, 하루 기준 전체 합은 **항상 100%**입니다."
            )

    # Composite indices toggle (KEWI/KIM/KICW)
    show_index_toggle = st.checkbox(
        "Show K‑Pop Engagement Indices (KEWI/KIM/KICW) / 지수 보기",
        value=False,
        help=("Display composite indices of the entire market using equal-weight (KEWI), "
              "market-cap style (KIM), and capped-weight (KICW). / "
              "동일가중(KEWI), 시가총액형(KIM), 상한가중(KICW) 방식으로 전체 시장 지수를 표시합니다.")
    )
    cap_pct = st.slider(
        "KICW cap (%) / KICW 상한(%)",
        min_value=5, max_value=50, value=20, step=1,
        help="Maximum single-brand weight per day for the capped index. / 상한가중 지수에서 하루 기준 단일 브랜드의 최대 비중."
    )

    # Normalize toggle
    norm_mode = st.checkbox(
        "Normalize (index = 100 at first date) / 정규화 (초일=100)",
        value=False,
        help=(
            "Rescale each brand's series so that its first available value is 100.\n"
            "브랜드별 첫 값이 100이 되도록 지수를 환산하여 성장/감소 추세를 비교합니다."
        ),
    )

    # Show Top 5 Movers toggle
    show_top5_toggle = st.checkbox(
        "Show Top 5 Movers Today / 오늘의 Top5 변동 보기",
        value=False,
        help=(
            "Display a mini table of the top 5 rising and falling brands by **day-over-day (DoD)** change in 7DMA at the selected as-of date.\n"
            "선택한 기준일에서 7DMA의 **일간 변화율(DoD)**이 가장 큰 상위 5개 브랜드(상승/하락)를 보여줍니다."
        ),
    )

    # Volatility lens toggle
    show_vol_toggle = st.checkbox(
        "Show Volatility Lens / 변동성 보기",
        value=False,
        help=(
            "Plot a scatter of average 7DMA (x-axis) vs the rolling standard deviation of day-over-day returns (y-axis).\n"
            "x축: 평균 7DMA (일별 기준선), y축: 7DMA 일간 수익률의 롤링 표준편차(변동성)."
        ),
    )

    # Correlation heatmap toggle
    show_corr_toggle = st.checkbox(
        "Show Correlation Heatmap / 상관관계 히트맵",
        value=False,
        help=(
            "Compute pairwise correlation across the **selected brands** using the selected metric (7-day total or 7DMA) over the visible date range.\n"
            "선택한 지표(7일 합계 또는 7DMA)로 **선택된 브랜드**들의 일자별 추세 상관관계를 계산해 히트맵으로 표시합니다."
        ),
    )

    # Returns / Cumulative returns toggles
    show_returns_toggle = st.checkbox(
        "Show Returns / 수익률 보기",
        value=False,
        help=(
            "Plot daily percent returns of the selected metric (7DMA or 7-day total) or cumulative returns since the first date.\n"
            "선택 지표(7DMA 또는 7일 합계)의 **일간 수익률(%)** 또는 **기준일부터의 누적 수익률**을 표시합니다."
        ),
    )
    returns_mode = st.radio(
        "Returns mode / 수익률 모드",
        options=["Daily % return", "Cumulative return"],
        index=1,
        help="Choose daily % return or cumulative return since the first available date. / 일간 수익률 또는 기준일부터의 누적 수익률을 선택하세요.",
    )

    # Sharpe-like ratio toggle
    show_sharpe_toggle = st.checkbox(
        "Show Sharpe-like Ratios / 샤프 유사 지표",
        value=False,
        help=(
            "Rank brands by risk-adjusted engagement momentum: mean daily return divided by volatility (std of daily returns).\n"
            "수익률의 평균을 변동성(일간 수익률 표준편차)으로 나눈 값으로 브랜드를 정렬합니다."
        ),
    )
    sharpe_window = st.slider(
        "Window for Sharpe (days) / 샤프 창(일)",
        min_value=7, max_value=60, value=14, step=1,
        help="Rolling window length (in days) to compute mean and std of daily returns. / 일간 수익률의 평균과 표준편차를 계산할 롤링 기간",
    )

    # Bar chart race toggle
    show_race_toggle = st.checkbox(
        "Show Rolling Rank (Bar Chart Race) / 순위 변동 차트",
        value=False,
        help=(
            "Animate top brands by the selected metric across days (7DMA or 7-day total).\n"
            "선택한 지표(7DMA 또는 7일 합계) 기준으로 일자별 상위 브랜드 순위가 변하는 모습을 애니메이션으로 보여줍니다."
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
        value=False,
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
if metric_mode == "7-day moving average":
    st.subheader("Daily engagement over time (7-day moving average)" if lang == "English" else "시간에 따른 일간 참여 (7일 이동평균)")
else:
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

if not sel_brands:
    st.info("Select at least one brand in the sidebar." if lang == "English" else "사이드바에서 최소 한 개의 브랜드를 선택하세요.")
else:
    series = compute_rolling_series(panel, sel_brands)
    # Restrict FIRST plot to the last 7 days ending at the selected as-of date
    latest_date = pd.to_datetime(sel_asof).date()
    plot_window_start = (pd.to_datetime(latest_date) - pd.Timedelta(days=WINDOW_DAYS-1)).date()
    series_plot = series[(series["as_of_date"] >= plot_window_start) & (series["as_of_date"] <= latest_date)].copy()
    if series.empty:
        st.info("No data available for the selected brands." if lang == "English" else "선택한 브랜드의 데이터가 없습니다.")
    else:
        y_col = "rolling_sum" if metric_mode == "7-day total" else "rolling_avg"

        if sov_mode:
            # Compute share-of-voice (%) per day across the *selected brands*
            plot_series = series_plot.copy()
            plot_series["total_day"] = plot_series.groupby("as_of_date")[y_col].transform("sum")
            plot_series["sov_pct"] = np.where(
                plot_series["total_day"] > 0,
                (plot_series[y_col] / plot_series["total_day"]) * 100.0,
                np.nan,
            )
            y_plot = "sov_pct"
            y_label_en = "Share of voice (%)"
            y_label_ko = "점유율 (%)"
            fig = px.area(
                plot_series,
                x="as_of_date",
                y=y_plot,
                color="brand",
                labels={"as_of_date": "As-of date / 기준일", y_plot: f"{y_label_en} / {y_label_ko}"},
                groupnorm=None,
            )
            fig.update_yaxes(range=[0, 100])
        else:
            # Absolute or normalized values
            plot_series = series_plot.copy()
            if norm_mode:
                def to_index100(g):
                    g = g.sort_values("as_of_date").copy()
                    base = g[y_col].replace(0, np.nan).iloc[0]
                    if pd.notna(base) and base != 0:
                        g["norm"] = (g[y_col] / base) * 100
                    else:
                        g["norm"] = np.nan
                    return g
                plot_series = (plot_series.groupby("brand", as_index=False)
                                           .apply(to_index100)
                                           .reset_index(drop=True))
                y_plot = "norm"
                y_label_en = "Index (start=100)"
                y_label_ko = "지수 (초일=100)"
            else:
                y_plot = y_col
                y_label_en = "7-day total engagement" if y_col == "rolling_sum" else "7-day moving average (per day)"
                y_label_ko = "7일 합계 참여" if y_col == "rolling_sum" else "7일 이동평균 (일별)"
            fig = px.line(
                plot_series, x="as_of_date", y=y_plot, color="brand", markers=True,
                labels={"as_of_date": "As-of date / 기준일", y_plot: f"{y_label_en} / {y_label_ko}"},
            )
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

        # ----------------------------- Returns & Cumulative Returns (toggle) -----------------------------
        if show_returns_toggle:
            ret = series.sort_values(["brand", "as_of_date"]).copy()
            # daily % return of selected metric
            ret["ret"] = ret.groupby("brand")[y_col].pct_change()

            if returns_mode == "Daily % return":
                plot_ret = ret.dropna(subset=["ret"]).copy()
                if lang == "English":
                    st.subheader("Daily returns — selected metric")
                    st.caption("Day-over-day % change of the selected metric (7DMA or 7-day total) for the selected brands.")
                else:
                    st.subheader("일간 수익률 — 선택 지표")
                    st.caption("선택한 지표(7DMA 또는 7일 합계)의 전일 대비 변화율(%)을 브랜드별로 표시합니다.")
                fig_ret = px.line(
                    plot_ret, x="as_of_date", y="ret", color="brand", markers=True,
                    labels={"as_of_date": "As-of date / 기준일", "ret": "Return / 수익률(%)"},
                )
                fig_ret.update_yaxes(tickformat=".1%")
                fig_ret.update_xaxes(tickformat="%Y-%m-%d")
                st.plotly_chart(fig_ret, use_container_width=True)
            else:
                # cumulative return since first available date: (1+ret).cumprod()-1 per brand
                ret["cret"] = (
                    ret.groupby("brand")["ret"]
                       .transform(lambda x: (1.0 + x.fillna(0)).cumprod() - 1.0)
                )
                plot_cret = ret.dropna(subset=["cret"]).copy()
                if lang == "English":
                    st.subheader("Cumulative return — since first date")
                    st.caption("Cumulative % change of the selected metric from the first available date for each brand.")
                else:
                    st.subheader("누적 수익률 — 기준일부터")
                    st.caption("브랜드별로 첫 가용 날짜부터의 누적 변화율(%)을 표시합니다.")
                fig_cret = px.line(
                    plot_cret, x="as_of_date", y="cret", color="brand", markers=True,
                    labels={"as_of_date": "As-of date / 기준일", "cret": "Cumulative return / 누적 수익률"},
                )
                fig_cret.update_yaxes(tickformat=".1%")
                fig_cret.update_xaxes(tickformat="%Y-%m-%d")
                st.plotly_chart(fig_cret, use_container_width=True)

        # ----------------------------- Sharpe-like Ratios (toggle) -----------------------------
        if show_sharpe_toggle:
            sharpe_min = max(5, int(np.ceil(sharpe_window * 0.6)))
            # Build returns for selected brands using the currently selected metric
            r = series.sort_values(["brand", "as_of_date"]).copy()
            r["ret"] = r.groupby("brand")[y_col].pct_change()

            # Rolling mean and std of returns per brand
            r["mu"] = (
                r.groupby("brand")["ret"].transform(
                    lambda x: x.rolling(sharpe_window, min_periods=sharpe_min).mean()
                )
            )
            r["sigma"] = (
                r.groupby("brand")["ret"].transform(
                    lambda x: x.rolling(sharpe_window, min_periods=sharpe_min).std()
                )
            )
            r["sharpe_like"] = r["mu"] / r["sigma"]

            # Snapshot at selected as-of date for the selected brands
            snap_date = pd.to_datetime(sel_asof).date()
            snap = r[(r["as_of_date"] == snap_date) & (r["brand"].isin(sel_brands))].copy()
            # Clean infinite/NaN
            snap.replace([np.inf, -np.inf], np.nan, inplace=True)
            snap = snap.dropna(subset=["sharpe_like"])  # require enough data in window

            if snap.empty:
                st.info("Not enough history to compute Sharpe-like ratios for the selected date/brands." if lang == "English" else "선택한 날짜/브랜드에 대해 샤프 유사 지표를 계산할 충분한 이력이 없습니다.")
            else:
                snap = snap.sort_values("sharpe_like", ascending=False)
                out = snap[["brand", "mu", "sigma", "sharpe_like"]].copy()
                # Format percentages for mu and sigma
                def _fmt_pct(x):
                    return "-" if pd.isna(x) or not np.isfinite(x) else f"{x*100:.2f}%"
                out["Mean daily return / 평균 일간 수익률"] = out["mu"].map(_fmt_pct)
                out["Volatility (std) / 변동성(표준편차)"] = out["sigma"].map(_fmt_pct)
                out["Sharpe*"] = out["sharpe_like"].map(lambda x: "-" if pd.isna(x) or not np.isfinite(x) else f"{x:.2f}")
                out = out.drop(columns=["mu", "sigma", "sharpe_like"]).rename(columns={"brand": "Brand / 브랜드"})

                if lang == "English":
                    st.subheader(f"Sharpe-like ratios — window {sharpe_window}d (as of {snap_date})")
                    st.caption("Sharpe* = mean daily return ÷ std of daily returns, using the selected metric. Higher is better (more consistent momentum per unit of volatility). No risk-free rate applied.")
                else:
                    st.subheader(f"샤프 유사 지표 — 창 {sharpe_window}일 (기준일 {snap_date})")
                    st.caption("Sharpe* = 일간 수익률 평균 ÷ 일간 수익률 표준편차. 선택 지표 기준. 값이 높을수록 변동성 대비 모멘텀이 우수합니다. 무위험 수익률은 적용하지 않았습니다.")

                st.dataframe(out, use_container_width=True)

        # ----------------------------- Composite indices: KEWI / KIM / KICW (toggle) -----------------------------
        if show_index_toggle:
            # Build full-universe daily series using the selected metric (y_col)
            series_all_for_idx = compute_rolling_series(panel, sorted(panel["brand"].unique().tolist()))
            if series_all_for_idx.empty:
                st.info("No data available to compute indices." if lang == "English" else "지수를 계산할 데이터가 없습니다.")
            else:
                # For each day, compute:
                # KEWI_raw: equal-weight average of metric across brands present that day
                # KIM_raw: market-cap style total (sum of metric)
                # KICW_raw: capped-weighted average with cap_pct (e.g., 20%)
                mcol = y_col  # 'rolling_sum' or 'rolling_avg'
                cap = cap_pct / 100.0

                def per_day_index(g: pd.DataFrame) -> pd.Series:
                    v = g[mcol].astype(float).values
                    if v.size == 0:
                        return pd.Series({"KEWI_raw": np.nan, "KIM_raw": np.nan, "KICW_raw": np.nan})
                    # Remove negatives if any (shouldn't exist)
                    v = np.where(np.isfinite(v), np.maximum(v, 0.0), np.nan)
                    v = v[~np.isnan(v)]
                    if v.size == 0:
                        return pd.Series({"KEWI_raw": np.nan, "KIM_raw": np.nan, "KICW_raw": np.nan})
                    total = v.sum()
                    n = float(len(v))
                    KEWI_raw = v.mean()
                    KIM_raw = total  # cap-weight analogue; rebase will normalize scale

                    # Capped weights
                    if total <= 0:
                        KICW_raw = np.nan
                    else:
                        w = v / total
                        # Initial cap
                        w_cap = np.minimum(w, cap)
                        capped_mask = (w >= cap)
                        uncapped_mask = ~capped_mask
                        remaining = 1.0 - w_cap.sum()
                        if remaining > 1e-12 and uncapped_mask.any():
                            w_uncapped_orig = w[uncapped_mask]
                            sum_unc = w_uncapped_orig.sum()
                            if sum_unc > 0:
                                w_cap[uncapped_mask] = w_cap[uncapped_mask] + remaining * (w_uncapped_orig / sum_unc)
                            # else: if all weights capped (rare), w_cap already sums to ~1
                        # Weighted average with capped weights (weights sum to ~1)
                        KICW_raw = np.dot(w_cap, v)

                    return pd.Series({"KEWI_raw": KEWI_raw, "KIM_raw": KIM_raw, "KICW_raw": KICW_raw})

                idx_daily = (series_all_for_idx.groupby("as_of_date", as_index=False)
                                               .apply(per_day_index)
                                               .reset_index())

                # Rebase all index levels to 100 at the first available date
                idx_daily = idx_daily.sort_values("as_of_date")
                base_row = idx_daily.iloc[0]
                def rebase(series_vals, base_val):
                    return (series_vals / base_val) * 100.0 if np.isfinite(base_val) and base_val != 0 else np.nan

                idx_daily["KEWI"] = rebase(idx_daily["KEWI_raw"], float(base_row["KEWI_raw"]))
                idx_daily["KIM"]  = rebase(idx_daily["KIM_raw"],  float(base_row["KIM_raw"]))
                idx_daily["KICW"] = rebase(idx_daily["KICW_raw"], float(base_row["KICW_raw"]))

                # Melt for plotting
                plot_idx = idx_daily.melt(id_vars=["as_of_date"], value_vars=["KEWI","KIM","KICW"],
                                          var_name="Index", value_name="Level")

                if lang == "English":
                    st.subheader("K‑Pop Engagement Indices — KEWI (equal‑weight), KIM (cap‑weight), KICW (capped)")
                    st.caption("All indices rebased to 100 on the first date. Metric follows your toggle (7‑day total or 7DMA).")
                else:
                    st.subheader("K‑Pop 참여 지수 — KEWI(동일가중), KIM(시가총액형), KICW(상한가중)")
                    st.caption("모든 지수는 첫 날짜를 100으로 기준화했습니다. 지표 토글(7일 합계/7DMA)을 따릅니다.")

                idx_fig = px.line(
                    plot_idx, x="as_of_date", y="Level", color="Index", markers=True,
                    labels={"as_of_date": "As-of date / 기준일", "Level": "Index level (base=100) / 지수 수준(기준=100)"},
                )
                idx_fig.update_xaxes(tickformat="%Y-%m-%d")
                st.plotly_chart(idx_fig, use_container_width=True)

        # ----------------------------- Bar chart race (toggle) -----------------------------
        if show_race_toggle:
            # Use the selected metric before SoV/normalization (y_col is rolling_sum or rolling_avg)
            if race_use_all:
                race_src = compute_rolling_series(panel, sorted(panel["brand"].unique().tolist()))
            else:
                race_src = series[series["brand"].isin(sel_brands)].copy()
            if race_src.empty:
                st.info("No data for the selected brands to render a race chart." if lang == "English" else "선택한 브랜드의 데이터가 없어 순위 차트를 표시할 수 없습니다.")
            else:
                # For each day, keep top N brands by the chosen metric
                race_src = race_src.sort_values(["as_of_date", y_col], ascending=[True, False])
                race_src["rank"] = race_src.groupby("as_of_date")[y_col].rank(method="first", ascending=False)
                race_df = race_src[race_src["rank"] <= topn_race].copy()

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
                        y_col: ("7-day total" if y_col == "rolling_sum" else "7DMA (per day)") + " / " + ("7일 합계" if y_col == "rolling_sum" else "7DMA (일별)"),
                        "brand": "Brand / 브랜드",
                    },
                )
                race_fig.update_layout(showlegend=False)
                race_fig.update_yaxes(autorange="reversed")  # largest on top
                st.plotly_chart(race_fig, use_container_width=True)

        # ----------------------------- Correlation heatmap (toggle) -----------------------------
        if show_corr_toggle:
            # Build a pivot: rows = as_of_date, cols = brand, values = selected metric (y_col)
            piv = series.pivot(index="as_of_date", columns="brand", values=y_col)
            # Require at least a few overlapping days to stabilize correlation
            corr = piv.corr(min_periods=5)
            # Reorder to match the sidebar brand order for readability
            order = [b for b in sel_brands if b in corr.columns]
            corr = corr.loc[order, order]

            if corr.empty or corr.shape[0] < 2:
                st.info("Not enough overlapping days among selected brands to compute correlations." if lang == "English" else "선택한 브랜드 간 상관관계를 계산하기에 겹치는 날짜가 충분하지 않습니다.")
            else:
                if lang == "English":
                    st.subheader("Correlation heatmap — selected brands")
                    st.caption("Pearson correlation of daily trends using the selected metric. +1 = move together, −1 = move opposite.")
                else:
                    st.subheader("상관관계 히트맵 — 선택된 브랜드")
                    st.caption("선택한 지표를 기준으로 일자별 추세의 피어슨 상관계수를 표시합니다. +1 = 동행, −1 = 반대.")

                heat = px.imshow(
                    corr,
                    labels=dict(color="Correlation / 상관계수"),
                    x=corr.columns,
                    y=corr.index,
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu_r",
                )
                heat.update_xaxes(side="top", tickangle=45)
                st.plotly_chart(heat, use_container_width=True)

        # ----------------------------- Momentum leaderboard (DoD change in 7DMA; entire brand group) -----------------------------
        # Build over **all brands** so movers reflect the whole universe, not just the current selection
        series_all = compute_rolling_series(panel, sorted(panel["brand"].unique().tolist()))
        series_sorted = series_all.sort_values(["brand", "as_of_date"]).copy()

        def _add_dod(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            # Day-over-day change: compare to the previous day (shift by 1)
            g["dod_pct"] = (g["rolling_avg"] / g["rolling_avg"].shift(1)) - 1.0
            return g

        mom = series_sorted.groupby("brand", as_index=False).apply(_add_dod).reset_index(drop=True)

        # Pick the selected as-of date
        sel_asof_date = pd.to_datetime(sel_asof).date()
        latest_mom = mom[mom["as_of_date"] == sel_asof_date][["brand", "rolling_avg", "dod_pct"]].copy()
        # Sort by DoD change descending
        latest_mom = latest_mom.sort_values("dod_pct", ascending=False)

        # Format for display
        fmt = latest_mom.copy()
        fmt["rolling_avg"] = fmt["rolling_avg"].map(lambda x: "-" if pd.isna(x) else f"{int(round(x)):,}")

        def _fmt_pct(x):
            if pd.isna(x) or not np.isfinite(x):
                return "-"
            return f"{x*100:+.1f}%"

        fmt["dod_pct"] = fmt["dod_pct"].map(_fmt_pct)

        if lang == "English":
            st.subheader(f"Momentum leaderboard — DoD change in 7DMA (as of {sel_asof_date})")
            st.caption("Shows which brands' daily baseline (7DMA) is rising or falling **vs yesterday**. DoD = day-over-day.")
        else:
            st.subheader(f"모멘텀 리더보드 — 7DMA 일간 변화율 (기준일 {sel_asof_date})")
            st.caption("브랜드별 **일간 기준선(7DMA)**이 **전일 대비** 얼마나 올랐는지/내렸는지 보여줍니다. DoD = 일간 변화.")

        st.dataframe(
            fmt.rename(columns={
                "brand": "Brand / 브랜드",
                "rolling_avg": "7DMA (per day) / 7일 이동평균(일별)",
                "dod_pct": "DoD change / 일간 변화"
            }),
            use_container_width=True,
        )

        # ----------------------------- Volatility Lens (toggle) -----------------------------
        if show_vol_toggle:
            # Use the full brand universe for a consistent scatter
            vol_df = series_all.copy()
            vol_df = vol_df.sort_values(["brand", "as_of_date"])  # ensure chronological order per brand
            # Day-over-day "returns" of 7DMA
            vol_df["dod_return"] = vol_df.groupby("brand")["rolling_avg"].pct_change()
            # Rolling volatility over 14 days (min 7 to be less sparse)
            vol_df["volatility"] = (
                vol_df.groupby("brand")["dod_return"]
                      .transform(lambda x: x.rolling(14, min_periods=7).std())
            )

            # Aggregate per brand for a stable snapshot: mean 7DMA and mean volatility across available window
            agg = (vol_df.groupby("brand")
                          .agg(avg_eng=("rolling_avg", "mean"),
                               vol=("volatility", "mean"))
                          .reset_index())
            agg = agg.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_eng", "vol"])  # clean

            if lang == "English":
                st.subheader("Volatility lens — Avg daily engagement vs rolling volatility")
                st.caption("x: mean 7DMA (per day). y: rolling std-dev of DoD changes over 14 days. Top-right = big & wild, bottom-right = big & steady.")
            else:
                st.subheader("변동성 뷰 — 평균 일간 참여 vs 롤링 변동성")
                st.caption("x: 평균 7DMA(일별). y: 14일 롤링 기준 7DMA 일간 변화율의 표준편차. 우상단=크고 요동, 우하단=크고 안정.")

            vol_fig = px.scatter(
                agg, x="avg_eng", y="vol", text="brand",
                labels={
                    "avg_eng": "Mean 7DMA / 평균 7DMA",
                    "vol": "Rolling volatility (σ) / 롤링 변동성 (σ)",
                },
            )
            vol_fig.update_traces(textposition="top center")
            st.plotly_chart(vol_fig, use_container_width=True)

        if show_top5_toggle:
            # ----------------------------- Top movers today (mini tables; DoD across entire brand group) -----------------------------
            _tbl = latest_mom.dropna(subset=["dod_pct"]).copy()
            risers = (
                _tbl.sort_values("dod_pct", ascending=False)
                    .head(5)[["brand", "dod_pct"]]
                    .copy()
            )
            decliners = (
                _tbl.sort_values("dod_pct", ascending=True)
                    .head(5)[["brand", "dod_pct"]]
                    .copy()
            )

            def _fmt_pct2(x):
                return "-" if pd.isna(x) or not np.isfinite(x) else f"{x*100:+.1f}%"

            risers["DoD"] = risers["dod_pct"].map(_fmt_pct2)
            decliners["DoD"] = decliners["dod_pct"].map(_fmt_pct2)
            risers = risers.drop(columns=["dod_pct"]).rename(columns={"brand": "Brand / 브랜드"})
            decliners = decliners.drop(columns=["dod_pct"]).rename(columns={"brand": "Brand / 브랜드"})

            c1, c2 = st.columns(2)
            if lang == "English":
                with c1:
                    st.subheader("Top movers today — Risers (DoD)")
                    st.dataframe(risers, use_container_width=True)
                with c2:
                    st.subheader("Top movers today — Decliners (DoD)")
                    st.dataframe(decliners, use_container_width=True)
            else:
                with c1:
                    st.subheader("오늘의 급등 — 일간 변화")
                    st.dataframe(risers, use_container_width=True)
                with c2:
                    st.subheader("오늘의 급락 — 일간 변화")
                    st.dataframe(decliners, use_container_width=True)

        # ----------------------------- Verification table: 7-day sums per as-of (from 2025-09-11) -----------------------------
        if lang == "English":
            st.subheader(f"Verification: 7-day sum by as-of date — {table_brand if table_brand else '—'}")
            st.caption("Brand-specific check: at each T, we sum publish-day totals for [T−6 … T]. This table verifies the 7-day total; if the chart is set to 7DMA, divide by 7 to match.")
        else:
            st.subheader(f"검증: 기준일별 7일 합계 — {table_brand if table_brand else '—'}")
            st.caption("브랜드별 확인용 표입니다. 각 기준일 T에서 [T−6…T] 게시일 합계를 보여줍니다. 그래프가 7일 이동평균(7DMA)일 경우, 이 값을 7로 나누면 일치합니다.")

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
                _, total = _window_contributions(panel_df, brand, T)
                rows.append({
                    "as_of_date": pd.to_datetime(T),
                    "window_start": pd.to_datetime(win_min),
                    "window_end": pd.to_datetime(T),
                    "rolling_sum": int(total),
                })
            return pd.DataFrame(rows)

        # Use the chosen brand from the sidebar for both tables
        if table_brand:
            dbg = _build_debug_table(panel, table_brand, start_date_str="2025-09-21")
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
if lang == "English":
    st.subheader(f"Publish-day breakdown — {table_brand if table_brand else '—'} (window [" \
                 f"{(pd.to_datetime(sel_asof)-pd.Timedelta(days=WINDOW_DAYS-1)):%Y-%m-%d} → {pd.to_datetime(sel_asof):%Y-%m-%d}])")
else:
    st.subheader(f"게시일별 분해 — {table_brand if table_brand else '—'} (윈도우 [" \
                 f"{(pd.to_datetime(sel_asof)-pd.Timedelta(days=WINDOW_DAYS-1)):%Y-%m-%d} → {pd.to_datetime(sel_asof):%Y-%m-%d}])")

if not sel_brands:
    st.stop()

focus_brand = table_brand

with st.expander("ℹ️ Details / 설명", expanded=False):
    if lang == "English":
        st.markdown(f"""
**What**  
At **{pd.to_datetime(sel_asof).date()}**, show each **publish day D ∈ [{(pd.to_datetime(sel_asof)-pd.Timedelta(days=WINDOW_DAYS-1)).date()} … {pd.to_datetime(sel_asof).date()}]** and its **total engagement at T** for **{focus_brand}**.

**How**  
Fix **T** and aggregate *(views + likes + comments)* for videos whose **publish_date = D**. Repeat for the seven days in the window.

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
**T**를 고정한 뒤, **게시일=D**인 영상들의 *(조회수+좋아요+댓글)*을 합산합니다. 이를 창의 7개 날짜에 대해 반복합니다.

**어떻게 해석하나요?**  
오늘의 7일 합이 **최근 업로드**의 영향인지, **며칠 전 영상의 계속 누적**인지 한눈에 구분할 수 있습니다.

**활용 팁**  
**신규 스파이크(최근 D)**와 **롱테일(이전 D)**을 구분하여 업로드 타이밍과 캠페인 전략을 조정하세요.
""")

table = table_publish_window(panel, focus_brand, sel_asof)
table = table.sort_values("publish_date", ascending=True)

# Ensure the displayed publish date has no time component
table_display = table.copy()
table_display["publish_date"] = table_display["publish_date"].dt.date
# Format engagement with commas
table_display["engagement"] = table_display["engagement"].apply(lambda x: "{:,}".format(x))
st.dataframe(
    table_display.rename(columns={"publish_date": "Publish date / 게시일", "engagement": "Engagement (as-of T) / 참여(기준일)"}),
    use_container_width=True,
)

# Footer
if lang == "English":
    st.caption("Data: yt_brand_daily_panel.csv. Engagement = views + likes + comments. Timestamps treated as UTC. Start date filtered at 2025-09-21.")
else:
    st.caption("데이터: yt_brand_daily_panel.csv. 참여 = 조회수+좋아요+댓글. 모든 시각은 UTC 기준입니다. 시작일은 2025-09-21로 필터링되었습니다.")