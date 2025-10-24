# postprocess_lang_en_only.py
# Filters gathered YouTube data to English-only videos, then rebuilds panel & roll7.
# Uses fastText LID (with optional CLD3 fallback); requires lid.176.bin. No argparse; configure paths & knobs below.

import os
import time
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Tuple

# Language ID: fastText (primary) with optional CLD3 fallback
# pip install fasttext pycld3
import pandas as pd
try:
    import fasttext  # type: ignore
except Exception:
    fasttext = None  # will error at runtime if model not available
try:
    import cld3  # type: ignore
except Exception:
    cld3 = None

# ---------- Config ----------
INPUT_DIR = Path("../data")
OUTPUT_DIR = Path("../filtered_data")

REGISTRY_CSV = INPUT_DIR / "yt_video_registry.csv"
STATS_CSV    = INPUT_DIR / "yt_video_stats_daily.csv"
PANEL_CSV    = INPUT_DIR / "yt_brand_daily_panel.csv"       # not strictly required to read
ROLL7_CSV    = INPUT_DIR / "yt_brand_roll7_daily.csv"       # not strictly required to read

OUT_REGISTRY = OUTPUT_DIR / "yt_video_registry.filtered.csv"
OUT_STATS    = OUTPUT_DIR / "yt_video_stats_daily.filtered.csv"
OUT_PANEL    = OUTPUT_DIR / "yt_brand_daily_panel.filtered.csv"
OUT_ROLL7    = OUTPUT_DIR / "yt_brand_roll7_daily.filtered.csv"
OUT_AUDIT    = OUTPUT_DIR / "yt_video_language_audit.csv"

LANG_KEEP = {"en"}          # keep only these languages (add "ko" if desired)
CONF_MIN = 0.60              # min confidence for trusted detection (fastText prob)
DROP_UNKNOWN = False         # unknown/low-conf will pass unless confidently non-EN

# fastText model path (download once: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_LID_PATH", str(Path(__file__).with_name("lid.176.bin")))

# English preference heuristics for mixed/codeswitched text
EN_MARGIN = 0.08             # prefer EN if within 8% of the top score
EN_FLOOR = 0.35              # minimum EN probability to consider margin preference

# Column hints (script tolerates missing ones)
TITLE_COL = "title"
DESC_COL = "description"
TAGS_COL = "tags"
CHANNEL_TITLE_COL = "channel_title"



# ---------- IO helpers ----------
def _read_csv(path: Path, cols: List[str] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols or [])
    df = pd.read_csv(path, encoding="utf-8")
    if cols:
        for c in cols:
            if c not in df.columns:
                df[c] = pd.Series(dtype="object")
    return df

def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


import re

# ---------- Language detection ----------
def _ensure_list_like(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        # attempt to parse a bracketed list like "['a','b']"
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]
            parts = [p.strip(" '\"") for p in inner.split(",")]
            return [p for p in parts if p]
        # treat pipe-separated as list
        if "|" in s:
            return [p.strip() for p in s.split("|") if p.strip()]
        return [s] if s else []
    return []

_URL_RE = re.compile(r"http\S+|www\.\S+", re.I)
_HASH_AT_RE = re.compile(r"[@#][\w_]+", re.U)
_NONWORD_RE = re.compile(r"[^\w\s]", re.U)

EN_STOPS = {"the","and","is","for","with","this","that","from","you","your","we","our","to","in","on","of","it","are","was","be","as","at","by","an","or","if","but"}

def clean_text(s: str) -> str:
    s = (s or "")
    s = _URL_RE.sub(" ", s)
    s = _HASH_AT_RE.sub(" ", s)
    s = _NONWORD_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# Load fastText model lazily
_FT_MODEL = None
def _load_ft():
    global _FT_MODEL
    if _FT_MODEL is None:
        if fasttext is None:
            raise RuntimeError("fasttext is not installed. `pip install fasttext` and place lid.176.bin.")
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            raise RuntimeError(f"fastText LID model not found at '{FASTTEXT_MODEL_PATH}'. Set FASTTEXT_LID_PATH env or place lid.176.bin.")
        _FT_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
    return _FT_MODEL

def ft_detect(text: str) -> tuple[str, float]:
    text = (text or "").replace("\n", " ")
    if not text.strip():
        return "", 0.0
    model = _load_ft()
    labels, probs = model.predict(text)
    lang = labels[0].replace("__label__", "")
    conf = float(probs[0])
    return lang, conf

def cld3_detect(text: str) -> tuple[str, float]:
    if cld3 is None:
        return "", 0.0
    r = cld3.get_language(text or "")
    if not r:
        return "", 0.0
    return r.language or "", float(r.probability or 0.0)

def english_hint(text: str, k: int = 3) -> bool:
    toks = (text or "").split()
    return sum(1 for t in toks if t in EN_STOPS) >= k

def detect_lang_ensemble(text: str) -> tuple[str, float, dict]:
    """Detect language using fastText primary with optional CLD3; apply EN margin and hints.
    Returns (lang, conf, debug) where debug has component scores.
    """
    txt = clean_text(text)
    if not txt:
        return "", 0.0, {"ft": ("", 0.0), "cld3": ("", 0.0), "hint_en": False}
    ft_lang, ft_p = ft_detect(txt)
    cld_lang, cld_p = cld3_detect(txt)

    # Prefer EN if either detector is confident
    if ft_lang == "en" and ft_p >= 0.70:
        lang, conf = "en", ft_p
    elif cld_lang == "en" and cld_p >= 0.70:
        lang, conf = "en", cld_p
    else:
        # Margin rule: if EN is close to top
        en_score = 0.0
        if ft_lang == "en":
            en_score = max(en_score, ft_p)
        if cld_lang == "en":
            en_score = max(en_score, cld_p)
        top_lang, top_p = (ft_lang, ft_p) if ft_p >= cld_p else (cld_lang, cld_p)
        if en_score >= EN_FLOOR and (top_lang != "en") and (top_p - en_score) <= EN_MARGIN:
            lang, conf = "en", en_score
        else:
            lang, conf = top_lang, top_p

    # Stopword hint: if still not EN but looks English, upgrade cautiously
    if lang != "en" and english_hint(txt):
        lang, conf = "en", max(conf, 0.60)

    return lang, conf, {"ft": (ft_lang, ft_p), "cld3": (cld_lang, cld_p), "hint_en": english_hint(txt)}


def build_video_language_map(registry: pd.DataFrame) -> pd.DataFrame:
    reg = registry.copy()
    for col in [TITLE_COL, DESC_COL, TAGS_COL, CHANNEL_TITLE_COL]:
        if col not in reg.columns:
            reg[col] = ""

    # Build detection text from title + description ONLY (cleaned)
    title_series = reg[TITLE_COL].astype(str)
    desc_series  = reg[DESC_COL].astype(str)
    text_all = (title_series + " " + desc_series).apply(clean_text)

    # Fallback corpus from tags + channel title (cleaned)
    tags_series_raw = reg[TAGS_COL].apply(_ensure_list_like).apply(lambda lst: " ".join(lst)).astype(str)
    chan_series_raw = reg[CHANNEL_TITLE_COL].astype(str)
    fallback_all = (tags_series_raw + " " + chan_series_raw).apply(clean_text)

    # We'll consider primary text "too short" if < 20 characters
    TOO_SHORT = 20

    # Prepare tags/title overrides for borderline cases
    tags_series = tags_series_raw.astype(str).str.lower()
    brand_series = reg.get("brand", pd.Series([""]*len(reg))).astype(str).str.lower()
    title_lc = title_series.str.lower()

    def brand_token_present(i: int) -> bool:
      b = brand_series.iloc[i].strip()
      if not b:
          return False
      # boundary-ish check by padding with spaces
      t = f" {title_lc.iloc[i]} "
      tg = f" {tags_series.iloc[i]} "
      bb = f" {b} "
      return (bb in t) or (bb in tg)

    langs, confs, reasons = [], [], []
    for i in range(len(reg)):
      txt = text_all.iloc[i]
      # Fallback if title+desc too short
      if len(txt) < TOO_SHORT:
          txt = fallback_all.iloc[i]

      lang, conf, dbg = detect_lang_ensemble(txt)

      # Override: if brand token present in title/tags and language unknown/low-conf, prefer keep as EN
      if (not lang or conf < CONF_MIN) and brand_token_present(i):
          lang, conf = "en", max(conf, 0.60)
          reasons.append(f"keep-override: brand_token, dbg={dbg}")
      else:
          reasons.append(f"auto: {lang}({conf:.2f}), dbg={dbg}")

      langs.append(lang)
      confs.append(conf)

    decisions = []
    final_reasons = []
    for lang, conf, why in zip(langs, confs, reasons):
      if lang in LANG_KEEP and conf >= CONF_MIN:
          decisions.append("keep")
          final_reasons.append(f"keep: {why}")
      elif (not lang) or conf < CONF_MIN:
          if DROP_UNKNOWN:
              decisions.append("drop")
              final_reasons.append(f"drop: unknown/lowconf — {why}")
          else:
              decisions.append("keep")
              final_reasons.append(f"keep: unknown allowed — {why}")
      else:
          decisions.append("drop")
          final_reasons.append(f"drop: non-keep {lang}({conf:.2f}) — {why}")

    df_lang = reg[["video_id", "brand"]].copy()
    df_lang["lang"] = langs
    df_lang["conf"] = confs
    df_lang["decision"] = decisions
    df_lang["reason"] = final_reasons
    return df_lang


# ---------- Rebuild panel & roll7 from filtered data ----------
def rebuild_panel_and_roll7(filtered_registry: pd.DataFrame, stats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recompute daily panel and roll7 *from filtered data only*.
    We aggregate using available stats for each as_of_date_utc and the registry's published_at_utc.
    """
    if filtered_registry.empty or stats.empty:
        panel_cols = ["date","brand","as_of_date_utc","video_mentions","views","likes","comments","top_channels"]
        roll7_cols = ["report_date_utc","brand","roll7_video_mentions","roll7_views","roll7_likes","roll7_comments","roll7_top_channels"]
        return pd.DataFrame(columns=panel_cols), pd.DataFrame(columns=roll7_cols)

    reg = filtered_registry.copy()
    st  = stats.copy()

    reg["published_at_utc"] = pd.to_datetime(reg["published_at_utc"]).dt.date
    st["as_of_date_utc"]    = pd.to_datetime(st["as_of_date_utc"]).dt.date

    panels = []
    roll7s = []
    for T in sorted(st["as_of_date_utc"].unique()):
        win_days = [T - timedelta(days=i) for i in range(6, -1, -1)]

        st_T = st[st["as_of_date_utc"] == T][["video_id","viewCount","likeCount","commentCount"]].copy()
        joined = reg.merge(st_T, on="video_id", how="left").fillna(
            {"viewCount":0, "likeCount":0, "commentCount":0}
        )

        rows = []
        for d in win_days:
            day_rows = joined[joined["published_at_utc"] == d]
            if day_rows.empty:
                continue
            grp = (day_rows.groupby("brand", as_index=False)
                          .agg(video_mentions=("video_id","nunique"),
                               views=("viewCount","sum"),
                               likes=("likeCount","sum"),
                               comments=("commentCount","sum")))
            grp["date"] = d
            grp["as_of_date_utc"] = T
            grp["top_channels"] = ""  # optional, needs channel stats
            rows.append(grp)

        panel_T = (pd.concat(rows, ignore_index=True)
                   if rows else
                   pd.DataFrame(columns=["brand","video_mentions","views","likes","comments","date","as_of_date_utc","top_channels"]))

        # Fill grid with zeros for missing (date, brand)
        all_days = [d for d in win_days]
        all_brands = sorted(reg["brand"].unique().tolist())
        if all_brands:
            idx = pd.MultiIndex.from_product([all_days, all_brands], names=["date","brand"])
            panel_T = (panel_T.set_index(["date","brand"])
                                .reindex(idx)
                                .reset_index())
            panel_T["as_of_date_utc"] = panel_T["as_of_date_utc"].fillna(pd.Timestamp(T))
            for c in ["video_mentions","views","likes","comments"]:
                panel_T[c] = panel_T[c].fillna(0).astype(int)
            panel_T["top_channels"] = panel_T["top_channels"].fillna("")

        panels.append(panel_T)

        # roll-7 from this panel slice
        if not panel_T.empty:
            agg = (panel_T.groupby("brand", as_index=False)
                           .agg(roll7_video_mentions=("video_mentions","sum"),
                                roll7_views=("views","sum"),
                                roll7_likes=("likes","sum"),
                                roll7_comments=("comments","sum")))
            agg["report_date_utc"] = pd.Timestamp(T)
            agg["roll7_top_channels"] = ""
            roll7s.append(agg)

    panel_all = (pd.concat(panels, ignore_index=True)
                 if panels else
                 pd.DataFrame(columns=["date","brand","as_of_date_utc","video_mentions","views","likes","comments","top_channels"]))
    roll7_all = (pd.concat(roll7s, ignore_index=True)
                 if roll7s else
                 pd.DataFrame(columns=["report_date_utc","brand","roll7_video_mentions","roll7_views","roll7_likes","roll7_comments","roll7_top_channels"]))

    # Ensure ordering
    panel_all["date"] = pd.to_datetime(panel_all["date"]).dt.date
    panel_all["as_of_date_utc"] = pd.to_datetime(panel_all["as_of_date_utc"]).dt.date
    panel_all = panel_all[["date","brand","as_of_date_utc","video_mentions","views","likes","comments","top_channels"]]

    roll7_all["report_date_utc"] = pd.to_datetime(roll7_all["report_date_utc"]).dt.date
    roll7_all = roll7_all[["report_date_utc","brand","roll7_video_mentions","roll7_views","roll7_likes","roll7_comments","roll7_top_channels"]]

    return panel_all, roll7_all


def main():
    # 1) Load inputs
    reg_cols = ["video_id","brand","published_at_utc","channel_id","channel_title","first_seen_utc",
                "matched_fields","cohort_date_utc","selected_at_ts_utc","rank_on_selection","selection_order",
                # optional text fields if present:
                "title","description","tags"]
    registry = _read_csv(REGISTRY_CSV, reg_cols)

    stats_cols = ["video_id","as_of_date_utc","as_of_ts_utc","viewCount","likeCount","commentCount"]
    stats = _read_csv(STATS_CSV, stats_cols)

    if registry.empty or stats.empty:
        print("Registry or stats are empty — nothing to filter.")
        return

    # 2) Build language decisions
    df_lang = build_video_language_map(registry)
    keep_ids = set(df_lang.loc[df_lang["decision"] == "keep", "video_id"].astype(str))

    print(f"[lang] keep={len(keep_ids)} / total={len(df_lang)} "
          f"(dropped={len(df_lang) - len(keep_ids)})")

    # 3) Filter registry/stats
    reg_f = registry[registry["video_id"].astype(str).isin(keep_ids)].copy()
    stats_f = stats[stats["video_id"].astype(str).isin(keep_ids)].copy()

    # 4) Rebuild panel & roll7 from filtered data
    panel_f, roll7_f = rebuild_panel_and_roll7(reg_f, stats_f)

    # 5) Write outputs
    _write_csv(reg_f, OUT_REGISTRY)
    _write_csv(stats_f, OUT_STATS)
    _write_csv(panel_f, OUT_PANEL)
    _write_csv(roll7_f, OUT_ROLL7)

    audit = df_lang.copy()
    audit["keep"] = audit["decision"].eq("keep")
    _write_csv(audit, OUT_AUDIT)

    print(f"✓ Wrote: {OUT_REGISTRY} ({len(reg_f)} rows)")
    print(f"✓ Wrote: {OUT_STATS} ({len(stats_f)} rows)")
    print(f"✓ Wrote: {OUT_PANEL} ({len(panel_f)} rows)")
    print(f"✓ Wrote: {OUT_ROLL7} ({len(roll7_f)} rows)")
    print(f"✓ Wrote: {OUT_AUDIT} ({len(audit)} rows)")

if __name__ == "__main__":
    main()