"""
Streamlit dashboard: early-detection risk timeline on real network traffic.

Reads the out-of-fold scores written by `python src/detect_eval.py`, so every score
shown was produced by a model that never trained on that part of the timeline.

Run:  streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OOF = ROOT / "models" / "detection_oof.parquet"
RES = ROOT / "models" / "detection_results.json"

st.set_page_config(page_title="Network Attack Early Detection", layout="wide")
st.title("Network Attack Early Detection")
st.caption(
    "One point = one minute of traffic. Score = how attack-like the last few minutes look. "
    "Scores are out-of-fold: the model never saw that part of the timeline in training."
)

if not OOF.exists():
    st.warning("No results yet. Run `python src/detect_eval.py` first.")
    st.stop()

df = pd.read_parquet(OOF).sort_values("window_idx").reset_index(drop=True)
results = json.loads(RES.read_text()) if RES.exists() else {}
models = [c.removeprefix("score_") for c in df.columns if c.startswith("score_")]
shap_cols = [c for c in df.columns if c.startswith("shap_")]

with st.sidebar:
    model = st.selectbox("Model", models, index=models.index("xgboost") if "xgboost" in models else 0)
    score = df[f"score_{model}"]
    default_thr = float(results.get(model, {}).get("threshold", score.quantile(0.98)))
    lo, hi = float(score.min()), float(score.max())
    thr = st.slider("Alert threshold", lo, hi, min(max(default_thr, lo), hi), (hi - lo) / 200 or 0.01)
    st.caption(
        "Default threshold is calibrated so about 2% of benign minutes raise a false alarm."
    )

t = pd.to_datetime(df["window_start"], unit="s")
alert = (score >= thr) & score.notna()
atk = df["is_attack"] == 1

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=score, mode="lines", name="Risk score", line=dict(width=1.2)))
fig.add_hline(y=thr, line_dash="dash", line_color="orange", annotation_text="alert threshold")
fig.add_trace(
    go.Scatter(
        x=t[atk], y=score[atk], mode="markers", name="Actual attack minute",
        marker=dict(color="red", size=6, symbol="x"),
        text=df.loc[atk, "attack_type"], hovertemplate="%{text}<br>score %{y:.2f}<extra></extra>",
    )
)
fig.update_layout(height=480, xaxis_rangeslider_visible=True, yaxis_title="Risk score", margin=dict(t=30))
st.plotly_chart(fig, use_container_width=True)

tp, fp = int((alert & atk).sum()), int((alert & ~atk).sum())
missed = int((~alert & atk & score.notna()).sum())
benign = int((~atk & score.notna()).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Attack minutes flagged", tp)
c2.metric("False alarms", fp, f"{fp / max(benign, 1):.1%} of benign minutes", delta_color="off")
c3.metric("Attack minutes missed", missed)
r = results.get(model)
if r:
    c4.metric("Episodes detected (at default threshold)", f"{r['detected']:.0%}", f"median delay {r['median_delay']:.1f} min", delta_color="off")

st.subheader("By attack type")
by = (
    df[atk].assign(flagged=alert[atk])
    .groupby("attack_type")["flagged"].agg(minutes="size", flagged="sum")
    .assign(share_flagged=lambda x: (x.flagged / x.minutes).round(2))
    .sort_values("share_flagged", ascending=False)
)
st.dataframe(by, use_container_width=True)
st.caption(
    "Attack types that blend into normal traffic at one-minute granularity (e.g. bots, "
    "web attacks) stay near zero. This is a limit of aggregate-flow features, not a bug."
)

st.subheader("Why did this alert fire?")
if model != "xgboost" or not shap_cols:
    st.info("Per-alert explanations are available for the XGBoost model (TreeSHAP).")
else:
    alerts = df[alert].copy()
    if alerts.empty:
        st.write("No alerts at this threshold.")
    else:
        alerts["when"] = t[alert].dt.strftime("%Y-%m-%d %H:%M")
        alerts["label"] = alerts["when"] + "  |  " + alerts["attack_type"].replace("", "benign (false alarm)")
        pick = st.selectbox("Alert", alerts["label"].tolist())
        row = alerts[alerts["label"] == pick].iloc[0]
        sv = row[shap_cols].astype(float)
        sv.index = [c.removeprefix("shap_") for c in sv.index]
        top = sv.reindex(sv.abs().sort_values(ascending=False).index)[:8][::-1]
        bar = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=["crimson" if v > 0 else "steelblue" for v in top.values]))
        bar.update_layout(height=340, xaxis_title="Contribution to risk (log-odds; red pushes toward 'attack')",
                          margin=dict(t=10))
        st.plotly_chart(bar, use_container_width=True)
        st.caption("`last:` = current minute, `mean:`/`max:` = over the lookback window, `slope:` = recent change.")
