"""
Offline demo for SIH PS 26153: upload a flow CSV or a PCAP, get the infiltration-probability timeline,
ATT&CK stage annotations, flagged flows and per-window SHAP explanations. No network access needed.

Run:  streamlit run dashboard/demo.py
(the older replay dashboard, dashboard/app.py, is unchanged)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import inference as inf  # noqa: E402
from state import STATE_COLUMNS  # noqa: E402

st.set_page_config(page_title="Network infiltration early-warning demo", layout="wide")

STAGE_COLORS = {
    "Reconnaissance": "#e69f00", "Initial Access": "#d55e00", "Lateral Movement": "#cc79a7",
    "Command & Control": "#0072b2", "Exfiltration": "#009e73",
    "Impact (outside the 5 stages)": "#7b2d8e", "Benign": "#9a9a9a",
}


# ------------------------------------------------------------------ cached heavy steps

@st.cache_resource(show_spinner="Loading model and calibrating thresholds...")
def thresholds():
    return inf.default_thresholds()


@st.cache_resource(show_spinner="Parsing input, building 60 s state windows, running the world model...")
def run_analysis(source: str, name: str, data: bytes | None, ws: float, max_packets: int):
    if source == "csv_sample":
        flows = inf.load_flows(ROOT / "data/sample/demo_flows.csv.gz")
    elif source == "pcap_sample":
        flows = inf.load_flows(ROOT / "data/sample/demo_traffic.pcap", max_packets=max_packets)
    else:
        suffix = "".join(Path(name).suffixes) or ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
        flows = inf.load_flows(f.name, "pcap" if suffix.lower().endswith((".pcap", ".pcapng", ".cap")) else "csv",
                               max_packets=max_packets)
    return inf.analyze(flows, ws)


@st.cache_data(show_spinner="Explaining flagged windows (SHAP)...")
def flagged_table(_an, key: str, rows: tuple):
    rows = np.array(rows)
    ex = inf.explain_windows(_an, rows, n_samples=64)
    out = []
    for i, t in enumerate(rows):
        top = np.argsort(-np.abs(ex["phi_feat"][i, 0]))[:3]
        out.append({
            "window": int(t), "time": _an.t.iloc[t].strftime("%Y-%m-%d %H:%M"),
            "P(now)": float(_an.pred["p_infil"][t, 0]),
            "max P next 6 min": float(_an.pred["p_infil"][t, 1:].max()),
            "predicted stage": _an.stages[int(ex["stage_pred"][i])],
            "top drivers": ", ".join(f"{STATE_COLUMNS[j]} ({'+' if ex['phi_feat'][i, 0, j] > 0 else '-'}"
                                      f"{abs(ex['phi_feat'][i, 0, j]):.2f})" for j in top),
            "truth": (_an.state["attack_type"].iloc[t] or "benign") if has_labels(_an) else "",
        })
    return pd.DataFrame(out)


def has_labels(an) -> bool:
    return bool((an.flows["label"] != "UNLABELLED").any())


# ------------------------------------------------------------------ sidebar

st.title("Network infiltration early-warning: offline demo")
with st.sidebar:
    st.header("Input")
    src_choice = st.radio("Source", [
        "Sample: real CIC-IDS2017 slice (CSV)", "Sample: synthetic traffic (PCAP)", "Upload CSV", "Upload PCAP"])
    up = None
    if src_choice.startswith("Upload"):
        up = st.file_uploader("File", type=["csv", "gz", "pcap", "pcapng"] if "PCAP" in src_choice else ["csv", "gz"])
    ws = st.number_input("Window length (s)", 10, 600, 60, 10,
                         help="The model was trained on 60 s windows. Other values change the feature scale.")
    max_packets = st.number_input("PCAP packet cap", 10_000, 20_000_000, 3_000_000, 100_000,
                                  help="Scapy parses roughly 10-30k packets/s.")
    thr_info = thresholds()
    st.header("Display")
    thr = st.slider("Alert threshold on P(now)", 0.0, 1.0, float(round(thr_info["world_model"], 2)), 0.01,
                    help="Default: 98th percentile of benign training windows (in-sample).")
    show_lr = st.checkbox("Overlay logistic-regression baseline", True,
                          help="The benchmark's best detector. Shown as sigmoid(score).")
    show_k3 = st.checkbox("Show world-model P(+3 min)", False)
    st.caption("Everything runs locally; no data leaves this machine.")

if src_choice.startswith("Sample: real"):
    source, name, data = "csv_sample", "demo_flows.csv.gz", None
elif src_choice.startswith("Sample: synth"):
    source, name, data = "pcap_sample", "demo_traffic.pcap", None
elif up is None:
    st.info("Upload a flow CSV (CICFlowMeter layout, as in CIC-IDS2017) or a PCAP to start, or pick a sample in the sidebar.")
    st.stop()
else:
    source, name, data = "upload", up.name, up.getvalue()

try:
    an = run_analysis(source, name, data, float(ws), int(max_packets))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not analyse this input: {type(e).__name__}: {e}")
    st.stop()

L = an.ck["cfg"]["lookback"]
valid = an.pred["valid"]
p0 = an.p_now
labels = has_labels(an)
flagged = np.where((p0 >= thr) & valid)[0]

# ------------------------------------------------------------------ honesty banner + summary

st.warning(
    "Research prototype. On CIC-IDS2017 the benchmark found that this world model does **not** beat a plain "
    "logistic-regression detector (PR-AUC 0.30 vs 0.45, not a significant difference), forecasting minutes ahead showed no reliable skill, and "
    "unseen attack types are mostly missed. "
    + ("The CIC-IDS2017 sample is **in-sample** for the demo model, so its scores look better than they would on new traffic. "
       if source == "csv_sample" else "")
    + ("This PCAP is synthetic and differs from the training data; scores on it say little about real performance. "
       if source == "pcap_sample" else "")
    + "See docs/BENCHMARK.md."
)

c = st.columns(5)
c[0].metric("Flows", f"{len(an.flows):,}")
c[1].metric("60 s windows", f"{len(an.state)}", f"{int((~valid).sum())} warm-up (padded history)", delta_color="off")
c[2].metric("Windows flagged", len(flagged))
if labels:
    gt = (an.state["window_is_attack"].to_numpy() == 1) & valid
    c[3].metric("Attack windows caught", f"{int(((p0 >= thr) & gt).sum())} / {int(gt.sum())}")
    c[4].metric("False alarms", f"{int(((p0 >= thr) & ~gt & valid).sum())} / {int((~gt & valid).sum())} benign")
else:
    c[3].metric("Ground truth", "none (unlabelled input)")

# ------------------------------------------------------------------ timeline

st.subheader("Infiltration-probability timeline")
t = an.t
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=p0, mode="lines", name="World model P(now)", line=dict(width=2, color="#1f77b4")))
if show_k3:
    fig.add_trace(go.Scatter(x=t, y=an.pred["p_infil"][:, 3], mode="lines", name="World model P(+3 min), rollout",
                             line=dict(width=1, dash="dot", color="#7f7f7f")))
if show_lr:
    lr_p = 1 / (1 + np.exp(-inf.logreg_scores(an, thr_info["clf"])))
    fig.add_trace(go.Scatter(x=t, y=lr_p, mode="lines", name="Logistic-regression baseline",
                             line=dict(width=1.3, color="#2ca02c")))
fig.add_hline(y=thr, line_dash="dash", line_color="orange", annotation_text="alert threshold")
warm = np.where(~valid)[0]
if len(warm):
    fig.add_vrect(x0=t.iloc[warm[0]], x1=t.iloc[warm[-1]] + pd.Timedelta(seconds=an.ws), fillcolor="grey", opacity=0.12,
                  line_width=0, annotation_text="warm-up", annotation_position="top left")
stage_idx = an.pred["stage_p"][:, 0].argmax(1)
for stg in sorted({an.stages[i] for i in stage_idx[flagged]} if len(flagged) else []):
    sel = flagged[[an.stages[stage_idx[i]] == stg for i in flagged]]
    label = "stage: " + (stg if stg != "Benign" else "unclassified (stage head says Benign)")
    fig.add_trace(go.Scatter(x=t.iloc[sel], y=p0[sel], mode="markers", name=label,
                             marker=dict(size=9, color=STAGE_COLORS.get(stg, "#444"), line=dict(width=1, color="white"))))
if labels:
    atk = np.where(an.state["window_is_attack"].to_numpy() == 1)[0]
    fig.add_trace(go.Scatter(x=t.iloc[atk], y=np.full(len(atk), 1.04), mode="markers", name="Ground truth: attack window",
                             marker=dict(symbol="line-ns-open", size=10, color="red"),
                             text=an.state["attack_type"].iloc[atk], hovertemplate="%{text}<extra>truth</extra>"))
fig.update_layout(height=430, yaxis=dict(range=[-0.02, 1.08], title="P(infiltration)"), xaxis_rangeslider_visible=True,
                  margin=dict(t=20), legend=dict(orientation="h", y=-0.35))
st.plotly_chart(fig, width="stretch")
st.caption("Marker colour = ATT&CK stage predicted for flagged windows. The stage head is weak (it has never seen "
           "Exfiltration, and C2/Recon only from little data): 'unclassified' means it predicted Benign.")

# ------------------------------------------------------------------ flagged windows

st.subheader("Flagged windows")
if len(flagged) == 0:
    st.write("No window reaches the alert threshold. Lower it in the sidebar to inspect borderline windows.")
else:
    order = flagged[np.argsort(-p0[flagged])][:200]
    key = f"{source}|{name}|{ws}|{len(an.flows)}"
    ft = flagged_table(an, key, tuple(int(i) for i in order))
    st.dataframe(ft.drop(columns=["window"]).style.format({"P(now)": "{:.2f}", "max P next 6 min": "{:.2f}"}),
                 width="stretch", hide_index=True, height=min(35 * len(ft) + 38, 320))
    if len(flagged) > len(order):
        st.caption(f"Showing the {len(order)} highest-risk of {len(flagged)} flagged windows.")

# ------------------------------------------------------------------ window inspector

st.subheader("Why this window? (per-window explanation)")
cand = np.where(valid)[0]
default = int(flagged[np.argmax(p0[flagged])]) if len(flagged) else (int(cand[np.argmax(p0[cand])]) if len(cand) else 0)
labels_sel = {int(i): f"{t.iloc[i].strftime('%H:%M')}  P={p0[i]:.2f}" + ("  FLAGGED" if i in set(flagged) else "") for i in cand}
if not labels_sel:
    st.info("Input is shorter than the 6-minute lookback; no window has a full history.")
    st.stop()
sel_t = st.selectbox("Window", list(labels_sel), index=list(labels_sel).index(default), format_func=labels_sel.get)
ex = inf.explain_windows(an, [sel_t], n_samples=256)
raw_row = an.state[STATE_COLUMNS].to_numpy()[sel_t]
z_row = an.pred["Z"][sel_t]
stg = an.stages[int(ex["stage_pred"][0])]

a, b = st.columns([1, 1])
with a:
    st.markdown(f"**{t.iloc[sel_t]}** &nbsp; P(infiltration now) = **{p0[sel_t]:.2f}** &nbsp; predicted stage: **{stg}**")
    if labels:
        st.markdown(f"Ground truth: **{an.state['attack_type'].iloc[sel_t] or 'benign'}**")
    kk = list(range(an.K + 1))
    pf = an.pred["p_infil"][sel_t]
    bar = go.Figure(go.Bar(x=[f"+{k} min" if k else "now" for k in kk], y=pf, marker_color=["#1f77b4"] + ["#aec7e8"] * an.K))
    bar.update_layout(height=250, yaxis=dict(range=[0, 1], title="P(infiltration)"), margin=dict(t=25, b=10),
                      title=dict(text="World-model rollout (imagined future windows)", font=dict(size=13)))
    st.plotly_chart(bar, width="stretch")
    st.caption("+1..+6 min are read from the model's own predicted future states. The benchmark found no reliable skill "
               "at predicting attacks 3 minutes ahead; treat these as the model's expectation, not a forecast.")
with b:
    k_sel = st.select_slider("Explain horizon", options=kk, value=0, format_func=lambda k: "now" if k == 0 else f"+{k} min")
    drv = inf.top_drivers(ex["phi_feat"][0, k_sel], z_row, raw_row, top=10)[::-1]
    sh = go.Figure(go.Bar(x=drv["shap"], y=drv["feature"], orientation="h",
                          marker_color=["#c0392b" if v > 0 else "#2874a6" for v in drv["shap"]],
                          customdata=np.stack([drv["value"], drv["z"]], 1),
                          hovertemplate="%{y}<br>SHAP %{x:+.2f}<br>value %{customdata[0]:.3g} (z=%{customdata[1]:+.1f})<extra></extra>"))
    sh.update_layout(height=340, margin=dict(t=25, b=10), xaxis_title="SHAP contribution to infiltration log-odds (red = towards attack)",
                     title=dict(text="Top drivers", font=dict(size=13)))
    st.plotly_chart(sh, width="stretch")

lag = ex["phi_lag"][0, k_sel]
lg = go.Figure(go.Bar(x=[f"-{L - 1 - i} min" if i < L - 1 else "current" for i in range(L)], y=lag,
                      marker_color=["#c0392b" if v > 0 else "#2874a6" for v in lag]))
lg.update_layout(height=200, margin=dict(t=25, b=10), yaxis_title="SHAP", title=dict(text="Which minute of the lookback mattered", font=dict(size=13)))
cl, cr = st.columns(2)
cl.plotly_chart(lg, width="stretch")
with cr:
    st.markdown("**Stage probabilities (now)**")
    ss = inf.stage_summary(an, sel_t)
    st.dataframe(ss.style.format({"probability": "{:.2f}"}), hide_index=True, width="stretch", height=215)
    sdrv = inf.top_drivers(ex["phi_stage_feat"][0, int(ex["stage_pred"][0])], z_row, raw_row, top=3)
    st.caption(f"Drivers of '{stg}': " + ", ".join(f"{r.feature} ({r.shap:+.2f})" for r in sdrv.itertuples()))

# ------------------------------------------------------------------ flagged flows

st.subheader("Which hosts and flows drive this window?")
st.caption("Counterfactual: all flows involving a host (as source or destination) are removed from the whole 6-minute "
           "lookback, the states are recomputed, and the model re-scores. A large drop = that host's traffic drives the score.")
if st.button("Analyse hosts for this window"):
    with st.spinner("Re-scoring with each candidate host removed..."):
        hosts = inf.suspect_hosts(an, sel_t)
    st.session_state["hosts"] = (sel_t, hosts)
if st.session_state.get("hosts", (None,))[0] == sel_t:
    hosts = st.session_state["hosts"][1]
    if hosts.empty:
        st.write("No flows in this window.")
    else:
        st.dataframe(hosts.style.format({"share_of_window": "{:.0%}", "P_without_host": "{:.2f}", "drop": "{:+.2f}"}),
                     hide_index=True, width="stretch")
        pick = st.selectbox("Show flows of host", hosts["host"].tolist())
        st.dataframe(inf.flow_table(an, sel_t, pick), hide_index=True, width="stretch", height=300)
        st.caption("First 500 flows of the selected host in this window.")

with st.expander("How this works and what it cannot do"):
    st.markdown(
        """
- **Pipeline:** flows are grouped into 60 s windows; 43 features per window (flow statistics, scan signatures, TCP-window and
  packet-size proxies); an LSTM world model predicts the next window's state distribution, is rolled out 6 steps, and reads
  an infiltration probability and an ATT&CK stage off every real or imagined state.
- **Explanations:** SHAP (expected-gradients estimator, validated against the `shap` library, r=0.99) on the infiltration
  and stage outputs; about 9% completeness error at 256 draws.
- **PCAP input:** converted to flows with Scapy, which also yields TTL, fragmentation, retransmission features; those are
  shown in the data but not used by the model (CIC-IDS2017 CSVs cannot supply them).
- **Limits:** trained on one week of lab traffic (41 attack episodes); the demo model was trained on all days and for only 3
  epochs; stage prediction cannot output Exfiltration and rarely identifies Recon/C2; thresholds drift across days.
        """)

