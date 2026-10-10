"""Phase 5 checks: the demo pipeline end to end on the bundled samples (no Streamlit needed).

    python -W ignore tests/test_phase5.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

import inference as inf  # noqa: E402
from state import STATE_COLUMNS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

# ---------------------------------------------------------------- CSV sample (real CIC-IDS2017 slice)
flows = inf.load_flows(os.path.join(ROOT, "data/sample/demo_flows.csv.gz"))
an = inf.analyze(flows)
n = len(an.state)
assert n >= 50, n
assert an.pred["p_infil"].shape == (n, an.K + 1) and np.isfinite(an.pred["p_infil"]).all()
assert np.allclose(an.pred["stage_p"].sum(-1), 1, atol=1e-5)
ground = an.state["window_is_attack"].to_numpy() == 1
valid = an.pred["valid"]
print(f"CSV: {len(flows):,} flows -> {n} windows ({valid.sum()} with full history), attack windows {ground.sum()}")
print(f"  mean P(now) attack windows {an.p_now[ground & valid].mean():.2f} vs benign {an.p_now[~ground & valid].mean():.2f}")

# explanation for the highest-risk window, completeness within tolerance
t = int(np.argmax(np.where(valid, an.p_now, -1)))
ex = inf.explain_windows(an, [t], n_samples=256)
tot = ex["phi_feat"][0, 0].sum()
tgt = ex["logit"][0, 0] - ex["base_logit"][0]
print(f"  window {t} ({an.t.iloc[t]}): P={an.p_now[t]:.2f} stage={an.stages[int(ex['stage_pred'][0])]} "
      f"sum(SHAP)={tot:.2f} vs logit-base={tgt:.2f}")
assert abs(tot - tgt) < 0.35 * max(abs(tgt), 1), (tot, tgt)
drv = inf.top_drivers(ex["phi_feat"][0, 0], an.pred["Z"][t], an.state[STATE_COLUMNS].to_numpy()[t])
print(drv.to_string(index=False))

# counterfactual: removing the attacker's flows must lower P and the attacker should rank first
src = inf.suspect_hosts(an, t)
print(src.round(3).to_string(index=False))
assert not src.empty and src.iloc[0]["drop"] > 0
ft = inf.flow_table(an, t, src.iloc[0]["host"])
assert len(ft) > 0
print(f"  flagged-flow table for {src.iloc[0]['host']}: {len(ft)} flows, labels {ft['label'].value_counts().to_dict()}")

# ---------------------------------------------------------------- PCAP sample (synthetic traffic)
pf = inf.load_flows(os.path.join(ROOT, "data/sample/demo_traffic.pcap"))
pa = inf.analyze(pf)
print(f"\nPCAP: {pf.attrs.get('packets', '?'):,} packets -> {len(pf):,} flows -> {len(pa.state)} windows")
for i in range(len(pa.state)):
    print(f"  min {i:2d}: P(now)={pa.p_now[i]:.2f} stage={pa.stages[int(pa.pred['stage_p'][i, 0].argmax())]:<18s} "
          f"ports/src={pa.state.max_ports_per_src.iloc[i]:.0f} flows={pa.state.flow_count.iloc[i]:.0f} "
          f"ttl_var={pa.state.ttl_var_mean.iloc[i]:.1f}")
assert "ttl_var_mean" in pa.state.columns

# unlabelled CSV (no Label column) must not be treated as an attack
import tempfile  # noqa: E402

tmp = os.path.join(tempfile.mkdtemp(), "nolabel.csv")
flows.drop(columns=["label", "is_attack"]).head(5000).to_csv(tmp, index=False)
nl = inf.load_flows(tmp)
assert (nl["is_attack"] == 0).all() and (nl["label"] == "UNLABELLED").all()
print("\nOK")

