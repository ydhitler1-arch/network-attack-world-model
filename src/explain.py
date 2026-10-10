"""
Explainability for every prediction (Phase 4).

World model (the required prediction engine): SHAP values via the *expected-gradients* estimator
(Erion et al.), the same estimator behind shap.GradientExplainer, implemented directly so it can
differentiate through the 6-step history AND the K-step rollout:

    phi_i(x) = E_{x' ~ background, a ~ U(0,1)} [ (x_i - x'_i) * d f / d x_i (x' + a (x - x')) ]

f is the infiltration logit read at rollout step k (k = 0 current window ... K), or the stage
logits at k = 0. Inputs are the L x D normalised history; attributions are reported per feature
(summed over the L lags) and per lag (summed over features). Completeness holds in expectation:
sum(phi) = f(x) - E[f(background)]; the residual is reported, not hidden.

Logistic regression baseline: exact linear SHAP, phi_i = w_i (x_i - E[x_i]).

    python src/explain.py            # background, explanations for all windows, validation report

Outputs: models/explain_background.npz, models/explanations_full.parquet,
         models/explanation_validation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from state import STATE_COLUMNS  # noqa: E402
from world_model import history_index, load_model  # noqa: E402


# ------------------------------------------------------------------ estimator

def expected_gradients(f, x: torch.Tensor, bg: torch.Tensor, n_samples: int = 64, seed: int = 0, chunk: int = 16):
    """x (B,L,D), bg (M,L,D), f: (N,L,D) -> (N,O). Returns phi (B,O,L,D)."""
    g = torch.Generator().manual_seed(seed)
    B, L, D = x.shape
    M = bg.shape[0]
    out_phi = []
    with torch.enable_grad():
        for a in range(0, B, chunk):
            xb = x[a:a + chunk]
            b = xb.shape[0]
            base = bg[torch.randint(M, (b, n_samples), generator=g)]            # b,n,L,D
            alpha = torch.rand(b, n_samples, 1, 1, generator=g)
            pts = (base + alpha * (xb[:, None] - base)).reshape(-1, L, D).requires_grad_(True)
            out = f(pts)
            O = out.shape[1]
            ph = []
            for o in range(O):
                (gr,) = torch.autograd.grad(out[:, o].sum(), pts, retain_graph=o < O - 1)
                ph.append(((xb[:, None] - base) * gr.view(b, n_samples, L, D)).mean(1))
            out_phi.append(torch.stack(ph, 1))
    return torch.cat(out_phi)


def make_background(model_ck: dict, state: pd.DataFrame, n: int = 200, seed: int = 0) -> np.ndarray:
    """Benign L-window histories (normalised) from `state`, used as the SHAP reference distribution."""
    Z, idx, valid = history_index(model_ck, state)
    benign = valid & (state["window_is_attack"].to_numpy() == 0)
    # also require the whole history benign, so the reference is "normal traffic"
    atk = state["window_is_attack"].to_numpy()
    hist_clean = atk[idx].sum(1) == 0
    cand = np.where(benign & hist_clean)[0]
    pick = np.random.default_rng(seed).choice(cand, min(n, len(cand)), replace=False)
    return Z[idx[pick]]


# ------------------------------------------------------------------ world model

def explain_world_model(model, ck, state: pd.DataFrame, bg: np.ndarray, K: int | None = None,
                        n_samples: int = 64, batch: int = 64, seed: int = 0, rows=None):
    """SHAP for the K+1 infiltration logits and the k=0 stage logits of every window (or only `rows`,
    in which case the outputs hold len(rows) entries in that order, and Z/idx stay full-length)."""
    cfg = ck["cfg"]
    L, K = cfg["lookback"], (cfg["horizon"] if K is None else K)
    Z, idx, valid = history_index(ck, state)
    if rows is not None:
        rows = np.asarray(rows)
        valid = valid[rows]
        X = torch.from_numpy(Z[idx[rows]])
    else:
        X = torch.from_numpy(Z[idx])
    BG = torch.from_numpy(bg.astype(np.float32))
    model.eval()

    def f_atk(x):
        return model(x, K)["atk"][:, L - 1:]

    def f_stage(x):
        return model(x, 0)["stage"][:, L - 1]

    with torch.no_grad():
        o = model(X, K)
        logit = o["atk"][:, L - 1:].numpy()
        stage_logit = o["stage"][:, L - 1].numpy()
        base_atk = f_atk(BG).mean(0).numpy()
        base_stage = f_stage(BG).mean(0).numpy()
    phi_a = expected_gradients(f_atk, X, BG, n_samples, seed, chunk=batch // 4)       # T,K+1,L,D
    phi_s = expected_gradients(f_stage, X, BG, n_samples, seed + 1, chunk=batch // 4)  # T,S,L,D
    return {
        "valid": valid, "logit": logit, "p": 1 / (1 + np.exp(-logit)), "base_logit": base_atk,
        "phi_feat": phi_a.sum(2).numpy(), "phi_lag": phi_a.sum(3).numpy(),
        "stage_logit": stage_logit, "stage_base": base_stage,
        "stage_pred": stage_logit.argmax(1), "phi_stage_feat": phi_s.sum(2).numpy(),
        "Z": Z, "idx": idx, "K": K, "L": L,
    }


def completeness_error(res):
    """Mean |sum(phi) - (f(x) - base)| and its size relative to |f(x) - base|, valid windows only."""
    v = res["valid"]
    tot = res["phi_feat"].sum(-1)                       # T,K+1
    target = res["logit"] - res["base_logit"][None]
    err = np.abs(tot - target)[v]
    return {"mean_abs_error": float(err.mean()), "mean_abs_target": float(np.abs(target[v]).mean()),
            "relative": float(err.mean() / np.abs(target[v]).mean())}


def deletion_test(model, ck, res, k=0, windows=None, ms=(3, 5, 10), seed=0):
    """Faithfulness: set the m features with the largest positive SHAP (for the infiltration logit at
    step k) to the benign reference (z = 0) across all lags, and compare the logit drop with deleting m
    random features and the m most-negative features. Faithful attributions: top >> random >> bottom."""
    L, K = res["L"], res["K"]
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(res["Z"][res["idx"]])
    D = X.shape[-1]
    phi = res["phi_feat"][:, k]
    out = {}
    with torch.no_grad():
        for m in ms:
            drops = {"top_shap": [], "random": [], "bottom_shap": []}
            for t in windows:
                x = X[t:t + 1]
                base = float(model(x, K)["atk"][0, L - 1 + k])
                order = np.argsort(-phi[t])
                for name, sel in (("top_shap", order[:m]), ("bottom_shap", order[-m:]),
                                  ("random", rng.choice(D, m, replace=False))):
                    xd = x.clone()
                    xd[:, :, sel] = 0.0
                    drops[name].append(base - float(model(xd, K)["atk"][0, L - 1 + k]))
            out[m] = {k_: float(np.mean(v)) for k_, v in drops.items()}
    return out


# ------------------------------------------------------------------ logistic regression baseline

def fit_logreg(Z_t: np.ndarray, y: np.ndarray):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Z_t, y)


def explain_logreg(clf, Z_t: np.ndarray, bg_mean: np.ndarray):
    """Exact linear SHAP for the decision function. Returns phi (T,D) and base value."""
    phi = clf.coef_[0][None, :] * (Z_t - bg_mean[None, :])
    base = float(clf.intercept_[0] + clf.coef_[0] @ bg_mean)
    return phi, base


# ------------------------------------------------------------------ per-window report

def explanation_table(res, state: pd.DataFrame, cols: list[str], stages: list[str], top: int = 5) -> pd.DataFrame:
    """Long table: for each valid window and horizon k, the top features by |SHAP| (infiltration), plus
    the top features for the predicted stage at k=0 (k = -1 rows)."""
    rows = []
    v = np.where(res["valid"])[0]
    raw = state[cols].to_numpy()
    for t in v:
        for k in range(res["K"] + 1):
            ph = res["phi_feat"][t, k]
            for r, j in enumerate(np.argsort(-np.abs(ph))[:top], 1):
                rows.append((t, k, float(res["p"][t, k]), None, r, cols[j], float(ph[j]),
                             float(raw[t, j]), float(res["Z"][t, j])))
        s = int(res["stage_pred"][t])
        ph = res["phi_stage_feat"][t, s]
        for r, j in enumerate(np.argsort(-np.abs(ph))[:top], 1):
            rows.append((t, -1, float(res["p"][t, 0]), stages[s], r, cols[j], float(ph[j]),
                         float(raw[t, j]), float(res["Z"][t, j])))
    df = pd.DataFrame(rows, columns=["window_idx", "k", "p_infil", "stage", "rank", "feature", "shap",
                                     "value_raw", "value_z"])
    df["window_start"] = state["window_start"].to_numpy()[df["window_idx"]]
    return df


def describe_window(table: pd.DataFrame, t: int, k: int = 0) -> str:
    """One-line human-readable explanation of a window's prediction."""
    g = table[(table.window_idx == t) & (table.k == k)].sort_values("rank")
    if g.empty:
        return "no explanation (window lacks a full history)"
    parts = [f"{r.feature} ({'+' if r.shap > 0 else '-'}{abs(r.shap):.2f}, z={r.value_z:+.1f})" for r in g.itertuples()]
    return f"P(infiltration at t+{k}) = {g.p_infil.iloc[0]:.2f}; drivers: " + ", ".join(parts)


# ------------------------------------------------------------------ main

N_SAMPLES = 256  # (background, alpha) draws per explanation; completeness error shrinks ~ 1/sqrt(N)


def main():
    cfg_state = "data/state_v1.parquet"
    state = pd.read_parquet(cfg_state).reset_index(drop=True)
    report = {}

    # ---- demo model (trained on all days) + background from benign CIC-IDS2017 windows
    model, ck = load_model("models/world_model_full.pt")
    bg = make_background(ck, state, n=200)
    np.savez("models/explain_background.npz", bg=bg, cols=np.array(ck["cols"]))
    print(f"background: {bg.shape}")

    res = explain_world_model(model, ck, state, bg, n_samples=N_SAMPLES)
    ce = completeness_error(res)
    report["completeness"] = ce
    print("completeness (sum phi vs f(x)-base):", ce)

    tab = explanation_table(res, state, ck["cols"], ck["cfg"]["stages"], top=5)
    tab.to_parquet("models/explanations_full.parquet")
    print(f"explanation table: {tab.shape}")

    # ---- faithfulness on the HOLDOUT model's test day (Fri, not seen in training)
    hm, hck = load_model("models/world_model.pt")
    hres = explain_world_model(hm, hck, state, bg, n_samples=N_SAMPLES)
    hce = completeness_error(hres)
    report["holdout_completeness"] = hce
    fri = np.where((state["segment"].to_numpy() == 4) & hres["valid"])[0]
    top_windows = fri[np.argsort(-hres["logit"][fri, 0])[:100]]
    report["deletion_test_holdout_fri_top100_k0"] = deletion_test(hm, hck, hres, 0, top_windows)
    allfri = fri
    report["deletion_test_holdout_fri_all_k0"] = deletion_test(hm, hck, hres, 0, allfri)
    print("deletion (logit drop), top-100 scored Fri windows:", report["deletion_test_holdout_fri_top100_k0"])

    # ---- cross-check against the shap library's GradientExplainer on a few windows
    try:
        import shap
        L, K = ck["cfg"]["lookback"], res["K"]

        class W(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.m = model

            def forward(s, x):
                return s.m(x, 0)["atk"][:, L - 1:L]

        sel = np.where(res["valid"])[0][::97][:20]
        X = torch.from_numpy(res["Z"][res["idx"][sel]])
        ge = shap.GradientExplainer(W(), torch.from_numpy(bg.astype(np.float32)))
        sv = np.asarray(ge.shap_values(X, nsamples=200)).reshape(len(sel), L, -1)
        mine = res["phi_feat"][sel, 0]
        theirs = sv.sum(1)
        corr = [float(np.corrcoef(mine[i], theirs[i])[0, 1]) for i in range(len(sel))]
        report["shap_library_crosscheck"] = {"windows": len(sel), "mean_pearson_feature_attribution": float(np.mean(corr)),
                                              "min_pearson": float(np.min(corr))}
        print("shap.GradientExplainer cross-check:", report["shap_library_crosscheck"])
    except Exception as e:  # noqa: BLE001
        report["shap_library_crosscheck"] = f"skipped: {type(e).__name__}: {e}"
        print(report["shap_library_crosscheck"])

    # ---- logistic regression: exactness check + agreement with the world model
    Zc = res["Z"]
    y = state["window_is_attack"].to_numpy()
    clf = fit_logreg(Zc, y)
    phi_lr, base_lr = explain_logreg(clf, Zc, bg[:, -1].mean(0))
    exact = np.abs(phi_lr.sum(1) + base_lr - clf.decision_function(Zc)).max()
    report["logreg_linear_shap_max_abs_error"] = float(exact)
    v = hres["valid"] & (state["segment"].to_numpy() == 4)
    from scipy.stats import spearmanr
    rho = [spearmanr(np.abs(phi_lr[t]), np.abs(hres["phi_feat"][t, 0])).statistic for t in np.where(v & (y == 1))[0]]
    report["logreg_vs_worldmodel_attack_windows_spearman_abs_shap"] = {"n": len(rho), "mean": float(np.nanmean(rho))}
    print("logreg exactness:", exact, "| agreement:", report["logreg_vs_worldmodel_attack_windows_spearman_abs_shap"])

    # ---- which features the world model leans on (global importance, mean |SHAP| k=0)
    imp = pd.Series(np.abs(res["phi_feat"][res["valid"], 0]).mean(0), index=ck["cols"]).sort_values(ascending=False)
    report["global_importance_k0_top10"] = imp.head(10).round(3).to_dict()
    lag = np.abs(res["phi_lag"][res["valid"], 0]).mean(0)
    report["lag_importance_k0_oldest_to_newest"] = (lag / lag.sum()).round(3).tolist()
    print("top features:", report["global_importance_k0_top10"])
    print("lag importance (oldest->newest):", report["lag_importance_k0_oldest_to_newest"])

    Path("models/explanation_validation.json").write_text(json.dumps(report, indent=1))
    print("saved models/explanation_validation.json")


if __name__ == "__main__":
    main()
