"""
Train and sanity-evaluate the world model (Phase 2).

    python src/wm_train.py                       # train on train segments, early-stop on val, report test
    python src/wm_train.py --mode full           # retrain on ALL segments for the epoch count found above
    python src/wm_train.py --config configs/world_model.json --exclude-type DDoS   # leave an attack type out

The holdout split in the config (Mon-Wed train, Thu val, Fri test) is a SANITY CHECK: it is a single
chronological split whose test day contains only attack types never seen in training. Phase 3 does
the proper blocked cross-validation and leave-one-attack-type-out comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from state import STATE_COLUMNS, StateScaler  # noqa: E402
from world_model import WorldModel, _norm_label, save_model, stage_ids  # noqa: E402


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_starts(seg: np.ndarray, span: int, allowed_segments, banned: np.ndarray | None = None):
    """Start indices i such that windows i..i+span-1 are in one allowed segment (and none banned)."""
    n = len(seg)
    ok = np.zeros(n - span + 1, dtype=bool)
    for s in allowed_segments:
        ok |= (seg[: n - span + 1] == s) & (seg[span - 1:] == s)
    starts = np.where(ok)[0]
    if banned is not None and banned.any():
        cs = np.concatenate([[0], np.cumsum(banned)])
        starts = starts[(cs[starts + span] - cs[starts]) == 0]
    return starts


def gather(arr: np.ndarray, starts: np.ndarray, span: int):
    return arr[starts[:, None] + np.arange(span)[None, :]]


def losses(out, x, y, st, cfg, pos_weight, stage_w):
    """x (B,T,D), y (B,T), st (B,T). Returns dict of scalar losses."""
    lw = cfg["loss_weights"]
    mu, lv = out["mu"][:, :-1], out["logvar"][:, :-1]
    nll = 0.5 * (lv + (x[:, 1:] - mu) ** 2 * torch.exp(-lv)).mean()
    atk, stg = out["atk"], out["stage"]
    if cfg.get("use_dynamics", True) and not cfg.get("rollout_task_loss", True):
        # infiltration/stage losses on the real history steps only (not on imagined rollout steps)
        L = cfg["lookback"]
        atk, stg, y, st = atk[:, :L], stg[:, :L], y[:, :L], st[:, :L]
    bce = F.binary_cross_entropy_with_logits(atk, y, pos_weight=pos_weight)
    ce = F.cross_entropy(stg.reshape(-1, stg.shape[-1]), st.reshape(-1), weight=stage_w)
    total = lw["infiltration"] * bce + lw["stage"] * ce
    if cfg.get("use_dynamics", True):
        total = total + lw["dynamics"] * nll
    return {"total": total, "nll": nll, "bce": bce, "ce": ce}


def run_epoch(model, X, Y, S, starts, cfg, K, pos_weight, stage_w, opt=None, gen=None):
    L, B = cfg["lookback"], cfg["batch_size"]
    shift = cfg.get("clf_shift", 0)  # classifier only: step j predicts the label of window j+shift
    T = L + K if cfg.get("use_dynamics", True) else L + shift
    train = opt is not None
    model.train(train)
    order = starts[torch.randperm(len(starts), generator=gen).numpy()] if train else starts
    agg, n = {"total": 0.0, "nll": 0.0, "bce": 0.0, "ce": 0.0}, 0
    with torch.set_grad_enabled(train):
        for a in range(0, len(order), B):
            idx = order[a:a + B]
            x = torch.from_numpy(gather(X, idx, T))
            y = torch.from_numpy(gather(Y, idx, T))
            st = torch.from_numpy(gather(S, idx, T))
            out = model(x[:, :L], K if cfg.get("use_dynamics", True) else 0)
            if not cfg.get("use_dynamics", True):
                x, y, st = x[:, :L], y[:, shift:], st[:, shift:]
            ls = losses(out, x, y, st, cfg, pos_weight, stage_w)
            if train:
                opt.zero_grad()
                ls["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            for k in agg:
                agg[k] += float(ls[k]) * len(idx)
            n += len(idx)
    return {k: v / max(n, 1) for k, v in agg.items()}


def fit_ar1(X, seg, train_segs, K):
    """Direct per-feature linear predictor x_{t+k} ~ a_k * x_t + b_k, fitted on training segments."""
    coefs = {}
    for k in range(1, K + 1):
        idx = make_starts(seg, k + 1, train_segs)
        x0, xk = X[idx], X[idx + k]
        xm, ym = x0.mean(0), xk.mean(0)
        a = ((x0 - xm) * (xk - ym)).sum(0) / np.maximum(((x0 - xm) ** 2).sum(0), 1e-8)
        coefs[k] = (a, ym - a * xm)
    return coefs


@torch.no_grad()
def evaluate(model, X, Y, S, starts, cfg, K, ar1=None):
    """Rollout metrics on `starts`: dynamics error vs baselines, infiltration AUC per horizon."""
    L = cfg["lookback"]
    T = L + K
    model.eval()
    x = torch.from_numpy(gather(X, starts, T))
    y = gather(Y, starts, T)
    st = gather(S, starts, T)
    out = model(x[:, :L], K)
    res = {"n_samples": int(len(starts)), "horizon": {}}
    p = torch.sigmoid(out["atk"]).numpy()
    sp = out["stage"].argmax(-1).numpy()
    for k in range(0, K + 1):
        j = L - 1 + k
        h = {"attack_rate": float(y[:, j].mean())}
        if 0 < y[:, j].mean() < 1:
            h["roc_auc"] = float(roc_auc_score(y[:, j], p[:, j]))
            h["pr_auc"] = float(average_precision_score(y[:, j], p[:, j]))
        if k >= 1:
            tgt = x[:, j]
            pred = out["mu"][:, j - 1]
            h["mse_model"] = float(((pred - tgt) ** 2).mean())
            h["mse_persistence"] = float(((x[:, L - 1] - tgt) ** 2).mean())
            h["mse_train_mean"] = float((tgt ** 2).mean())  # predictor "benign training mean" (z = 0)
            if ar1 is not None:
                a, b = ar1[k]
                h["mse_ar1"] = float(((x[:, L - 1].numpy() * a + b - tgt.numpy()) ** 2).mean())
        res["horizon"][k] = h
    a0 = y[:, L - 1] == 1
    res["stage_at_k0_attack_windows"] = {
        "true": pd.Series(st[:, L - 1][a0]).value_counts().sort_index().to_dict(),
        "pred": pd.Series(sp[:, L - 1][a0]).value_counts().sort_index().to_dict(),
        "accuracy": float((sp[:, L - 1][a0] == st[:, L - 1][a0]).mean()) if a0.any() else None,
    }
    return res


def prepare(cfg, exclude_types=()):
    state = pd.read_parquet(cfg["state_path"]).reset_index(drop=True)
    seg = state["segment"].to_numpy()
    Y = state["window_is_attack"].to_numpy(np.float32)
    S = stage_ids(state["attack_type"], cfg)
    banned = None
    if exclude_types:
        ex = {_norm_label(e) for e in exclude_types}
        banned = state["attack_type"].map(_norm_label).isin(ex).to_numpy()
    return state, seg, Y, S, banned


def fit(cfg, state, seg, Y, S, banned, train_segs, val_segs, epochs=None, log=None, verbose=True):
    L, K = cfg["lookback"], cfg["horizon"]
    span = L + K if cfg.get("use_dynamics", True) else L
    Kuse = K if cfg.get("use_dynamics", True) else 0
    cols = STATE_COLUMNS
    tr_starts = make_starts(seg, L + Kuse, train_segs, banned)
    va_starts = make_starts(seg, L + Kuse, val_segs) if val_segs else np.array([], dtype=int)

    # scaler: benign, non-banned windows of the training segments only
    tr_win = np.isin(seg, train_segs)
    scaler = StateScaler(cols).fit(state, tr_win & (Y == 0) & (~banned if banned is not None else True))
    X = np.clip(scaler.transform(state), -cfg["clip_z"], cfg["clip_z"]).astype(np.float32)

    seed_everything(cfg["seed"])
    model = WorldModel(len(cols), len(cfg["stages"]), cfg["hidden"], cfg["dropout"], cfg.get("use_dynamics", True))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tw = np.unique(np.concatenate([tr_starts[:, None] + np.arange(L + Kuse)[None, :]]).ravel())
    tw = tw[tw < len(Y)]
    pos = Y[tw].mean()
    pos_weight = torch.tensor(min(cfg["pos_weight_cap"], (1 - pos) / max(pos, 1e-6)), dtype=torch.float32)
    cnt = np.bincount(S[tw], minlength=len(cfg["stages"])).astype(np.float64)
    sw = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0.0)
    stage_w = torch.tensor(sw / sw[cnt > 0].mean(), dtype=torch.float32)  # absent classes get weight 0

    gen = torch.Generator().manual_seed(cfg["seed"])
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    n_ep = epochs or cfg["epochs"]
    for ep in range(1, n_ep + 1):
        tr = run_epoch(model, X, Y, S, tr_starts, cfg, Kuse, pos_weight, stage_w, opt, gen)
        va = run_epoch(model, X, Y, S, va_starts, cfg, Kuse, pos_weight, stage_w) if len(va_starts) else tr
        if log is not None:
            log.append({"epoch": ep, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"  ep {ep:3d}  train nll {tr['nll']:+.3f} bce {tr['bce']:.3f} ce {tr['ce']:.3f} | "
                  f"val nll {va['nll']:+.3f} bce {va['bce']:.3f} ce {va['ce']:.3f}")
        if len(va_starts):
            if va["total"] < best - 1e-4:
                best, best_state, best_ep, bad = va["total"], copy.deepcopy(model.state_dict()), ep, 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_ep = n_ep
    model.eval()
    return model, scaler, X, best_ep, {"train_samples": int(len(tr_starts)), "val_samples": int(len(va_starts)),
                                       "pos_weight": float(pos_weight), "stage_weights": stage_w.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world_model.json")
    ap.add_argument("--mode", choices=["holdout", "full"], default="holdout")
    ap.add_argument("--exclude-type", action="append", default=[], help="attack_type left out of training")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override a top-level config value")
    ap.add_argument("--tag", default="", help="suffix for output files (to keep tuning runs apart)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    if args.tag:
        cfg["outputs"] = {k: v.replace(".", f"_{args.tag}.", 1) if k != "model_full" else v
                          for k, v in cfg["outputs"].items()}
    sp = cfg["split"]
    state, seg, Y, S, banned = prepare(cfg, args.exclude_type)
    K = cfg["horizon"]
    out_cfg = cfg["outputs"]
    metrics_path = Path(out_cfg["metrics"])
    log: list = []

    if args.mode == "holdout":
        model, scaler, X, best_ep, info = fit(cfg, state, seg, Y, S, banned, sp["train_segments"], sp["val_segments"], log=log)
        print(f"best epoch {best_ep} ({info['train_samples']} train / {info['val_samples']} val samples)")
        res = {"config": args.config, "excluded_types": args.exclude_type, "best_epoch": best_ep, **info}
        ar1 = fit_ar1(X, seg, sp["train_segments"], K)
        res["overrides"] = args.set
        for name, segs in (("val", sp["val_segments"]), ("test", sp["test_segments"])):
            starts = make_starts(seg, cfg["lookback"] + K, segs)
            res[name] = evaluate(model, X, Y, S, starts, cfg, K, ar1)
        save_model(out_cfg["model"], model, cfg, STATE_COLUMNS, {
            "cols": scaler.cols, "mean": scaler.mean_.tolist(), "std": scaler.std_.tolist()},
            {"mode": "holdout", "best_epoch": best_ep, "train_segments": sp["train_segments"]})
        pd.DataFrame(log).to_csv(out_cfg["train_log"], index=False)
        metrics_path.write_text(json.dumps(res, indent=1))
        print_report(res, K)
    else:
        prev = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        n_ep = int(prev.get("best_epoch", 30))
        all_segs = sorted(set(sp["train_segments"] + sp["val_segments"] + sp["test_segments"]))
        model, scaler, X, _, info = fit(cfg, state, seg, Y, S, banned, all_segs, [], epochs=n_ep, log=log)
        save_model(out_cfg["model_full"], model, cfg, STATE_COLUMNS, {
            "cols": scaler.cols, "mean": scaler.mean_.tolist(), "std": scaler.std_.tolist()},
            {"mode": "full", "epochs": n_ep, "segments": all_segs})
        print(f"saved {out_cfg['model_full']} (trained on all segments for {n_ep} epochs; "
              f"do not use it to evaluate on CIC-IDS2017 itself)")


def print_report(res, K):
    for name in ("val", "test"):
        r = res[name]
        print(f"\n[{name}] {r['n_samples']} samples")
        print("  k  attack_rate  ROC-AUC  PR-AUC | MSE model  persistence  train-mean  AR(1)")
        for k, h in r["horizon"].items():
            print(f"  {k}  {h['attack_rate']:.3f}        "
                  f"{h.get('roc_auc', float('nan')):.3f}    {h.get('pr_auc', float('nan')):.3f}  | "
                  + (f"{h['mse_model']:.3f}      {h['mse_persistence']:.3f}        {h['mse_train_mean']:.3f}"
                     f"       {h.get('mse_ar1', float('nan')):.3f}" if k else "-"))
        s = r["stage_at_k0_attack_windows"]
        print(f"  stage at k=0 on attack windows: true {s['true']} pred {s['pred']} acc {s['accuracy']}")


if __name__ == "__main__":
    main()
