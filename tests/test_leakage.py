"""Leakage checks for the evaluation splits (src/benchmark.py): the README's claims about purged, blocked
cross-validation and leave-one-attack-type-out rest on these.

    python -W ignore tests/test_leakage.py        (or: pytest tests/test_leakage.py)
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import benchmark as B  # noqa: E402
import threshold_drift as TD  # noqa: E402
from state import STATE_COLUMNS, StateScaler  # noqa: E402
from world_model import _norm_label  # noqa: E402

STATE = pd.read_parquet(os.path.join(ROOT, "data", "state_v1.parquet")).reset_index(drop=True)
SEG = STATE["segment"].to_numpy()
N = len(STATE)
L, K = 6, 6
SPANS = (L, L + 3, L + K)  # classifier / evaluation / world-model training spans


def min_gap(a: np.ndarray, b: np.ndarray) -> int:
    """Smallest |i - j| between a window index in mask a and one in mask b."""
    ia, ib = np.where(a)[0], np.where(b)[0]
    if len(ia) == 0 or len(ib) == 0:
        return 10**9
    pos = np.searchsorted(ib, ia)
    left = np.abs(ia - ib[np.clip(pos - 1, 0, len(ib) - 1)])
    right = np.abs(ia - ib[np.clip(pos, 0, len(ib) - 1)])
    return int(np.minimum(left, right).min())


def covered(starts: np.ndarray, span: int) -> np.ndarray:
    """Boolean mask of the windows touched by samples that start at `starts`."""
    m = np.zeros(N, bool)
    for off in range(span):
        m[starts + off] = True
    return m


def test_regions_are_disjoint_and_purged():
    for k in range(B.NFOLD):
        tr, va, te = B.regions(N, k, None)
        assert not (tr & va).any() and not (tr & te).any() and not (va & te).any(), f"fold {k}: regions overlap"
        assert min_gap(tr, te) > B.PURGE, f"fold {k}: train within purge distance of test"
        assert min_gap(tr, va) > B.PURGE, f"fold {k}: train within purge distance of validation"
        assert min_gap(va, te) > B.PURGE, f"fold {k}: validation within purge distance of test"


def test_samples_stay_inside_their_region_and_day():
    for k in range(B.NFOLD):
        regions = dict(zip(("train", "val", "test"), B.regions(N, k, None)))
        for name, region in regions.items():
            for span in SPANS:
                starts = B.region_starts(SEG, span, region)
                assert len(starts) > 0, (k, name, span)
                idx = starts[:, None] + np.arange(span)[None, :]
                assert region[idx].all(), f"fold {k} {name} span {span}: sample reaches outside its region"
                assert (SEG[idx] == SEG[idx[:, :1]]).all(), f"fold {k} {name} span {span}: sample spans two capture days"


def test_train_and_test_samples_share_no_window():
    for k in range(B.NFOLD):
        tr, va, te = B.regions(N, k, None)
        train_w = covered(B.region_starts(SEG, L + K, tr), L + K)
        val_w = covered(B.region_starts(SEG, L + 3, va), L + 3)
        test_w = covered(B.region_starts(SEG, L + 3, te), L + 3)
        assert not (train_w & test_w).any() and not (train_w & val_w).any() and not (val_w & test_w).any()
        assert min_gap(train_w, test_w) > B.PURGE - (L + K), f"fold {k}: sample histories overlap"


def test_each_test_sample_belongs_to_exactly_one_fold():
    seen = np.zeros(N, int)
    for k in range(B.NFOLD):
        _, _, te = B.regions(N, k, None)
        starts = B.region_starts(SEG, L + 3, te)
        np.add.at(seen, starts, 1)
    assert seen.max() == 1, "a sample is in more than one test fold"


def test_leave_one_attack_type_out_keeps_the_type_out_of_training():
    types = sorted({_norm_label(a) for a in STATE["attack_type"].unique() if a})
    assert len(types) >= 10, types
    norm = STATE["attack_type"].map(_norm_label).to_numpy()
    for typ in types:
        banned = norm == typ
        assert banned.sum() > 0
        in_test = np.zeros(N, bool)
        for k in range(B.NFOLD):
            tr, va, te = B.regions(N, k, banned)
            assert not (tr & banned).any(), f"{typ}: held-out windows in training"
            assert not (va & banned).any(), f"{typ}: held-out windows in validation"
            assert min_gap(tr, banned) > B.PURGE, f"{typ}: training too close to held-out windows"
            assert min_gap(va, banned) > B.PURGE, f"{typ}: validation too close to held-out windows"
            in_test |= te & banned
        assert (in_test == banned).all(), f"{typ}: some held-out windows never appear in a test fold"


def test_scaler_ignores_rows_outside_its_mask():
    rng = np.random.default_rng(0)
    tr, _, te = B.regions(N, 0, None)
    fit_mask = tr & (STATE["window_is_attack"].to_numpy() == 0)
    a = StateScaler(STATE_COLUMNS).fit(STATE, fit_mask)
    poisoned = STATE.copy()
    # Cast to float first: pandas >= 3.0 no longer silently upcasts an
    # int64 column when a float value is assigned into it via .loc, and
    # raises instead. The poisoning itself only cares about the values.
    poisoned[STATE_COLUMNS] = poisoned[STATE_COLUMNS].astype(float)
    poisoned.loc[~fit_mask, STATE_COLUMNS] = poisoned.loc[~fit_mask, STATE_COLUMNS].to_numpy() + rng.normal(1e6, 1, (int((~fit_mask).sum()), len(STATE_COLUMNS)))
    b = StateScaler(STATE_COLUMNS).fit(poisoned, fit_mask)
    assert np.allclose(a.mean_, b.mean_) and np.allclose(a.std_, b.std_), "scaler statistics depend on unmasked rows"
    assert not (fit_mask & te).any(), "scaler fit mask includes test windows"


def test_threshold_calibration_uses_only_benign_validation_scores():
    rng = np.random.default_rng(1)
    s = rng.normal(size=500)
    y = (rng.random(500) < 0.2).astype(int)
    t1 = B.calibrate(s, y)
    s2 = s.copy()
    s2[y == 1] += 100.0  # attack scores must not move the threshold
    assert t1 == B.calibrate(s2, y)
    assert abs(t1 - np.quantile(s[y == 0], 1 - B.FAR_TARGET)) < 1e-12


def test_adaptive_thresholds_never_look_ahead():
    """Re-centring may use only past scores; feedback may use a label only `delay` windows after the window."""
    rng = np.random.default_rng(2)
    n = 300
    s = rng.normal(size=n)
    y = (rng.random(n) < 0.15).astype(int)
    sv, yv = rng.normal(size=200), (rng.random(200) < 0.15).astype(int)

    # re-centring: changing score i must not change the median used at any t <= i
    m0 = TD.trailing_median(s, 0.0)
    for i in (25, 100, 250):
        s2 = s.copy()
        s2[i] += 50.0
        assert np.array_equal(m0[: i + 1], TD.trailing_median(s2, 0.0)[: i + 1]), "median used a current or future score"

    # feedback: flipping the label of window j can only change alarms at i >= j + delay
    base = TD.feedback_alarms(sv=sv, yv=yv, st=s, y_test=y, eta=0.1, delay=TD.D)
    for j in (10, 120, 200):
        y2 = y.copy()
        y2[j] = 1 - y2[j]
        alt = TD.feedback_alarms(sv=sv, yv=yv, st=s, y_test=y2, eta=0.1, delay=TD.D)
        assert np.array_equal(base[: j + TD.D], alt[: j + TD.D]), f"alarms before window {j + TD.D} used the label of window {j}"
    # labels of the final `delay` windows can never be used at all
    y3 = y.copy()
    y3[-TD.D:] = 1 - y3[-TD.D:]
    assert np.array_equal(base, TD.feedback_alarms(sv=sv, yv=yv, st=s, y_test=y3, eta=0.1, delay=TD.D))


def test_the_check_would_catch_missing_purge():
    """Mutation check: with the purge disabled the same assertion must fail, proving the test is sensitive."""
    old = B.PURGE
    try:
        B.PURGE = 0
        tr, _, te = B.regions(N, 0, None)
        assert min_gap(tr, te) <= old, "test would not detect leakage without a purge"
    finally:
        B.PURGE = old


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok   {name}")
    print(f"OK - {len(tests)} leakage checks passed")
