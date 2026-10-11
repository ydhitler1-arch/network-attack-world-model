"""Checks for the finite-sample conservative threshold (src/threshold_drift.py): it must do what it says.

    python -W ignore tests/test_threshold_methods.py        (or: pytest tests/test_threshold_methods.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

import threshold_drift as TD  # noqa: E402


def threshold_of(delta, sv):
    """Recover the threshold the method uses: the smallest score that is NOT alarmed on."""
    yv = np.zeros(len(sv), int)
    alarms = TD.finite_sample_alarms(sv=sv, yv=yv, st=sv, delta=delta)
    return sv[~alarms].max() if (~alarms).any() else -np.inf


def test_threshold_rises_with_confidence():
    rng = np.random.default_rng(0)
    for _ in range(20):
        sv = rng.standard_normal(300)
        t50, t80, t90, t99 = (threshold_of(d, sv) for d in (0.5, 0.8, 0.9, 0.99))
        assert t50 <= t80 <= t90 <= t99, (t50, t80, t90, t99)


def test_coverage_matches_the_stated_confidence():
    """With n i.i.d. benign calibration scores, P(true false-alarm rate <= FAR) should be about delta."""
    rng = np.random.default_rng(1)
    n, reps = 288, 3000          # about the size of one validation fold
    for delta in (0.5, 0.8, 0.9):
        ok = 0
        for _ in range(reps):
            th = threshold_of(delta, rng.standard_normal(n))
            ok += norm.sf(th) <= TD.FAR     # exact exceedance probability of this threshold under N(0, 1)
        cover = ok / reps
        # the method is conservative by construction (discrete ranks), so allow it to exceed delta but not fall short
        assert cover >= delta - 0.04, f"coverage {cover:.3f} below the stated confidence {delta}"
        assert cover <= min(delta + 0.15, 1.0), f"coverage {cover:.3f} is needlessly conservative for delta {delta}"


def test_plain_quantile_overshoots_far_more_often_than_it_should():
    """The benchmark's plain quantile exceeds the 2% target about half of the time; the 90% rule almost never does."""
    rng = np.random.default_rng(2)
    plain = conservative = 0
    reps = 2000
    for _ in range(reps):
        sv = rng.standard_normal(288)
        plain += norm.sf(np.quantile(sv, 1 - TD.FAR)) > TD.FAR
        conservative += norm.sf(threshold_of(0.9, sv)) > TD.FAR
    assert plain / reps > 0.35, plain / reps
    assert conservative / reps < 0.15, conservative / reps


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok   {name}")
    print(f"OK - {len(tests)} threshold-method checks passed")
