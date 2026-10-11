"""Smoke test for the Streamlit demo (headless): both samples load, the host analysis runs, no exceptions.

    python -W ignore tests/test_demo_app.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard", "demo.py"))

at = AppTest.from_file(APP, default_timeout=600).run()
assert not at.exception, [e.value for e in at.exception]
metrics = {m.label: m.value for m in at.metric}
assert metrics["Flows"] == "201,690" and metrics["60 s windows"] == "56", metrics
caught, total = (int(x) for x in metrics["Attack windows caught"].split("/"))
assert total == 21 and caught >= 18, metrics  # in-sample sample: the demo model should catch nearly all

[b for b in at.button if "Analyse hosts" in b.label][0].click().run()
assert not at.exception, [e.value for e in at.exception]

at.radio[0].set_value("Sample: synthetic traffic (PCAP)").run()
assert not at.exception, [e.value for e in at.exception]
assert {m.label: m.value for m in at.metric}["Flows"] == "6,025"
print(f"OK - demo runs on both samples ({metrics['Attack windows caught']} attack windows caught on the CIC sample)")
