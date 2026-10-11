# 2-minute demo script

Setup before recording: `streamlit run dashboard/demo.py`, browser at full width, sample "real CIC-IDS2017 slice" selected, "Overlay logistic-regression baseline" on. About 280 spoken words (roughly 1:55 at a steady pace); speak at a steady pace. Times are targets.

| time | on screen | say |
|---|---|---|
| 0:00 | Title and the yellow banner | "This is our submission for PS 26153: a world model of network state that estimates infiltration risk and ATT&CK stage, with an explanation for every prediction, fully offline. I'll show it, and I'll also show where it falls short." |
| 0:15 | Sample selected; metrics row; timeline | "This is fifty-five minutes of real CIC-IDS2017 traffic, summarised into sixty-second windows. The blue line is the world model's infiltration probability; the red ticks are the true attack minutes. It catches twenty of twenty-one attack windows with three false alarms. This sample is in-sample for the demo model, and the banner says so. Note the green line: a plain logistic regression, which in our benchmark scores better overall." |
| 0:45 | Flagged windows table, then the window inspector at 16:00 | "Every flagged window comes with its drivers. For this DDoS minute, the share of flows between one pair of hosts, the share of large packets and the ports touched by one source push the score up. The bars to the right are SHAP values, checked against the shap library. The model rolls forward six minutes; these future bars are its expectation, not a proven forecast." |
| 1:10 | Click "Analyse hosts" | "Which traffic drives this? We remove each host from the six-minute lookback and re-score. Removing either end of the DDoS pair drops the probability from point nine eight to about point zero five; every other host changes nothing. Below are the flows themselves." |
| 1:30 | Switch to synthetic PCAP sample | "Now a synthetic PCAP, processed with Scapy. The pipeline works end to end, but the model does not flag the scan in minute eight. It has never seen traffic like this, and we show that rather than tune it away." |
| 1:50 | Banner / README results table | "Bottom line: the full system works and is reproducible, but on this data the temporal model does not beat logistic regression. Thank you." |

## Recording checklist

- Close other tabs; zoom the browser to 110% so tables are readable.
- Run the app once before recording so the model and threshold are cached (first load takes about 10 s).
- Do not edit the sidebar threshold on camera; the default is intended.
- Keep to 2:00 or less; if over, cut the PCAP scene to one sentence.
- The numbers spoken (20 of 21, 3 false alarms, 0.98 to about 0.05) come from the bundled sample at the default threshold; re-check them if the model or threshold changes.


