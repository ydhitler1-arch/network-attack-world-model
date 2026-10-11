# 2-minute demo script

*Updated for the retrained model (v1.0.0). Every number below was read from the running demo; re-check them if the model, the samples or the default threshold change (see the checklist).*

Setup before recording: `streamlit run dashboard/demo.py`, browser at full width, sample "real CIC-IDS2017 slice" selected, "Overlay logistic-regression baseline" on, threshold left at its default (0.69). 280 spoken words: about 1:52 at 150 words per minute, 2:00 at 140, so speak briskly or trim the PCAP scene. Times are targets.

| time | on screen | say |
|---|---|---|
| 0:00 | Title and the yellow banner | "This is a network intrusion early-warning demo. A model learns how traffic evolves, estimates infiltration risk and an ATT&CK stage, and explains every prediction. It runs fully offline. I'll show it working, and I'll show where it falls short." |
| 0:15 | Metrics row, then the timeline | "Fifty-five minutes of real CIC-IDS2017 traffic, in sixty-second windows. The blue line is the infiltration probability; the red ticks are the true attack minutes. At the default threshold it catches all twenty-one attack minutes, with one false alarm, the minute right after the attack ended. This sample is in-sample for the demo model, as the banner says. The green line is a plain logistic regression: noisier here, but better overall in our benchmark." |
| 0:45 | Flagged windows table, then the window inspector (16:15) | "Every flagged minute comes with reasons. For this DDoS minute, the share of flows between one pair of hosts, large packets, and many ports touched by one source push the score up. These are SHAP values, checked against the shap library. The bars on the left are the model's own six-minute rollout: its expectation, not a proven forecast." |
| 1:10 | Click "Analyse hosts" | "Which traffic drives it? We remove each host from the six-minute lookback and re-score. Remove either end of the DDoS pair and the probability falls from one to about point zero seven or point one one. Every other host changes nothing." |
| 1:30 | Switch to the synthetic PCAP sample | "Now a synthetic PCAP, parsed with Scapy. The pipeline works end to end, but the model misses the port scan in minute eight and flags the connection burst in minute ten. It has never seen traffic like this, and we show that rather than tune it away." |
| 1:50 | Banner / README results table | "Bottom line: the system works and is reproducible, but on this data the temporal model does not beat logistic regression. Thank you." |

## Recording checklist

- Close other tabs; zoom the browser to 110% so tables are readable.
- Run the app once before recording so the model and threshold are cached (first load takes about 10 s).
- Do not edit the sidebar threshold on camera; the default (0.69) is what the numbers below assume.
- Keep to 2:00 or less; if over, cut the PCAP scene to one sentence.
- Numbers spoken, and where they come from (default threshold, bundled samples):
  - sample CSV: 56 windows, 22 flagged, 21 of 21 attack windows caught, 1 false alarm (16:17, the minute after the attack ended; the attack spans 15:56 to 16:16);
  - default inspected window 16:15, P = 1.00, stage "Impact (outside the 5 stages)", top drivers `top_pair_share`, `large_pkt_ratio`, `max_ports_per_src`;
  - host analysis: either DDoS host removed takes P from 1.00 to 0.07 or 0.11, other hosts 0.997 unchanged;
  - synthetic PCAP: 12 windows; the scan minute (minute 8) scores 0.15 and is not flagged; the connection burst (minute 10, 0.89) and the minute after it (0.77) are flagged; ordinary synthetic minutes sit at 0.50 to 0.64, under the threshold.
- The banner quotes the benchmark: world model PR-AUC 0.30 against 0.45 for logistic regression, not a significant difference.
