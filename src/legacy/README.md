# Legacy scripts

The earlier early-detection study (forecasting at chance, reframed as early detection; see
[../../docs/EARLY_DETECTION_STUDY.md](../../docs/EARLY_DETECTION_STUDY.md)). They still run from the repository root, for example:

```bash
python src/features.py data/raw 60           # core module: 60 s windows with the 13 original features
python src/legacy/evaluate.py                # forecasting experiment
python src/legacy/detect_eval.py --lstm      # early-detection evaluation (writes models/detection_oof.parquet)
```

`src/features.py` stays in `src/` because the current pipeline (`src/state.py`) builds on it. The synthetic
data generator here (`make_synthetic_data.py`) plants a reconnaissance ramp and is a methodology demo only.
