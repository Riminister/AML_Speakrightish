# Architecture

This project is a simple end-to-end audio classification pipeline:

1. Read `labels.csv` + audio files
2. Create a stratified train/val split
3. Extract features (e.g. log-mel spectrogram summaries)
4. Train a model:
   - **sklearn** baseline (fast, good for small data)
   - **tensorflow** neural net (Keras)
5. Evaluate and write outputs (metrics + plots)

## Keep it simple

Everything you’ll touch lives in just two places:

- `scripts/`
  - `train.py`: loads data → extracts features → trains (`--model sklearn|tensorflow`) → saves model + plots
  - `eval.py`: loads a saved model (`.joblib` or `.keras`) and prints metrics

- `src/aml_speakrightish/`
  - `data/`: reading `labels.csv` and train/test splitting
  - `features/`: audio feature extraction
  - `models/`: sklearn + tensorflow models and a small registry to pick one by name

