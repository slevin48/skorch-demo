# skorch + skore time series demo

LSTM and MLP forecasters for the AirPassengers dataset, built with
[skorch](https://skorch.readthedocs.io/) (sklearn-compatible PyTorch wrapper) and
compared with [skore](https://skore.probabl.ai/) (evaluation & reporting).

## What's in here

- `timeseries_demo.ipynb` — end-to-end notebook:
  1. Load AirPassengers, scale, build 12-month sliding windows
  2. Define an LSTM `nn.Module` and train it via `skorch.NeuralNetRegressor`
  3. Evaluate (RMSE/MAPE) and plot train/test predictions + training history
  4. Run a 24-month rolling multi-step forecast
  5. Sweep architectures (LSTM vs MLP), optimizers (Adam vs SGD), hidden sizes
     and learning rates, then compare them with `skore.ComparisonReport`
- `pyproject.toml` / `uv.lock` — dependencies managed by [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Run the notebook

Open it in your editor of choice (VS Code, JupyterLab, etc.), or execute headlessly:

```bash
uv run jupyter nbconvert --to notebook --execute timeseries_demo.ipynb --output timeseries_demo.ipynb
```

Or launch Jupyter:

```bash
uv run jupyter lab
```

## Notes on the skore integration

skorch's `NeuralNet.predict` internally delegates to `predict_proba`, which makes
skore's response-method probe mis-detect the estimator as a classifier. The
notebook wraps each fitted skorch net in a small `BaseEstimator + RegressorMixin`
(`SkorchRegressor`) that only exposes `fit`/`predict`, so skore treats it cleanly
as a regressor.

Key skore API used:

```python
comparison = ComparisonReport(reports)
summary = comparison.metrics.summarize().frame()       # metrics table
comparison.metrics.prediction_error().plot()           # predicted-vs-actual
```
