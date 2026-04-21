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
  6. Push every report to [skore hub](https://hub.probabl.ai/) via
     `skore-hub-project` for browsing in the hub UI
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

## Pushing to skore hub

The last section of the notebook persists every sweep report to
[skore hub](https://hub.probabl.ai/). It runs `skore.login()` (device-code
browser flow), then opens a hub-backed project and uploads each report:

```python
skore.login()
project = skore.Project(name="skorch-airpassengers-demo", mode="hub", workspace=WORKSPACE)
for name, report in reports.items():
    project.put(name, report)
```

Set `SKORE_WORKSPACE` in your environment (or edit the cell) to a workspace
you own. These cells are not auto-executed because they need interactive auth.
