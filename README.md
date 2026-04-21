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

### Gotchas for PyTorch models on skore hub (as of `skore-hub-project` 0.0.22)

Two constraints bite PyTorch/skorch users that don't apply to the local backend:

1. **X must be ≤ 2D.** Local `EstimatorReport` only calls `estimator.predict(X)` and
   never validates shape. The hub push path runs sklearn's `check_array` while
   building the payload, which rejects any `X` with `ndim > 2` — so a sequence
   model taking `(n_samples, lookback, channels)` will fail with
   `ValueError: Found array with dim 3, while dim <= 2 is required.`

   Workaround: present a 2D view to skore and reshape to 3D inside the wrapper.
   In this repo `SkorchRegressor._to_3d` re-adds the channel axis before calling
   `self.net.predict(...)`, and `EstimatorReport` is given `X.squeeze(-1)`.

2. **`predict_proba` must not be exposed on a regressor.** skorch's
   `NeuralNet.predict` internally calls `self.predict_proba(X)`, which makes
   skore's response-method probe (used when caching predictions) mis-detect the
   estimator as a classifier and try to read `classes_`.

   Workaround: wrap the fitted net in a `BaseEstimator + RegressorMixin` subclass
   that only exposes `fit` / `predict`. `RegressorMixin` sets
   `_estimator_type = "regressor"` and the missing `predict_proba` keeps the
   probe on the regression code path.

Together the wrapper looks like:

```python
class SkorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, net, lookback):
        self.net = net
        self.lookback = lookback

    def _to_3d(self, X):
        X = np.asarray(X, dtype="float32")
        return X[..., np.newaxis] if X.ndim == 2 else X

    def fit(self, X, y):
        self.net.fit(self._to_3d(X), y)
        return self

    def predict(self, X):
        return self.net.predict(self._to_3d(X)).ravel()
```

Train the skorch net on the native 3D tensor as usual, then build the
`EstimatorReport` with the 2D view (`X.squeeze(-1)`) and pass the wrapper as the
estimator. `project.put(key, report)` then succeeds for both LSTM and MLP
modules.
