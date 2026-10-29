# HR Attrition Analytics — Stratified ML Pipeline with Categorical Encoding

Full-stack Flask + scikit-learn app that predicts employee attrition on the **IBM HR Analytics Employee Attrition** dataset (Kaggle), downloaded automatically with KaggleHub. No fabricated data, no hardcoded metrics: every chart, metric and prediction comes from the real CSV and the trained pipeline.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python app.py            # open http://127.0.0.1:5000
```

On first start the app downloads the dataset (KaggleHub), caches a copy in `data/`, and trains models in the background if no valid artifact exists (watch the top-bar pills or the **Model Training** page). Train from the CLI instead with:

```bash
python -c "from app import create_app; s=create_app({'BOOTSTRAP_ON_START':False}).extensions['service']; s.start_training({}, background=False); print(s.status()['training']['status'])"
```

Run tests (no internet or Kaggle credentials needed): `pytest -q`

## Architecture

```
app.py            Flask routes + JSON API (no ML logic)
config.py         paths and defaults
ml/data_loader.py KaggleHub download, CSV discovery, schema validation, cache
ml/preprocessing.py  excluded columns, ColumnTransformer, model pipelines, tuning grids
ml/train.py       stratified split, CV comparison, selection, test evaluation, joblib persistence
ml/evaluate.py    imbalance-aware metrics, curves, CV summaries, threshold sweep
ml/predict.py     input validation + prediction + local sensitivity
ml/analytics.py   dataset explorer and attrition insights
ml/service.py     dataset cache, loaded artifact, background training job
templates/, static/  dashboard UI (vanilla JS, Chart.js, custom CSS)
tests/            pytest suite (mocked download, synthetic schema-only fixtures)
```

## ML design

- **Dataset acquisition.** `kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")`, then the CSV is located by scanning the returned folder for a file containing `Attrition`. If a later download fails, the cached copy of the *real* file in `data/` is used.
- **Excluded features.** `EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours`, plus any column constant in the training split. `EmployeeNumber` is also hidden from the explorer and CSV export.
- **Categorical encoding.** `ColumnTransformer`: numeric → median imputer (+ `StandardScaler` for logistic regression); categorical → most-frequent imputer + `OneHotEncoder(handle_unknown="ignore")`. Column types are detected from the training frame. Ordinal survey scores (1–4) are treated as numeric.
- **No leakage.** All fitted preprocessing lives inside each `Pipeline`, so `cross_validate` refits it per fold. Constant-column detection uses the training split only.
- **Splits & CV.** Stratified 80/20 split (`random_state=seed`, default 42); `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` on the training part.
- **Models.** Logistic Regression, Random Forest, HistGradientBoosting, each with its own pipeline. `class_weight="balanced"` is optional (no resampling, so validation/test data are never altered).
- **Tuning (optional).** `RandomizedSearchCV` runs *inside* every outer fold (nested CV), so CV scores stay unbiased.
- **Selection.** Highest mean CV score of a predefined metric (default PR-AUC / average precision). The held-out test set scores only the selected model, once.
- **Imbalance.** Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix and classification report are reported. The UI shows the majority-class baseline to make clear that high accuracy ≠ finding leavers.
- **Threshold.** Default 0.5, adjustable on the Prediction page (and savable) without retraining. The threshold trade-off chart uses out-of-fold predictions on training data, never the test set.
- **Importance.** Logistic coefficients (per 1 SD for numeric inputs), random-forest importances aggregated back to original features, and permutation importance on held-out data. These show association, not causation.
- **Persistence.** `models/hr_attrition_artifact.joblib` stores the fitted pipeline, schema, CV and test metrics, importances, seed, threshold, timestamp and a dataset signature. It is rejected and retrained if missing, corrupted, from another scikit-learn version, or if the dataset changed. Only this app-written file is ever loaded; uploads are not accepted.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness, dataset/model/training state |
| GET | `/api/model/status` | pills, training progress and logs |
| GET | `/api/dataset/summary` · `/columns` · `/duplicates` | profile, dtypes, missing values, stats |
| GET | `/api/dataset/preview` | `page, page_size(≤100), search, sort_by, sort_dir, columns, department, gender, job_role, overtime, attrition` |
| GET | `/api/dataset/column-chart?column=&bins=` | per-column distribution split by attrition |
| GET | `/api/dataset/download` | filtered CSV (same filters) |
| GET | `/api/overview` · `/api/analytics/attrition` | dashboard metrics; insights (accepts filters) |
| GET | `/api/models/metrics` · `/feature-importance` · `/api/model/schema` | CV/test results, curves; importances; form schema |
| POST | `/api/models/train` | JSON `{n_splits, seed, class_weight, tune, selection_metric}` → 202, 409 if already running |
| POST | `/api/model/threshold` | `{threshold}` |
| POST | `/api/predict` | feature JSON (+ optional `_threshold`) → 200 / 422 invalid fields / 409 no model |

Errors are always JSON `{"error": ...}`: 400 bad parameters, 409 no model or training in progress, 422 invalid prediction input, 503 dataset unavailable.

## Troubleshooting

- **Dataset: unavailable.** KaggleHub could not download. Check your connection; if Kaggle asks for credentials set `KAGGLE_USERNAME` and `KAGGLE_KEY` (or place `kaggle.json` in `~/.kaggle/`). Once downloaded, a cached copy in `data/` lets the app start offline.
- **“Model was trained with a different scikit-learn version.”** Retrain from the Training page.
- **Training seems slow.** Tuning (nested CV) multiplies the work; leave it off for quick runs.
- **Port in use.** Edit the port in the last line of `app.py`.
- The front end loads Chart.js, Bootstrap Icons and Inter from CDNs, so the browser needs internet access.

## Responsible use

Decision support for aggregate workforce analysis only. Not for employment, disciplinary or termination decisions about individuals.

## Verification and compatibility

- Python source can be syntax-checked with `python -m compileall -q .`.
- Browser JavaScript can be syntax-checked with `node --check static/js/app.js` (repeat for each file in `static/js/`).
- GitHub Actions runs the pytest suite on Python 3.10, 3.11, and 3.12.
- Saved model artifacts are checked for required evaluation metadata, feature schema, and a `predict_proba`-capable pipeline before use. If an artifact fails validation, retrain from **Model Training**.
- The test suite uses synthetic fixtures and mocked KaggleHub calls; the production app always loads the specified IBM HR Analytics dataset and does not substitute synthetic data.
