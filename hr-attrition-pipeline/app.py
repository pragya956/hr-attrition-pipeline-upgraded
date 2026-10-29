"""Flask application: pages + JSON API. All ML logic lives in the ml/ package."""
from __future__ import annotations

import logging
import threading

from flask import Flask, Response, jsonify, render_template, request

from config import Config
from ml import analytics
from ml.data_loader import DatasetError
from ml.predict import ValidationError, predict_profile
from ml.service import ModelService
from ml.train import ArtifactError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hr-app")

PAGES = [
    ("index", "/", "Overview", "bi-grid-1x2"),
    ("dataset", "/dataset", "Dataset Explorer", "bi-table"),
    ("analytics", "/analytics", "Attrition Insights", "bi-bar-chart-line"),
    ("training", "/training", "Model Training", "bi-cpu"),
    ("evaluation", "/evaluation", "Model Evaluation", "bi-speedometer2"),
    ("prediction", "/prediction", "Predict Attrition", "bi-person-check"),
    ("feature_importance", "/feature-importance", "Feature Importance", "bi-sliders"),
    ("about", "/about", "About Project", "bi-info-circle"),
]

# Fields shown on the prediction form (labels / grouping only; options & ranges come from the model schema)
FORM_FIELDS = [
    ("Age", "Age"), ("Department", "Department"), ("JobRole", "Job role"), ("MonthlyIncome", "Monthly income (USD)"),
    ("JobLevel", "Job level"), ("YearsAtCompany", "Years at company"), ("YearsInCurrentRole", "Years in current role"),
    ("YearsSinceLastPromotion", "Years since last promotion"), ("OverTime", "Overtime"),
    ("BusinessTravel", "Business travel"), ("JobSatisfaction", "Job satisfaction"),
    ("EnvironmentSatisfaction", "Environment satisfaction"), ("WorkLifeBalance", "Work-life balance"),
    ("JobInvolvement", "Job involvement"), ("DistanceFromHome", "Distance from home (km)"),
    ("MaritalStatus", "Marital status"), ("EducationField", "Education field"), ("StockOptionLevel", "Stock option level"),
]


def create_app(test_config: dict | None = None, dataset_provider=None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    service = ModelService(app.config, dataset_provider)
    app.extensions["service"] = service

    if app.config.get("BOOTSTRAP_ON_START", True):
        threading.Thread(target=service.bootstrap, daemon=True, name="bootstrap").start()

    # ----------------------------------------------------------------- helpers
    def err(message, status=400, **extra):
        return jsonify({"error": message, **extra}), status

    def dataset():
        return service.get_dataset()

    def filters_from_args():
        return {k: request.args.get(k, "").strip() for k in analytics.FILTER_COLUMNS if request.args.get(k)}

    def int_arg(name, default):
        raw = request.args.get(name)
        if raw in (None, ""):
            return default
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"Query parameter '{name}' must be an integer") from None

    @app.errorhandler(DatasetError)
    def _dataset_error(e):
        log.error("Dataset error: %s", e)
        return err(str(e), 503, code="dataset_unavailable")

    @app.errorhandler(ArtifactError)
    def _artifact_error(e):
        return err(str(e), 409, code="model_unavailable")

    @app.errorhandler(ValueError)
    def _value_error(e):
        return err(str(e), 400)

    @app.errorhandler(404)
    def _404(e):
        if request.path.startswith("/api/"):
            return err("Endpoint not found", 404)
        return render_template("about.html", active="about", pages=PAGES), 404

    @app.errorhandler(500)
    def _500(e):
        log.exception("Unhandled error")
        return err("Internal server error", 500)

    @app.context_processor
    def inject():
        return {"pages": PAGES}

    # ----------------------------------------------------------------- pages
    for endpoint, path, title, _icon in PAGES:
        template = "index.html" if endpoint == "index" else f"{endpoint}.html"
        app.add_url_rule(path, endpoint, (lambda t, e, ti: lambda: render_template(t, active=e, title=ti))(template, endpoint, title))

    # ----------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "dataset": service.dataset.state,
                        "model_available": service.artifact is not None,
                        "training": service.train_state["status"]})

    @app.get("/api/model/status")
    def model_status():
        return jsonify(analytics.jsonable(service.status()))

    @app.get("/api/dataset/summary")
    def ds_summary():
        df, info = dataset()
        return jsonify(analytics.dataset_summary(df, info))

    @app.get("/api/dataset/columns")
    def ds_columns():
        df, _ = dataset()
        return jsonify(analytics.column_summary(df))

    @app.get("/api/dataset/preview")
    def ds_preview():
        df, _ = dataset()
        cols = [c for c in request.args.get("columns", "").split(",") if c]
        return jsonify(analytics.preview(
            df, page=int_arg("page", 1), page_size=int_arg("page_size", 15),
            search=request.args.get("search", ""), sort_by=request.args.get("sort_by") or None,
            sort_dir=request.args.get("sort_dir", "asc"), columns=cols or None,
            filters=filters_from_args(), max_page_size=app.config["MAX_PAGE_SIZE"]))

    @app.get("/api/dataset/column-chart")
    def ds_column_chart():
        df, _ = dataset()
        col = request.args.get("column")
        if not col:
            return err("Query parameter 'column' is required")
        return jsonify(analytics.column_chart(df, col, int_arg("bins", 12)))

    @app.get("/api/dataset/duplicates")
    def ds_duplicates():
        df, _ = dataset()
        return jsonify(analytics.duplicates(df))

    @app.get("/api/dataset/download")
    def ds_download():
        df, _ = dataset()
        csv = analytics.filtered_csv(df, filters_from_args(), request.args.get("search", ""))
        return Response(csv, mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=hr_attrition_filtered.csv"})

    @app.get("/api/overview")
    def overview():
        df, _ = dataset()
        return jsonify(analytics.overview(df, service.artifact))

    @app.get("/api/analytics/attrition")
    def attrition():
        df, _ = dataset()
        return jsonify(analytics.attrition_insights(df, filters_from_args()))

    def _public_art(art):
        return {"model_name": art["model_name"], "best_params": art.get("best_params", {}),
                "selection_metric": art["selection_metric"], "trained_at": art["trained_at"], "seed": art["seed"],
                "threshold": art["threshold"], "params": art["params"], "split": art["split"]}

    @app.get("/api/models/metrics")
    def metrics():
        art = service.require_artifact()
        return jsonify(analytics.jsonable({**_public_art(art), "cv_results": art["cv_results"],
                                           "test_metrics": art["test_metrics"], "curves": art["curves"],
                                           "threshold_sweep": art["threshold_sweep"]}))

    @app.get("/api/models/feature-importance")
    def importance():
        art = service.require_artifact()
        return jsonify(analytics.jsonable({"model_name": art["model_name"], **art["importance"]}))

    @app.get("/api/model/schema")
    def schema():
        art = service.require_artifact()
        feats = {f["name"]: f for f in art["schema"]["features"]}
        fields = [{**feats[n], "label": label} for n, label in FORM_FIELDS if n in feats]
        return jsonify(analytics.jsonable({
            "fields": fields, "threshold": art["threshold"], "model_name": art["model_name"],
            "likert": {k: {str(i): l for i, l in v.items()} for k, v in analytics.LIKERT.items()}}))

    @app.post("/api/models/train")
    def train():
        body = request.get_json(silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return err("Request body must be a JSON object")
        params = service.normalize_params(body)  # raises ValueError -> 400
        dataset()  # fail fast with 503 if the dataset cannot be loaded
        if not service.start_training(params, background=True):
            return err("A training job is already running.", 409, code="training_in_progress")
        return jsonify({"status": "started", "params": params}), 202

    @app.post("/api/model/threshold")
    def set_threshold():
        body = request.get_json(silent=True) or {}
        try:
            value = float(body.get("threshold"))
        except (TypeError, ValueError):
            return err("'threshold' must be a number between 0.01 and 0.99")
        if not 0.01 <= value <= 0.99:
            return err("'threshold' must be between 0.01 and 0.99")
        art = service.set_threshold(value)
        return jsonify({"threshold": art["threshold"]})

    @app.post("/api/predict")
    def predict():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return err("Request body must be a JSON object of feature values")
        art = service.require_artifact()
        threshold = body.pop("_threshold", None)
        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                return err("'_threshold' must be a number")
            if not 0.01 <= threshold <= 0.99:
                return err("'_threshold' must be between 0.01 and 0.99")
        try:
            return jsonify(predict_profile(art, body, threshold))
        except ValidationError as exc:
            return err("Invalid input", 422, fields=exc.errors)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
