import json

import pytest
from ml.train import save_artifact


def jpost(client, url, body):
    return client.post(url, data=json.dumps(body), content_type="application/json")


def test_health_and_pages(service_app):
    c = service_app.test_client()
    assert c.get("/api/health").json["status"] == "ok"
    for path in ["/", "/dataset", "/analytics", "/training", "/evaluation", "/prediction", "/feature-importance", "/about"]:
        assert c.get(path).status_code == 200, path
    r = c.get("/api/nope")
    assert r.status_code == 404 and "error" in r.json


def test_dataset_endpoints(service_app):
    c = service_app.test_client()
    s = c.get("/api/dataset/summary").json
    assert s["rows"] == 400 and "EmployeeNumber" not in s["column_names"] and s["target"]["counts"][1] > 0
    cols = c.get("/api/dataset/columns").json
    assert any(x["name"] == "MonthlyIncome" and x["missing"] == 1 for x in cols["columns"])
    p = c.get("/api/dataset/preview?page=2&page_size=10&sort_by=Age&sort_dir=desc").json
    assert p["page"] == 2 and len(p["rows"]) == 10 and "EmployeeNumber" not in p["rows"][0]
    assert p["rows"][0]["Age"] >= p["rows"][1]["Age"]
    f = c.get("/api/dataset/preview?overtime=Yes&attrition=Yes&page_size=1000").json
    assert f["page_size"] == 100 and all(r["OverTime"] == "Yes" for r in f["rows"])
    assert c.get("/api/dataset/preview?search=zzzzzz").json["total"] == 0
    assert c.get("/api/dataset/preview?sort_by=Bogus").status_code == 400
    assert c.get("/api/dataset/preview?page=abc").status_code == 400
    assert c.get("/api/dataset/column-chart?column=Age").json["labels"]
    assert c.get("/api/dataset/column-chart?column=Nope").status_code == 400
    assert c.get("/api/dataset/duplicates").status_code == 200


def test_download_filtered_csv(service_app):
    r = service_app.test_client().get("/api/dataset/download?overtime=Yes")
    assert r.mimetype == "text/csv"
    lines = r.data.decode().strip().splitlines()
    assert "EmployeeNumber" not in lines[0] and len(lines) > 1


def test_analytics_and_empty_filter(service_app):
    c = service_app.test_client()
    a = c.get("/api/analytics/attrition").json
    assert a["overall"]["total"] == 400 and a["by_age_group"]["labels"]
    assert sum(a["by_department"]["total"]) == 400
    e = c.get("/api/analytics/attrition?department=Nonexistent").json
    assert e["empty"] is True
    assert c.get("/api/overview").json["total"] == 400


def test_model_endpoints_before_training(service_app):
    c = service_app.test_client()
    for url in ["/api/models/metrics", "/api/models/feature-importance", "/api/model/schema"]:
        r = c.get(url)
        assert r.status_code == 409 and "error" in r.json
    assert jpost(c, "/api/predict", {"Age": 30}).status_code == 409


def test_train_validation_and_duplicate_protection(service_app):
    c = service_app.test_client()
    assert jpost(c, "/api/models/train", {"n_splits": 99}).status_code == 400
    assert jpost(c, "/api/models/train", {"selection_metric": "accuracy"}).status_code == 400
    assert jpost(c, "/api/models/train", {"class_weight": "yes"}).status_code == 400
    svc = service_app.extensions["service"]
    assert svc._train_lock.acquire(blocking=False)  # simulate a running job
    try:
        r = jpost(c, "/api/models/train", {})
        assert r.status_code == 409 and r.json["code"] == "training_in_progress"
    finally:
        svc._train_lock.release()


def test_full_train_predict_flow(trained_app):
    c = trained_app.test_client()
    st = c.get("/api/model/status").json
    assert st["training"]["status"] == "completed" and st["model"]["available"]
    m = c.get("/api/models/metrics").json
    assert set(m["cv_results"]) == {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    assert m["test_metrics"]["confusion_matrix"]["matrix"] and m["curves"]["roc"]["fpr"]
    assert c.get("/api/models/feature-importance").json["permutation"]["items"]
    schema = c.get("/api/model/schema").json
    assert any(f["name"] == "Department" and f["options"] for f in schema["fields"])
    ok = jpost(c, "/api/predict", {"Age": 29, "OverTime": "Yes", "MonthlyIncome": 2500})
    assert ok.status_code == 200 and 0 <= ok.json["probability"] <= 1
    assert jpost(c, "/api/predict", {"Age": "x"}).status_code == 422
    assert jpost(c, "/api/predict", {"Bogus": 1}).status_code == 422
    assert jpost(c, "/api/predict", {"Age": 30, "_threshold": 5}).status_code == 400
    assert c.post("/api/predict", data="not json", content_type="application/json").status_code == 400
    t = jpost(c, "/api/model/threshold", {"threshold": 0.3})
    assert t.status_code == 200 and c.get("/api/model/schema").json["threshold"] == 0.3
    assert jpost(c, "/api/model/threshold", {"threshold": 2}).status_code == 400


def test_background_training_reports_real_status(service_app):
    import time
    c = service_app.test_client()
    r = jpost(c, "/api/models/train", {"n_splits": 3})
    assert r.status_code == 202
    for _ in range(120):
        st = c.get("/api/model/status").json["training"]
        if st["status"] != "running":
            break
        time.sleep(0.5)
    assert st["status"] == "completed" and st["progress"] == 1.0 and st["logs"]


def test_corrupted_artifact_is_handled(service_app):
    svc = service_app.extensions["service"]
    svc.artifact_path.write_bytes(b"garbage")
    assert svc.load_existing() is False
    assert "corrupted" in svc.artifact_note
    assert service_app.test_client().get("/api/models/metrics").status_code == 409


def test_stale_artifact_detected_when_dataset_changes(trained_app):
    svc = trained_app.extensions["service"]
    svc.artifact = None
    svc.dataset.df = svc.dataset.df.iloc[:-1].copy()  # dataset changed
    assert svc.load_existing() is False and "Dataset changed" in svc.artifact_note


def test_dataset_unavailable_returns_503(tmp_path):
    from app import create_app
    from ml.data_loader import DatasetError

    def broken():
        raise DatasetError("Kaggle unreachable")
    app = create_app({"TESTING": True, "MODEL_DIR": tmp_path, "BOOTSTRAP_ON_START": False}, dataset_provider=broken)
    r = app.test_client().get("/api/dataset/summary")
    assert r.status_code == 503 and "Kaggle" in r.json["error"]
    assert app.test_client().get("/api/health").status_code == 200
