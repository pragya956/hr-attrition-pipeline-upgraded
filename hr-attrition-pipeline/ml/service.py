"""Application service: owns the dataset cache, the loaded artifact and background training."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from ml import data_loader
from ml.data_loader import DatasetCache, DatasetError
from ml.train import (ArtifactError, TrainingError, dataset_signature, load_artifact, save_artifact,
                      train_pipeline)

log = logging.getLogger(__name__)


class ModelService:
    def __init__(self, config, dataset_provider=None):
        self.config = config
        self.artifact_path = Path(config["MODEL_DIR"]) / config["ARTIFACT_NAME"]
        loader = dataset_provider or (lambda: data_loader.load_dataset(config["DATA_DIR"], config["LOCAL_CSV_NAME"]))
        self.dataset = DatasetCache(loader)
        self.artifact: dict | None = None
        self.artifact_note = "No model loaded yet."
        self._art_lock = threading.RLock()
        self._train_lock = threading.Lock()
        self.train_state = {"status": "idle", "progress": 0.0, "logs": [], "error": None,
                            "started_at": None, "finished_at": None, "params": None}

    # ------------------------------------------------------------- dataset / artifact
    def get_dataset(self):
        return self.dataset.get()

    def load_existing(self) -> bool:
        """Load the saved artifact if it exists and matches the current dataset."""
        try:
            df, _ = self.get_dataset()
            sig = dataset_signature(df)
        except DatasetError:
            sig = None  # dataset unavailable: still allow predictions from a valid artifact
        try:
            art = load_artifact(self.artifact_path, expected_signature=sig)
        except ArtifactError as exc:
            with self._art_lock:
                self.artifact, self.artifact_note = None, str(exc)
            log.warning("Artifact not usable: %s", exc)
            return False
        with self._art_lock:
            self.artifact, self.artifact_note = art, "Loaded saved model."
        return True

    def bootstrap(self):
        """Startup: load dataset, load artifact or train a fresh one."""
        try:
            self.get_dataset()
        except DatasetError as exc:
            log.error("Dataset unavailable at startup: %s", exc)
        if not self.load_existing() and self.config.get("AUTO_TRAIN", True) and self.dataset.df is not None:
            self.start_training({}, background=True)

    def require_artifact(self) -> dict:
        with self._art_lock:
            if self.artifact is None:
                raise ArtifactError(self.artifact_note or "No trained model is available. Train a model first.")
            return self.artifact

    def set_threshold(self, value: float) -> dict:
        art = self.require_artifact()
        with self._art_lock:
            art["threshold"] = float(value)
            save_artifact(art, self.artifact_path)
        return art

    # ------------------------------------------------------------- training
    def normalize_params(self, raw: dict) -> dict:
        from ml.evaluate import SELECTION_METRICS
        c = self.config
        def as_int(key, default, lo, hi):
            v = raw.get(key, default)
            if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                raise ValueError(f"'{key}' must be an integer")
            try:
                v = int(v)
            except ValueError:
                raise ValueError(f"'{key}' must be an integer") from None
            if not lo <= v <= hi:
                raise ValueError(f"'{key}' must be between {lo} and {hi}")
            return v
        def as_bool(key, default):
            v = raw.get(key, default)
            if not isinstance(v, bool):
                raise ValueError(f"'{key}' must be true or false")
            return v
        metric = raw.get("selection_metric", c["DEFAULT_SELECTION_METRIC"])
        if metric not in SELECTION_METRICS:
            raise ValueError(f"'selection_metric' must be one of {SELECTION_METRICS}")
        return {"n_splits": as_int("n_splits", c["DEFAULT_FOLDS"], 2, 10),
                "seed": as_int("seed", c["DEFAULT_SEED"], 0, 2**31 - 1),
                "class_weight": as_bool("class_weight", True), "tune": as_bool("tune", False),
                "selection_metric": metric}

    def is_training(self) -> bool:
        return self.train_state["status"] == "running"

    def start_training(self, raw_params: dict, background: bool = True) -> bool:
        """Returns False if a training job is already running (no duplicates)."""
        params = self.normalize_params(raw_params)
        if not self._train_lock.acquire(blocking=False):
            return False
        self.train_state.update(status="running", progress=0.0, logs=[], error=None, params=params,
                                started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                finished_at=None)
        runner = threading.Thread(target=self._run, args=(params,), daemon=True, name="trainer")
        if background:
            runner.start()
        else:
            self._run(params)
        return True

    def _log(self, msg):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.train_state["logs"].append(f"[{stamp}] {msg}")
        log.info(msg)

    def _run(self, params):
        try:
            df, _ = self.get_dataset()
            self._log(f"Training started with {params}")
            art = train_pipeline(df, threshold=self.config["DEFAULT_THRESHOLD"], test_size=self.config["TEST_SIZE"],
                                 log_fn=self._log, progress_fn=lambda f: self.train_state.update(progress=round(f, 3)),
                                 **params)
            save_artifact(art, self.artifact_path)
            with self._art_lock:
                self.artifact, self.artifact_note = art, "Freshly trained model."
            self._log("Artifact saved. Training complete.")
            self.train_state.update(status="completed", progress=1.0)
        except (TrainingError, DatasetError, ValueError) as exc:
            self._log(f"Training failed: {exc}")
            self.train_state.update(status="failed", error=str(exc))
        except Exception as exc:  # never leave the state stuck on 'running'
            log.exception("Unexpected training failure")
            self._log(f"Unexpected error: {exc}")
            self.train_state.update(status="failed", error=f"Unexpected error: {exc}")
        finally:
            self.train_state["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._train_lock.release()

    def status(self) -> dict:
        art = self.artifact
        return {
            "dataset": {"state": self.dataset.state, "error": self.dataset.error,
                        "rows": (self.dataset.info or {}).get("rows"),
                        "filename": (self.dataset.info or {}).get("filename")},
            "model": {"available": art is not None, "name": art["model_name"] if art else None,
                      "trained_at": art["trained_at"] if art else None, "threshold": art["threshold"] if art else None,
                      "note": self.artifact_note},
            "training": {k: self.train_state[k] for k in ("status", "progress", "error", "started_at", "finished_at", "params")}
                        | {"logs": list(self.train_state["logs"])},
        }
