"""Input validation and prediction against the saved pipeline."""
from __future__ import annotations

import math

import pandas as pd


class ValidationError(ValueError):
    def __init__(self, errors: dict):
        super().__init__("Invalid input")
        self.errors = errors


def validate_profile(schema: dict, payload: dict) -> tuple[pd.DataFrame, dict, list[str]]:
    """Validate a profile against the training schema.

    Unknown fields are rejected (incompatible schema). Fields omitted by the caller are filled
    with the training median / mode so the model always receives the exact training columns.
    """
    if not isinstance(payload, dict):
        raise ValidationError({"_": "Request body must be a JSON object of feature values."})
    feats = {f["name"]: f for f in schema["features"]}
    errors, warnings, row, provided = {}, [], {}, set()
    unknown = [k for k in payload if k not in feats]
    if unknown:
        raise ValidationError({k: "Unknown feature; not part of the trained schema." for k in unknown})
    for name, f in feats.items():
        value = payload.get(name)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            row[name] = f["default"]
            continue
        provided.add(name)
        if f["kind"] == "numeric":
            try:
                num = float(value)
                if math.isnan(num) or math.isinf(num):
                    raise ValueError
            except (TypeError, ValueError):
                errors[name] = "Must be a number."
                continue
            if f.get("integer") and num != int(num):
                errors[name] = "Must be a whole number."
                continue
            lo, hi = f["min"], f["max"]
            if f.get("ordinal"):
                if not lo <= num <= hi:
                    errors[name] = f"Must be between {lo:g} and {hi:g}."
                    continue
            else:
                if num < 0:
                    errors[name] = "Must not be negative."
                    continue
                if num > hi * 2 + 1:
                    errors[name] = f"Unrealistically large (training maximum is {hi:g})."
                    continue
                if not lo <= num <= hi:
                    warnings.append(f"{name}={num:g} lies outside the training range {lo:g}-{hi:g}; "
                                    "the estimate is an extrapolation.")
            row[name] = num
        else:
            text = str(value).strip()
            if text not in f["options"]:
                warnings.append(f"'{text}' is an unseen category for {name}; it is encoded as all zeros.")
            row[name] = text
    if errors:
        raise ValidationError(errors)
    df = pd.DataFrame([row], columns=schema["columns"])
    return df, row, warnings


def predict_profile(artifact: dict, payload: dict, threshold: float | None = None) -> dict:
    schema = artifact["schema"]
    df, row, warnings = validate_profile(schema, payload)
    thr = artifact["threshold"] if threshold is None else float(threshold)
    pipe = artifact["pipeline"]
    proba = float(pipe.predict_proba(df)[0, 1])

    # Local sensitivity: reset each *provided* field to the training default, see how the estimate moves.
    drivers = []
    for name in payload:
        if name not in row or payload.get(name) in (None, ""):
            continue
        feat = next(f for f in schema["features"] if f["name"] == name)
        if row[name] == feat["default"]:
            continue
        alt = df.copy()
        alt.at[0, name] = feat["default"]
        p_alt = float(pipe.predict_proba(alt)[0, 1])
        drivers.append({"feature": name, "value": row[name], "typical": feat["default"],
                        "effect": proba - p_alt})
    drivers.sort(key=lambda d: abs(d["effect"]), reverse=True)

    if proba >= 0.66:
        band = "high"
    elif proba >= 0.33:
        band = "moderate"
    else:
        band = "low"
    return {
        "prediction": "Likely to leave" if proba >= thr else "Likely to stay",
        "predicted_class": int(proba >= thr),
        "probability": proba,
        "threshold": thr,
        "risk_band": band,
        "model": artifact["model_name"],
        "drivers": drivers[:6],
        "warnings": warnings,
        "disclaimer": ("Statistical estimate from historical aggregate data. Not a certain outcome and "
                       "not suitable for employment, disciplinary or termination decisions about individuals."),
    }
