"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
produced in earlier labs:

    1. Regression       -- SME monthly revenue prediction
    2. Classification   -- SME credit risk category
    3. Clustering       -- Mall customer segment assignment
    4. Association Rule -- Product recommendation from a cart
       Mining (ARM)

The point of this file is to show you that a fitted scikit-learn /
imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
missing values, encode categorical columns, scale numeric columns, and (for
the classifier) apply SMOTE only during training -- never during inference.
Because all of that preprocessing lives inside the Pipeline itself, this file
never manually label-encodes anything, never manually engineers date features,
and never manually reorders columns before calling .predict(). It builds a
DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
This is the main lesson of the lab: "the pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------
MODEL_DIR = "./model"

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "lasso_regressor_for_farm_income.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "logisticregression_classifier_for_invoice_payment_risk.joblib")
)


clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_farmer_segmentation.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
clustering_feature_cols = clustering_artifact["feature_cols"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]  # {frozenset(item_ids): [(item_id, confidence, lift), ...]}
item_lookup = arm_artifact["item_lookup"]                      # DataFrame indexed by item_id
ITEM_NAME_TO_ID = {str(name).lower(): item_id for item_id, name in item_lookup["item_name"].items()}

print("All the four artifacts have been loaded. The API is ready.")

# ---------------------------- COPY TO HERE ----------------------------------


import math

import numpy as np

REG_NUMERIC = ["years_farming_experience", "soil_quality_score", "num_workers_employed",
               "market_distance_km", "rainfall_mm"]
REG_NUMERIC_OPTIONAL = ["irrigation_maintenance_cost_kes"]   # leave out when the farm has no irrigation
REG_CATEGORICAL = ["crop_type", "county", "cooperative_member"]
REG_CATEGORICAL_OPTIONAL = ["irrigation_type"]               # leave out when the farm has no irrigation

CLF_NUMERIC = ["payment_terms_days", "invoice_amount_kes", "client_relationship_years",
               "prior_late_payments_count", "discount_offered_pct",
               "days_outstanding_avg_historical", "economic_sector_growth_rate_pct"]
CLF_NUMERIC_OPTIONAL = ["guarantee_value_kes"]               # leave out for an unsecured invoice
CLF_CATEGORICAL = ["client_industry", "payment_method", "has_dispute_history",
                   "collateral_or_guarantee", "client_credit_rating", "client_company_size"]
# The classifier's encoder has fixed categories, so these two are checked first.
CLF_ALLOWED = {
    "client_credit_rating": ["Poor", "Fair", "Good", "Excellent"],
    "client_company_size": ["Small", "Medium", "Large", "Enterprise"],
}


def _as_number(payload, field, required=True):
    value = payload.get(field)
    if value is None or value == "":
        if required:
            raise ValueError(f"Missing required field: '{field}'.")
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number, got {value!r}.")
    if not math.isfinite(number):
        raise ValueError(f"'{field}' must be a finite number.")
    return number


def _as_text(payload, field, required=True):
    value = payload.get(field)
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"Missing required field: '{field}'.")
        return float("nan")
    return str(value).strip()


def _build_row(payload, numeric, numeric_optional, categorical, categorical_optional):
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object with Content-Type: application/json.")
    row = {f: _as_number(payload, f) for f in numeric}
    row.update({f: _as_number(payload, f, required=False) for f in numeric_optional})
    row.update({f: _as_text(payload, f) for f in categorical})
    row.update({f: _as_text(payload, f, required=False) for f in categorical_optional})
    return pd.DataFrame([row])


# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/farm-income-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"years_farming_experience": 15, "soil_quality_score": 7,
#             "num_workers_employed": 6, "market_distance_km": 12,
#             "rainfall_mm": 900, "crop_type": "Tea", "county": "Kericho",
#             "cooperative_member": "Yes"}'
#
# Note: the pipeline was trained on log1p(income), so np.expm1 turns the
# prediction back into KES. irrigation_type and irrigation_maintenance_cost_kes
# are left out on purpose for a farm with no irrigation; the pipeline's own
# imputer handles the gap.
@app.route("/api/v1/models/farm-income-regressor/predictions", methods=["POST"])
def predict_farm_income():
    try:
        input_row = _build_row(request.get_json(silent=True), REG_NUMERIC,
                               REG_NUMERIC_OPTIONAL, REG_CATEGORICAL,
                               REG_CATEGORICAL_OPTIONAL)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    predicted_log_income = regression_pipeline.predict(input_row)[0]
    return jsonify({
        "predicted_annual_farm_income_kes": round(float(np.expm1(predicted_log_income)), 2),
        "model": "Lasso Regressor",
    })


# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/invoice-risk-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"client_industry": "Technology", "client_credit_rating": "Excellent",
#             "client_company_size": "Enterprise", "payment_method": "Bank Transfer",
#             "has_dispute_history": "No", "collateral_or_guarantee": "Yes",
#             "payment_terms_days": 15, "invoice_amount_kes": 400000,
#             "client_relationship_years": 8, "prior_late_payments_count": 0,
#             "discount_offered_pct": 2.0, "days_outstanding_avg_historical": 12,
#             "guarantee_value_kes": 450000, "economic_sector_growth_rate_pct": 4.5}'
@app.route("/api/v1/models/invoice-risk-classifier/predictions", methods=["POST"])
def predict_invoice_risk():
    payload = request.get_json(silent=True)
    try:
        if isinstance(payload, dict):
            for field, allowed in CLF_ALLOWED.items():
                value = _as_text(payload, field)
                if value not in allowed:
                    raise ValueError(f"'{field}' must be one of {allowed}, got {value!r}.")
        input_row = _build_row(payload, CLF_NUMERIC, CLF_NUMERIC_OPTIONAL,
                               CLF_CATEGORICAL, [])
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_

    return jsonify({
        "predicted_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Logistic Regression (SMOTE applied at training only)",
    })



# 3. CLUSTERING -- assign a new farmer to an existing segment (k = 5)
#
# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/farmer-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"farm_size_acres": 1.5, "annual_yield_tonnes": 16,
#             "income_per_acre_kes": 52000, "years_farming_experience": 6,
#             "market_access_score": 70}'
#
# Note: K-Means does not bundle its own scaler or feature engineering, so this
# route repeats the notebook's steps in the same order: yield per acre, log1p
# on the skewed columns, scaler.transform, then kmeans.predict. Nothing is
# fitted here.
CLUSTER_INPUTS = ["farm_size_acres", "annual_yield_tonnes", "income_per_acre_kes",
                  "years_farming_experience", "market_access_score"]


@app.route("/api/v1/models/farmer-segmenter/predictions", methods=["POST"])
def predict_farmer_segment():
    import math

    import numpy as np

    payload = request.get_json(silent=True)
    try:
        if not isinstance(payload, dict):
            raise ValueError("Send a JSON object with Content-Type: application/json.")
        values = {}
        for field in CLUSTER_INPUTS:
            raw = payload.get(field)
            if raw is None or raw == "":
                raise ValueError(f"Missing required field: '{field}'.")
            try:
                values[field] = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"'{field}' must be a number, got {raw!r}.")
            if not math.isfinite(values[field]):
                raise ValueError(f"'{field}' must be a finite number.")
        if values["farm_size_acres"] <= 0:
            raise ValueError("'farm_size_acres' must be greater than 0.")
        for field in ("annual_yield_tonnes", "income_per_acre_kes"):
            if values[field] < 0:
                raise ValueError(f"'{field}' cannot be negative.")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    yield_per_acre = values["annual_yield_tonnes"] / values["farm_size_acres"]
    input_row = pd.DataFrame([{
        "farm_size_acres_log": np.log1p(values["farm_size_acres"]),
        "yield_per_acre_log": np.log1p(yield_per_acre),
        "income_per_acre_log": np.log1p(values["income_per_acre_kes"]),
        "years_farming_experience": values["years_farming_experience"],
        "market_access_score": values["market_access_score"],
    }])[clustering_feature_cols]

    scaled_input = clustering_scaler.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": "K-Means (k=5, standardized features)",
    })

# 4. ASSOCIATION RULE MINING -- recommend menu items for a cart
#
# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/menu-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["Chicken Biryani", "Raita"]}'
#
#   curl -X POST http://127.0.0.1:5000/api/v1/models/menu-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["Pizza", "Soda"], "top_n": 2}'
#
# Note: there is no .predict() here. The mined rules were turned into a lookup
# dictionary (antecedent item set -> ranked consequents) in the notebook. This
# route only looks up and ranks. It never mines the rules again.
def _resolve_item(raw):
    """Accept an item id such as 'F10' or an item name such as 'Chicken Biryani'."""
    text = str(raw).strip()
    if text.upper() in item_lookup.index:
        return text.upper()
    if text.lower() in ITEM_NAME_TO_ID:
        return ITEM_NAME_TO_ID[text.lower()]
    raise ValueError(f"Unknown menu item: {raw!r}. See GET /api/v1/menu-items.")


@app.route("/api/v1/menu-items", methods=["GET"])
def menu_items():
    table = item_lookup.reset_index()[["item_id", "item_name", "category", "price_kes"]]
    return jsonify(table.sort_values("item_id").to_dict(orient="records"))


@app.route("/api/v1/models/menu-recommender/predictions", methods=["POST"])
def recommend_menu_items():
    payload = request.get_json(silent=True)
    try:
        if not isinstance(payload, dict):
            raise ValueError("Send a JSON object with Content-Type: application/json.")
        raw_cart = payload.get("cart")
        if not isinstance(raw_cart, list) or not raw_cart:
            raise ValueError("Provide a non-empty list under 'cart' (item ids or names).")
        try:
            top_n = int(payload.get("top_n", 3))
        except (TypeError, ValueError):
            raise ValueError("'top_n' must be a whole number.")
        cart = {_resolve_item(item) for item in raw_cart}
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for item_id, confidence, lift in consequent_list:
                if item_id not in cart:
                    if item_id not in candidates or confidence > candidates[item_id][0]:
                        candidates[item_id] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "item_id": str(item_id),
            "item_name": item_lookup.loc[item_id, "item_name"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for item_id, (confidence, lift) in ranked
    ]

    return jsonify({
        "cart": sorted(item_lookup.loc[item_id, "item_name"] for item_id in cart),
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=True)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))
