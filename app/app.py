"""
app.py — Interface Flask
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

artifacts = {
    "feature_names": joblib.load(os.path.join(BASE, "models/feature_names.pkl")),
    "pca":           joblib.load(os.path.join(BASE, "models/pca.pkl")),
    "kmeans":        joblib.load(os.path.join(BASE, "models/kmeans_model.pkl")),
    "classifier":    joblib.load(os.path.join(BASE, "models/rf_classifier.pkl")),
    "regressor":     joblib.load(os.path.join(BASE, "models/rf_regressor.pkl")),
    "scaler":        joblib.load(os.path.join(BASE, "models/scaler.pkl")),
}

CLUSTER_LABELS = {
    0: "Clients perdus (inactifs)",
    1: "Clients fidèles",
    2: "Clients réguliers",
    3: "Clients à risque",
}


def predict_client(form_data):
    feature_names = artifacts["feature_names"]

    # Valeurs saisies
    recency      = float(form_data.get("recency", 0))
    frequency    = float(form_data.get("frequency", 1))
    monetary     = float(form_data.get("monetary", 0))
    tenure       = float(form_data.get("tenure", 0))
    satisfaction = float(form_data.get("satisfaction", 3))
    tickets      = float(form_data.get("tickets", 0))
    age          = float(form_data.get("age", 35))

    # Construire vecteur avec toutes les features finales à 0
    X = pd.DataFrame([{col: 0.0 for col in feature_names}])

    # Remplir les features disponibles depuis le formulaire
    # (les autres restent à 0 — valeur neutre après scaling)
    values = {
        "Recency":             recency,
        "Frequency":           frequency,
        "MonetaryTotal":       monetary,
        "CustomerTenureDays":  tenure,
        "SatisfactionScore":   satisfaction,
        "SupportTicketsCount": tickets,
        "Age":                 age,
        "MonetaryPerDay":      monetary / (recency + 1),
        "AvgBasketValue":      monetary / (frequency + 1),
        "TenureRatio":         recency / (tenure + 1),
    }

    for col, val in values.items():
        if col in X.columns:
            X[col] = val

    # Normalisation manuelle des features numériques clés
    # On utilise mean_ et scale_ du scaler pour normaliser
    scaler = artifacts["scaler"]
    scaler_cols = list(scaler.feature_names_in_)

    for col in feature_names:
        if col in scaler_cols and col in values:
            idx = scaler_cols.index(col)
            mean  = scaler.mean_[idx]
            scale = scaler.scale_[idx]
            X[col] = (values[col] - mean) / scale

    # Prédiction churn
    churn_pred  = int(artifacts["classifier"].predict(X)[0])
    churn_proba = float(artifacts["classifier"].predict_proba(X)[0][1])

    # Prédiction cluster
    X_clipped = X.clip(lower=-5, upper=5)
    X_pca     = artifacts["pca"].transform(X_clipped)
    cluster   = int(artifacts["kmeans"].predict(X_pca)[0])

    # Prédiction monétaire
    X_reg = X.drop(columns=["MonetaryTotal"]) if "MonetaryTotal" in X.columns else X.copy()
    monetary_pred = float(artifacts["regressor"].predict(X_reg)[0])

    # Niveau de risque
    if churn_proba >= 0.75:
        risque, couleur = "ÉLEVÉ", "danger"
    elif churn_proba >= 0.50:
        risque, couleur = "MOYEN", "warning"
    elif churn_proba >= 0.25:
        risque, couleur = "FAIBLE", "info"
    else:
        risque, couleur = "TRÈS FAIBLE", "success"

    return {
        "churn_pred":    churn_pred,
        "churn_proba":   round(churn_proba * 100, 1),
        "churn_label":   "Churner" if churn_pred == 1 else "Fidèle",
        "cluster":       cluster,
        "cluster_label": CLUSTER_LABELS.get(cluster, f"Cluster {cluster}"),
        "monetary":      round(monetary_pred, 3),
        "risque":        risque,
        "couleur":       couleur,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form
        result = predict_client(form_data)

    return render_template("index.html", result=result, form=form_data)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)