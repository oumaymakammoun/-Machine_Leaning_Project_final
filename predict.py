"""
predict.py
==========
Pipeline d'inférence — Retail E-commerce (Churn & Segmentation)

Usage :
    python predict.py                        # prédit sur X_test complet
    python predict.py --client 42            # prédit pour le client index 42
    python predict.py --csv mon_fichier.csv  # prédit sur un fichier externe
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts():
    """Charge tous les modèles et transformateurs sauvegardés."""
    print("\n  [LOAD] Chargement des artefacts...")

    artifacts = {
        "scaler":        joblib.load("models/scaler.pkl"),
        "knn_imputer":   joblib.load("models/knn_imputer.pkl"),
        "feature_names": joblib.load("models/feature_names.pkl"),
        "dropped_cols":  joblib.load("models/dropped_corr_cols.pkl"),
        "pca":           joblib.load("models/pca.pkl"),
        "kmeans":        joblib.load("models/kmeans_model.pkl"),
        "classifier":    joblib.load("models/rf_classifier.pkl"),
        "regressor":     joblib.load("models/rf_regressor.pkl"),
    }

    print(f"  ✓ {len(artifacts)} artefacts chargés")
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DE PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(X_raw, artifacts):
    """
    Applique le même pipeline que preprocessing.py sur de nouvelles données,
    puis retourne les prédictions des 3 modèles.

    Paramètres
    ----------
    X_raw : pd.DataFrame
        Données déjà preprocessées (sorties de preprocessing.py) OU
        données brutes issues de X_test.csv (déjà scaled).

    artifacts : dict
        Dictionnaire des artefacts chargés par load_artifacts().

    Retourne
    --------
    results : pd.DataFrame
        DataFrame avec colonnes :
        - churn_pred       : 0 (fidèle) ou 1 (churner)
        - churn_proba      : probabilité de churn (0.0 à 1.0)
        - churn_label      : "Fidèle" ou "Churner"
        - cluster          : numéro de segment (0 à K-1)
        - cluster_label    : nom métier du cluster
        - monetary_pred    : valeur MonetaryTotal prédite (scalée)
    """

    feature_names = artifacts["feature_names"]

    # ── Alignement des colonnes ───────────────────────────────────────────────
    # S'assurer que X_raw contient exactement les bonnes colonnes dans le bon ordre
    missing = [c for c in feature_names if c not in X_raw.columns]
    extra   = [c for c in X_raw.columns if c not in feature_names]

    if missing:
        print(f"  ⚠ Colonnes manquantes (remplies par 0) : {missing}")
        for col in missing:
            X_raw[col] = 0

    if extra:
        X_raw = X_raw.drop(columns=extra)

    X = X_raw[feature_names].copy()

    # ── Extraction target régression ──────────────────────────────────────────
    if "MonetaryTotal" in X.columns:
        X_reg = X.drop(columns=["MonetaryTotal"])
    else:
        X_reg = X.copy()

    # ── Prédiction Classification — Churn ─────────────────────────────────────
    clf = artifacts["classifier"]
    churn_pred  = clf.predict(X)
    churn_proba = clf.predict_proba(X)[:, 1]

    # ── Prédiction Clustering ─────────────────────────────────────────────────
    pca    = artifacts["pca"]
    kmeans = artifacts["kmeans"]

    # Écrêtage ±5σ comme dans preprocessing (pour le clustering uniquement)
    X_clipped = X.clip(lower=-5, upper=5)
    X_pca     = pca.transform(X_clipped)
    clusters  = kmeans.predict(X_pca)

    # ── Prédiction Régression — MonetaryTotal ─────────────────────────────────
    reg = artifacts["regressor"]
    monetary_pred = reg.predict(X_reg)

    # ── Labels métier des clusters ────────────────────────────────────────────
    cluster_labels_map = {
        0: "Clients perdus (inactifs)",
        1: "Clients fidèles",
        2: "Clients réguliers",
        3: "Clients à risque",
    }

    results = pd.DataFrame({
        "churn_pred":    churn_pred,
        "churn_proba":   np.round(churn_proba, 4),
        "churn_label":   ["Churner" if p == 1 else "Fidèle" for p in churn_pred],
        "cluster":       clusters,
        "cluster_label": [cluster_labels_map.get(c, f"Cluster {c}") for c in clusters],
        "monetary_pred": np.round(monetary_pred, 4),
    })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────

def display_results(results, X_raw=None, idx=None):
    """Affiche les résultats de manière lisible."""

    print(f"\n{'═'*60}")
    print(f"  RÉSULTATS DE PRÉDICTION")
    print(f"{'═'*60}")

    if idx is not None:
        # Affichage détaillé pour un seul client
        row = results.iloc[0]
        print(f"\n  Client index : {idx}")
        print(f"  {'─'*40}")
        print(f"  Churn prédit    : {row['churn_label']}")
        print(f"  Probabilité     : {row['churn_proba']*100:.1f}%")
        print(f"  Segment         : {row['cluster_label']} (cluster {row['cluster']})")
        print(f"  Valeur monétaire: {row['monetary_pred']:.4f} (scalée)")

        # Niveau de risque
        proba = row["churn_proba"]
        if proba >= 0.75:
            risque = "🔴 ÉLEVÉ"
        elif proba >= 0.50:
            risque = "🟠 MOYEN"
        elif proba >= 0.25:
            risque = "🟡 FAIBLE"
        else:
            risque = "🟢 TRÈS FAIBLE"

        print(f"  Niveau de risque: {risque}")

    else:
        # Résumé statistique pour plusieurs clients
        n_total   = len(results)
        n_churn   = (results["churn_pred"] == 1).sum()
        n_fidele  = (results["churn_pred"] == 0).sum()

        print(f"\n  Clients analysés : {n_total}")
        print(f"  Churners prédits : {n_churn} ({n_churn/n_total*100:.1f}%)")
        print(f"  Fidèles prédits  : {n_fidele} ({n_fidele/n_total*100:.1f}%)")

        print(f"\n  Distribution par cluster :")
        for cl, label in sorted({
            0: "Clients perdus",
            1: "Clients fidèles",
            2: "Clients réguliers",
            3: "Clients à risque"
        }.items()):
            n = (results["cluster"] == cl).sum()
            if n > 0:
                print(f"    Cluster {cl} — {label:25s} : {n:4d} clients ({n/n_total*100:.1f}%)")

        print(f"\n  Probabilité churn moyenne : {results['churn_proba'].mean()*100:.1f}%")
        print(f"  Probabilité churn médiane : {results['churn_proba'].median()*100:.1f}%")

        # Top 5 clients les plus à risque
        top5 = results.nlargest(5, "churn_proba")[["churn_proba", "churn_label", "cluster_label"]]
        print(f"\n  Top 5 clients les plus à risque :")
        for i, (idx_r, row) in enumerate(top5.iterrows(), 1):
            print(f"    {i}. Index {idx_r:4d} | Proba {row['churn_proba']*100:.1f}% | {row['cluster_label']}")

    print(f"\n{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions(results, path="reports/predictions.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path, index=True)
    print(f"  [SAVE] Prédictions sauvegardées → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline d'inférence ML Retail")
    parser.add_argument("--client", type=int, default=None,
                        help="Index du client dans X_test (0 à 874)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Chemin vers un fichier CSV externe à prédire")
    parser.add_argument("--save", action="store_true",
                        help="Sauvegarder les prédictions dans reports/predictions.csv")
    args = parser.parse_args()

    # Chargement des artefacts
    artifacts = load_artifacts()

    # Chargement des données
    if args.csv:
        print(f"\n  [DATA] Chargement fichier externe : {args.csv}")
        X_raw = pd.read_csv(args.csv)
        idx   = None
    else:
        print(f"\n  [DATA] Chargement X_test.csv...")
        X_raw = pd.read_csv("data/train_test/X_test.csv")
        idx   = args.client

        if args.client is not None:
            if args.client < 0 or args.client >= len(X_raw):
                print(f"  ❌ Index {args.client} invalide. X_test contient {len(X_raw)} clients (0 à {len(X_raw)-1}).")
                return
            X_raw = X_raw.iloc[[args.client]]

    print(f"  ✓ {len(X_raw)} client(s) à prédire")

    # Prédiction
    results = predict(X_raw, artifacts)

    # Affichage
    display_results(results, X_raw, idx=idx if args.client is not None else None)

    # Sauvegarde optionnelle
    if args.save:
        save_predictions(results)


if __name__ == "__main__":
    main()