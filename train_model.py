"""
train_model.py  — v3
====================
Pipeline de modélisation — Retail E-commerce (Churn & Segmentation)

CORRECTIONS v3 :
    - ChurnRiskCategory supprimée dans preprocessing → leakage éliminé
    - Clustering : données réelles écrêtées (±5σ), n_components=10
      → ni trop haut (27 → outlier domine), ni trop bas (2 → 1 cluster)
    - Target Encoding Country calculé sur X_train uniquement (après split)
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold


def log(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")

def save_artifact(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"  [SAVE] {path}")

def save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {path}")

def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

log("CHARGEMENT — Données preprocessées (v3)")

X_train = pd.read_csv("data/train_test/X_train.csv")
X_test  = pd.read_csv("data/train_test/X_test.csv")
y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()
y_test  = pd.read_csv("data/train_test/y_test.csv").squeeze()

# Données réelles écrêtées (±5σ, sans SMOTE) pour le clustering
X_cluster = pd.read_csv("data/train_test/X_test_scaled_no_smote.csv")
y_cluster  = y_test.copy()

scaler        = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

print(f"  X_train   : {X_train.shape}  (SMOTE + scaled)")
print(f"  X_test    : {X_test.shape}   (scaled)")
print(f"  X_cluster : {X_cluster.shape}  (réel, écrêté ±5σ, sans SMOTE)")
print(f"  y_train   : Churn 0={(y_train==0).sum()} | 1={(y_train==1).sum()}")
print(f"  y_test    : Churn 0={(y_test==0).sum()} | 1={(y_test==1).sum()}")

# Extraction target régression
if "MonetaryTotal" in X_train.columns:
    y_train_reg = X_train["MonetaryTotal"].copy()
    y_test_reg  = X_test["MonetaryTotal"].copy()
    X_train_reg = X_train.drop(columns=["MonetaryTotal"])
    X_test_reg  = X_test.drop(columns=["MonetaryTotal"])
    print(f"\n  MonetaryTotal extrait comme target régression")
else:
    raise ValueError("MonetaryTotal introuvable dans X_train")

os.makedirs("reports", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# TARGET ENCODING Country — après split, sur X_train uniquement
# ─────────────────────────────────────────────────────────────────────────────
# JUSTIFICATION :
#   Le Target Encoding remplace chaque code pays par le taux de churn moyen
#   des clients de ce pays. Calculé ICI sur X_train (après split) → zéro leakage.
#   En v1/v2, ce calcul sur tout le dataset encodait la réponse dans X → AUC=1.0.
#
#   Note technique : y_train contient les labels SMOTE (synthétiques pour 50%
#   des churners). Pour un Target Enc plus propre, on utiliserait les labels
#   pré-SMOTE. Ici l'approximation est acceptable pour un contexte pédagogique.

log("TARGET ENCODING Country (X_train uniquement — zéro leakage)")

global_churn_rate = float(y_train.mean())

if "Country_enc" in X_train.columns:
    temp = pd.DataFrame({"Country_enc": X_train["Country_enc"].values,
                         "Churn": y_train.values})
    country_map = temp.groupby("Country_enc")["Churn"].mean().to_dict()

    for df_ref in [X_train, X_test, X_train_reg, X_test_reg]:
        if "Country_enc" in df_ref.columns:
            df_ref["Country_target_enc"] = df_ref["Country_enc"].map(country_map).fillna(global_churn_rate)

    save_artifact(country_map, "models/country_target_enc_map.pkl")
    print(f"  ✓ Target Encoding appliqué ({len(country_map)} pays)")
    print(f"  Taux churn global (fallback inconnus) : {global_churn_rate:.3f}")
else:
    print("  ℹ  Country_enc absente des données")


# ═════════════════════════════════════════════════════════════════════════════
# MODÈLE 1 — CLUSTERING KMEANS
# ═════════════════════════════════════════════════════════════════════════════

log("MODÈLE 1 — Clustering KMeans (segmentation clients)")

# ── ACP ───────────────────────────────────────────────────────────────────────
# JUSTIFICATION du choix n_components=10 :
#   • n_components=2  → CP1 capte 24.5% variance + outlier index 412 (CP1=125.70)
#     → un seul point extrême monopolise l'axe → K=2 dégénéré (874 vs 1 client)
#   • n_components=27 (85%) → trop de dimensions → distances quasi-uniformes
#     → silhouette < 0.12 (malédiction de la dimensionnalité)
#   • n_components=10 → compromis : ~50-60% variance, espace réduit mais pas
#     dominé par un outlier, clusters plus séparés et interprétables métier.
#   L'écrêtage ±5σ (fait dans preprocessing.py) a déjà neutralisé l'outlier
#   extrême, mais on évite quand même n_components=2 qui est trop restrictif.

print("\n  [ACP] Calcul variance expliquée...")

pca_full = PCA(random_state=42)
pca_full.fit(X_cluster)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# Affichage des seuils
for seuil in [50, 70, 85, 95]:
    n = int(np.argmax(cumvar >= seuil/100)) + 1
    print(f"  Variance à {seuil}% : {n} composantes")

# Choix fixé à 10 (justifié ci-dessus)
N_COMPONENTS = 10
var_retenue = cumvar[N_COMPONENTS - 1] * 100
print(f"\n  → n_components retenu : {N_COMPONENTS} ({var_retenue:.1f}% variance)")

# Graphe PCA
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, 21), pca_full.explained_variance_ratio_[:20] * 100, color="steelblue", alpha=0.8)
axes[0].axvline(N_COMPONENTS, color="red", linestyle="--", label=f"n={N_COMPONENTS}")
axes[0].set_title("Scree Plot (20 premières CP)")
axes[0].set_xlabel("Composante")
axes[0].set_ylabel("Variance expliquée (%)")
axes[0].legend()

axes[1].plot(range(1, len(cumvar)+1), cumvar*100, color="steelblue", marker=".")
axes[1].axhline(85, color="orange", linestyle="--", label="85%")
axes[1].axvline(N_COMPONENTS, color="red", linestyle="--", label=f"n={N_COMPONENTS} → {var_retenue:.0f}%")
axes[1].set_title("Variance Cumulée")
axes[1].set_xlabel("Composantes")
axes[1].set_ylabel("Variance cumulée (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
fig.suptitle("ACP — Choix n_components (données réelles écrêtées ±5σ)", fontsize=13)
save_figure(fig, "reports/pca_variance.png")

pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_cluster_pca = pca.fit_transform(X_cluster)
print(f"  X_cluster_pca : {X_cluster_pca.shape}")
save_artifact(pca, "models/pca.pkl")

# ── Méthode du coude ─────────────────────────────────────────────────────────

print("\n  [KMeans] Recherche K optimal...")

K_range     = range(2, 11)
inertias    = []
silhouettes = []
db_scores   = []
ch_scores   = []

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_cluster_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cluster_pca, labels))
    db_scores.append(davies_bouldin_score(X_cluster_pca, labels))
    ch_scores.append(calinski_harabasz_score(X_cluster_pca, labels))
    print(f"    K={k:2d} | Inertia={km.inertia_:8.0f} | Silhouette={silhouettes[-1]:.4f} | DB={db_scores[-1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes[0,0].plot(list(K_range), inertias, marker="o", color="steelblue")
axes[0,0].set_title("Inertie (méthode du coude)"); axes[0,0].grid(True, alpha=0.3)
axes[0,1].plot(list(K_range), silhouettes, marker="o", color="green")
axes[0,1].set_title("Silhouette Score (↑ mieux)"); axes[0,1].grid(True, alpha=0.3)
axes[1,0].plot(list(K_range), db_scores, marker="o", color="red")
axes[1,0].set_title("Davies-Bouldin (↓ mieux)"); axes[1,0].grid(True, alpha=0.3)
axes[1,1].plot(list(K_range), ch_scores, marker="o", color="purple")
axes[1,1].set_title("Calinski-Harabasz (↑ mieux)"); axes[1,1].grid(True, alpha=0.3)
for ax in axes.flat:
    ax.set_xlabel("K")
fig.suptitle("Sélection K optimal — KMeans (données réelles, n_components=10)", fontsize=13)
save_figure(fig, "reports/kmeans_elbow.png")

best_k = list(K_range)[int(np.argmax(silhouettes))]
print(f"\n  → K optimal : {best_k} (Silhouette = {max(silhouettes):.4f})")

# ── KMeans final ─────────────────────────────────────────────────────────────

kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=15, max_iter=500, random_state=42)
cluster_labels = kmeans.fit_predict(X_cluster_pca)

final_sil = silhouette_score(X_cluster_pca, cluster_labels)
final_db  = davies_bouldin_score(X_cluster_pca, cluster_labels)
final_ch  = calinski_harabasz_score(X_cluster_pca, cluster_labels)

print(f"\n  Métriques KMeans final (K={best_k}) :")
print(f"    Silhouette     : {final_sil:.4f}  (> 0.25 = acceptable)")
print(f"    Davies-Bouldin : {final_db:.4f}   (< 1.0  = bon)")
print(f"    Calinski-Harabasz : {final_ch:.1f}")

unique, counts = np.unique(cluster_labels, return_counts=True)
print(f"\n  Distribution des clusters :")
for cl, cnt in zip(unique, counts):
    churn_rate = y_cluster.values[cluster_labels == cl].mean()
    print(f"    Cluster {cl} : {cnt:4d} clients ({cnt/len(cluster_labels)*100:.1f}%) — Churn: {churn_rate:.1%}")

# Visualisation 2D (CP1 vs CP2 pour affichage)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette = sns.color_palette("tab10", best_k)

for cl in range(best_k):
    mask = cluster_labels == cl
    axes[0].scatter(X_cluster_pca[mask, 0], X_cluster_pca[mask, 1],
                    s=20, alpha=0.6, color=palette[cl], label=f"Cluster {cl}")

centroids_2d = kmeans.cluster_centers_[:, :2]
axes[0].scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=200, c="black",
                marker="X", zorder=5, label="Centroïdes")
axes[0].set_title(f"KMeans K={best_k} (CP1 vs CP2)\nSilhouette={final_sil:.4f} | DB={final_db:.4f}")
axes[0].set_xlabel("CP1"); axes[0].set_ylabel("CP2")
axes[0].legend(markerscale=2); axes[0].grid(True, alpha=0.2)

churn_colors = ["#2196F3" if c == 0 else "#F44336" for c in y_cluster.values]
axes[1].scatter(X_cluster_pca[:, 0], X_cluster_pca[:, 1], c=churn_colors, s=15, alpha=0.5)
axes[1].set_title("Distribution Churn\nBleu=Fidèle | Rouge=Churner")
axes[1].set_xlabel("CP1"); axes[1].set_ylabel("CP2")
axes[1].grid(True, alpha=0.2)

fig.suptitle(f"Clustering — Données réelles écrêtées | n_components={N_COMPONENTS}", fontsize=13)
save_figure(fig, "reports/kmeans_clusters_2d.png")

save_artifact(kmeans, "models/kmeans_model.pkl")
pd.Series(cluster_labels, name="cluster").to_csv("data/train_test/test_clusters.csv", index=False)

save_metrics({
    "n_components_pca": N_COMPONENTS,
    "variance_retenue_pct": round(var_retenue, 2),
    "k": best_k,
    "silhouette": round(final_sil, 4),
    "davies_bouldin": round(final_db, 4),
    "calinski_harabasz": round(final_ch, 2),
    "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
    "cluster_churn_rates": {int(cl): round(float(y_cluster.values[cluster_labels==cl].mean()), 4)
                            for cl in unique},
}, "reports/kmeans_metrics.json")

print(f"\n  ✓ KMeans K={best_k} | Silhouette={final_sil:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# MODÈLE 2 — CLASSIFICATION RF (Churn)
# ═════════════════════════════════════════════════════════════════════════════

log("MODÈLE 2 — Classification Random Forest (Churn)")

# JUSTIFICATION :
#   Après suppression de ChurnRiskCategory (leakage r=0.882), le modèle doit
#   apprendre depuis les vraies features comportementales (Recency, Frequency,
#   Monetary, etc.). Le ROC-AUC attendu est 0.80–0.92 selon la puissance
#   prédictive résiduelle des features légitimes.

print("\n  [GridSearch] RF Classification...")

param_grid_clf = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "class_weight":      ["balanced"],
}

grid_clf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_clf,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1, verbose=1, refit=True, return_train_score=True,
)
grid_clf.fit(X_train, y_train)

print(f"\n  Meilleurs hyperparamètres :")
for k, v in grid_clf.best_params_.items():
    print(f"    {k:25s} : {v}")
print(f"  ROC-AUC CV : {grid_clf.best_score_:.4f}")

best_clf = grid_clf.best_estimator_

y_pred_clf  = best_clf.predict(X_test)
y_proba_clf = best_clf.predict_proba(X_test)[:, 1]
roc_auc     = roc_auc_score(y_test, y_proba_clf)
train_roc   = roc_auc_score(y_train, best_clf.predict_proba(X_train)[:, 1])
gap         = train_roc - roc_auc
report      = classification_report(y_test, y_pred_clf, output_dict=True)
report_str  = classification_report(y_test, y_pred_clf, target_names=["Fidèle", "Churner"])

print(f"\n{report_str}")
print(f"  ROC-AUC Test  : {roc_auc:.4f}")
print(f"  ROC-AUC Train : {train_roc:.4f}  "
      f"{'⚠ Overfitting (gap=' + f'{gap:.3f})' if gap > 0.05 else '✓ Généralisation correcte'}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

cm = confusion_matrix(y_test, y_pred_clf)
ConfusionMatrixDisplay(cm, display_labels=["Fidèle", "Churner"]).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Matrice de Confusion")

fpr, tpr, _ = roc_curve(y_test, y_proba_clf)
axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"RF (AUC={roc_auc:.4f})")
axes[1].plot([0,1], [0,1], "k--", lw=1, label="Aléatoire")
axes[1].set_title("Courbe ROC"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

feat_imp = pd.Series(best_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(20)
feat_imp.plot(kind="barh", ax=axes[2], color="steelblue", alpha=0.8)
axes[2].set_title("Top 20 Features Importantes"); axes[2].invert_yaxis()

fig.suptitle("RF Classification — Churn", fontsize=13); fig.tight_layout()
save_figure(fig, "reports/clf_evaluation.png")

save_artifact(best_clf, "models/rf_classifier.pkl")
save_artifact(grid_clf, "models/rf_classifier_gridsearch.pkl")
save_metrics({
    "best_params": grid_clf.best_params_,
    "cv_roc_auc": round(grid_clf.best_score_, 4),
    "test_roc_auc": round(roc_auc, 4),
    "train_roc_auc": round(train_roc, 4),
    "overfitting_gap": round(gap, 4),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "top10_features": feat_imp.head(10).to_dict(),
}, "reports/clf_metrics.json")

print(f"\n  ✓ RF Classifier | AUC Test={roc_auc:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# MODÈLE 3 — RÉGRESSION RF (MonetaryTotal)
# ═════════════════════════════════════════════════════════════════════════════

log("MODÈLE 3 — Régression Random Forest (MonetaryTotal)")

print(f"  X_train_reg : {X_train_reg.shape}")

param_grid_reg = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 5, 10],
    
    "max_features":      ["sqrt", "log2"],
}

grid_reg = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid_reg,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1, refit=True,
)
grid_reg.fit(X_train_reg, y_train_reg)

best_reg = grid_reg.best_estimator_
y_pred_reg = best_reg.predict(X_test_reg)

mae  = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2   = r2_score(y_test_reg, y_pred_reg)
r2_train = r2_score(y_train_reg, best_reg.predict(X_train_reg))
mask = y_test_reg != 0
mape = np.mean(np.abs((y_test_reg[mask] - y_pred_reg[mask]) / y_test_reg[mask])) * 100

print(f"\n  Meilleurs hyperparamètres : {grid_reg.best_params_}")
print(f"  MAE  : {mae:.4f} | RMSE : {rmse:.4f} | R² test : {r2:.4f} | R² train : {r2_train:.4f} | MAPE : {mape:.1f}%")
print(f"  {'⚠ Possible overfitting' if r2_train - r2 > 0.1 else '✓ Généralisation correcte'}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].scatter(y_test_reg, y_pred_reg, alpha=0.3, s=10, color="steelblue")
lims = [min(y_test_reg.min(), y_pred_reg.min()), max(y_test_reg.max(), y_pred_reg.max())]
axes[0].plot(lims, lims, "r--", lw=2); axes[0].set_title(f"Réels vs Prédits | R²={r2:.4f}")

residuals = y_test_reg.values - y_pred_reg
axes[1].hist(residuals, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--"); axes[1].set_title(f"Résidus | RMSE={rmse:.4f}")

feat_imp_reg = pd.Series(best_reg.feature_importances_, index=X_train_reg.columns).sort_values(ascending=False).head(20)
feat_imp_reg.plot(kind="barh", ax=axes[2], color="steelblue", alpha=0.8)
axes[2].set_title("Top 20 Features (Régression)"); axes[2].invert_yaxis()

fig.suptitle("RF Régression — MonetaryTotal", fontsize=13); fig.tight_layout()
save_figure(fig, "reports/reg_evaluation.png")

save_artifact(best_reg, "models/rf_regressor.pkl")
save_artifact(grid_reg, "models/rf_regressor_gridsearch.pkl")
save_metrics({
    "best_params": grid_reg.best_params_,
    "cv_rmse": round(-grid_reg.best_score_, 6),
    "test_mae": round(mae, 6), "test_rmse": round(rmse, 6),
    "test_r2": round(r2, 4), "train_r2": round(r2_train, 4),
    "mape_pct": round(mape, 2),
    "top10_features": feat_imp_reg.head(10).to_dict(),
}, "reports/reg_metrics.json")

print(f"  ✓ RF Regressor | R²={r2:.4f} | RMSE={rmse:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# RÉCAPITULATIF
# ═════════════════════════════════════════════════════════════════════════════

log("RÉCAPITULATIF — Tous les modèles entraînés")

print(f"""
  ┌────────────────────────────────────────────────────┐
  │  MODÈLE 1 — KMeans Clustering                     │
  │    ACP            : {N_COMPONENTS} composantes ({var_retenue:.0f}% variance)       │
  │    K optimal      : {best_k}                                │
  │    Silhouette     : {final_sil:.4f}                           │
  │    Davies-Bouldin : {final_db:.4f}                           │
  ├────────────────────────────────────────────────────┤
  │  MODÈLE 2 — RF Classification (Churn)             │
  │    ROC-AUC CV     : {grid_clf.best_score_:.4f}  (sans ChurnRiskCategory)  │
  │    ROC-AUC Test   : {roc_auc:.4f}                           │
  │    Gap overfitting: {gap:.4f}                           │
  ├────────────────────────────────────────────────────┤
  │  MODÈLE 3 — RF Régression (MonetaryTotal)         │
  │    R² Test        : {r2:.4f}                           │
  │    RMSE (scalé)   : {rmse:.4f}                           │
  │    MAPE           : {mape:.1f}%                           │
  └────────────────────────────────────────────────────┘
""")

print(f"{'═'*60}")
print(f"  TRAIN_MODEL v3 TERMINÉ")
print(f"{'═'*60}")