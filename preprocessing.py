"""
preprocessing.py  — v3
======================
Pipeline complet de préparation des données — Retail E-commerce

CORRECTIONS v3 (après diagnostic debug_leakage.py) :
    - Suppression ChurnRiskCategory : |r|=0.882 avec Churn → leakage confirmé
    - Suppression LoyaltyLevel      : proxy indirect Churn
    - Écrêtage outliers ±5σ sur X_test avant clustering (index 412 : CP1=125.70)
    - Country_enc : LabelEncoder neutre (Target Enc fait dans train_model.py)
"""

import pandas as pd
import numpy as np
import ipaddress
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE


def log(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


def save_artifact(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"  [SAVE] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

log("ÉTAPE 1 — Chargement des données brutes")

df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv", low_memory=False)
print(f"  Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"  Mémoire utilisée : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — SUPPRESSION DES FEATURES INUTILES / FUYANTES
# ─────────────────────────────────────────────────────────────────────────────
# JUSTIFICATION :
#   a) NewsletterSubscribed : variance nulle (100% "Yes") → aucune information
#      discriminante. Une feature constante encodée 1 partout = bruit pur.
#
#   b) CustomerID : identifiant unique par client. Le modèle mémoriserait les IDs
#      (overfitting parfait) sans capacité de généralisation.
#
#   c) ChurnRiskCategory : LEAKAGE CONFIRMÉ par diagnostic (|r|=0.882 avec Churn).
#      Cette feature est construite DANS LE SYSTÈME SOURCE à partir du comportement
#      d'achat ET de la variable Churn elle-même → donner la réponse au modèle
#      → ROC-AUC = 1.0 garanti. Doit être supprimée impérativement.
#
#   d) LoyaltyLevel : encode ancienneté × activité récente → proxy indirect de
#      Churn, redondant avec CustomerTenureDays et Recency déjà présents.

log("ÉTAPE 2 — Suppression des features inutiles et fuyantes")

colonnes_a_supprimer = [
    "NewsletterSubscribed",   # variance nulle
    "CustomerID",             # identifiant unique
    "ChurnRiskCategory",      # 🚨 LEAKAGE : r=0.882 avec Churn (diagnostic confirmé)
    "LoyaltyLevel",  
    "CustomerType",         # proxy indirect Churn
]

colonnes_presentes = [c for c in colonnes_a_supprimer if c in df.columns]
df.drop(columns=colonnes_presentes, inplace=True)
print(f"  Supprimées : {colonnes_presentes}")
print(f"  Features restantes : {df.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — CORRECTION DES VALEURS ABERRANTES
# ─────────────────────────────────────────────────────────────────────────────

log("ÉTAPE 3 — Correction des valeurs aberrantes")

avant = df["SupportTicketsCount"].isna().sum()
df["SupportTicketsCount"] = df["SupportTicketsCount"].replace([-1, 999], np.nan)
print(f"  SupportTicketsCount : {df['SupportTicketsCount'].isna().sum() - avant} NaN ajoutés (-1, 999)")

avant = df["SatisfactionScore"].isna().sum()
df["SatisfactionScore"] = df["SatisfactionScore"].replace([-1, 0, 99], np.nan)
print(f"  SatisfactionScore   : {df['SatisfactionScore'].isna().sum() - avant} NaN ajoutés (-1, 0, 99)")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — PARSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

log("ÉTAPE 4 — Parsing et Feature Engineering")

# ── 4a. RegistrationDate ─────────────────────────────────────────────────────
# 3 formats coexistants : UK "JJ/MM/AA", ISO "AAAA-MM-JJ", US "MM/JJ/AAAA"
# dayfirst=True → priorité format UK (majoritaire). errors='coerce' → NaT si échec.
df["RegistrationDate"] = pd.to_datetime(df["RegistrationDate"], dayfirst=True, errors="coerce")
df["RegYear"]    = df["RegistrationDate"].dt.year
df["RegMonth"]   = df["RegistrationDate"].dt.month
df["RegWeekday"] = df["RegistrationDate"].dt.weekday
print(f"  RegistrationDate → RegYear, RegMonth, RegWeekday | NaT: {df['RegistrationDate'].isna().sum()}")
df.drop(columns=["RegistrationDate"], inplace=True)

# ── 4b. LastLoginIP → IP_isPrivate ───────────────────────────────────────────
# IP brute = identifiant quasi-unique (4372 valeurs distinctes) → non apprenables.
# On extrait : IP privée (réseau local/VPN=1) vs publique (0) → signal comportemental.
def is_private_ip(ip_str):
    try:
        return int(ipaddress.ip_address(str(ip_str)).is_private)
    except (ValueError, AttributeError):
        return np.nan

df["IP_isPrivate"] = df["LastLoginIP"].apply(is_private_ip)
print(f"  LastLoginIP → IP_isPrivate | {df['IP_isPrivate'].mean()*100:.1f}% IPs privées")
df.drop(columns=["LastLoginIP"], inplace=True)

# ── 4c. Feature Engineering ──────────────────────────────────────────────────
# NOTE sur Recency (r=0.860 avec Churn) :
#   Recency est une feature métier LÉGITIME — un client inactif depuis 300 jours
#   churne effectivement plus souvent. Ce n'est pas un leakage : on observe un
#   comportement passé réel, pas la valeur future de Churn.
#   On la conserve et on surveille le gap train/test pour détecter une fuite
#   temporelle éventuelle.
df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)
df["TenureRatio"]    = df["Recency"] / (df["CustomerTenureDays"] + 1)
print(f"  Features créées : MonetaryPerDay, AvgBasketValue, TenureRatio")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — ENCODAGE DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────────────────────────────────────

log("ÉTAPE 5 — Encodage des variables catégorielles")

# ── 5a. Encodage ordinal ──────────────────────────────────────────────────────
# ChurnRiskCategory et LoyaltyLevel supprimées → ne figurent plus ici.
# Seules les features ordinales légitimes restent.
ordinal_maps = {
    "SpendingCategory":  ["Low", "Medium", "High", "VIP"],
    "AgeCategory":       ["Inconnu", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "BasketSizeCategory":["Petit", "Moyen", "Grand"],
    "PreferredTimeOfDay":["Matin", "Midi", "Après-midi", "Soir", "Nuit"],
}

for col, ordre in ordinal_maps.items():
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=ordre, ordered=True).codes
        df[col] = df[col].replace(-1, np.nan)
        print(f"  Ordinal  | {col:25s} → {len(ordre)} niveaux")

# ── 5b. One-Hot Encoding ──────────────────────────────────────────────────────
# Features nominales : pas d'ordre naturel entre catégories.
# drop_first=True : évite la dummy variable trap (multicolinéarité parfaite).
one_hot_cols = [
    "RFMSegment", "CustomerType", "FavoriteSeason",
    "Region", "WeekendPreference", "ProductDiversity",
    "Gender", "AccountStatus"
]
one_hot_cols = [c for c in one_hot_cols if c in df.columns]
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
print(f"\n  One-Hot  | {one_hot_cols}")
print(f"  Features après One-Hot : {df.shape[1]}")

# ── 5c. Country — LabelEncoder NEUTRE ────────────────────────────────────────
# JUSTIFICATION :
#   Le Target Encoding (pays → taux churn moyen) doit se calculer APRÈS le
#   split et UNIQUEMENT sur X_train pour éviter le leakage.
#   Ici on utilise un LabelEncoder alphabétique (purement neutre, aucune
#   référence à Churn). Le vrai Target Encoding sera appliqué dans train_model.py.
le_country = LabelEncoder()
df["Country_enc"] = le_country.fit_transform(df["Country"].astype(str))
df.drop(columns=["Country"], inplace=True)
save_artifact(le_country, "models/le_country.pkl")
print(f"  LabelEnc | Country → Country_enc ({df['Country_enc'].nunique()} pays) — NEUTRE (sans Churn)")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 — SÉPARATION X / y
# ─────────────────────────────────────────────────────────────────────────────
# Churn ne doit JAMAIS être normalisé, imputé ou transformé.

log("ÉTAPE 6 — Séparation X / y")

y = df["Churn"].copy()
X = df.drop(columns=["Churn"])
print(f"  X : {X.shape} | y : 0={( y==0).sum()} | 1={(y==1).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 7 — TRAIN / TEST SPLIT (80/20 stratifié)
# ─────────────────────────────────────────────────────────────────────────────
# Règle d'or : JAMAIS normaliser/imputer avant le split.
# stratify=y : préserve la proportion 67%/33% dans les deux ensembles.

log("ÉTAPE 7 — Train / Test Split (80/20 stratifié)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  X_train : {X_train.shape} | X_test : {X_test.shape}")
print(f"  Churn — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 8 — IMPUTATION KNN
# ─────────────────────────────────────────────────────────────────────────────
# KNNImputer trouve les k=5 clients les plus similaires et remplace les NaN
# par la moyenne de leurs valeurs → imputation contextualisée.
# fit() UNIQUEMENT sur X_train → transform() appliqué sur X_test.

log("ÉTAPE 8 — Imputation KNN (k=5) — fit sur train uniquement")

knn_imputer = KNNImputer(n_neighbors=5)
X_train_arr = knn_imputer.fit_transform(X_train)
X_test_arr  = knn_imputer.transform(X_test)

X_train = pd.DataFrame(X_train_arr, columns=X_train.columns, index=X_train.index)
X_test  = pd.DataFrame(X_test_arr,  columns=X_test.columns,  index=X_test.index)

print(f"  NaN restants — train: {X_train.isna().sum().sum()} | test: {X_test.isna().sum().sum()}")
save_artifact(knn_imputer, "models/knn_imputer.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 9 — SUPPRESSION MULTICOLINÉARITÉ (|r| > 0.85)
# ─────────────────────────────────────────────────────────────────────────────
# Corrélations > 0.85 = redondance → gonfle la dimensionnalité, déstabilise
# les modèles linéaires, ralentit l'entraînement sans gain de performance.

log("ÉTAPE 9 — Suppression multicolinéarité (|r| > 0.85)")

corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]

print(f"  Features supprimées ({len(cols_to_drop)}) :")
for col in cols_to_drop:
    for partner, val in upper_tri[col][upper_tri[col] > 0.85].items():
        print(f"    - {col:35s} ↔ {partner} (r={val:.3f})")

X_train.drop(columns=cols_to_drop, inplace=True)
X_test.drop(columns=cols_to_drop, inplace=True)
print(f"  Features restantes : {X_train.shape[1]}")
save_artifact(cols_to_drop, "models/dropped_corr_cols.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 10 — NORMALISATION (StandardScaler)
# ─────────────────────────────────────────────────────────────────────────────
# Règle absolue : fit() sur X_train → transform() sur X_test uniquement.

log("ÉTAPE 10 — Normalisation StandardScaler")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"  Recency — avant: mean={X_train['Recency'].mean():.1f} std={X_train['Recency'].std():.1f}")
print(f"  Recency — après: mean={X_train_scaled['Recency'].mean():.4f} std={X_train_scaled['Recency'].std():.4f}")

# ── Écrêtage outliers pour le clustering ──────────────────────────────────────
# JUSTIFICATION :
#   Diagnostic révèle : index 412 avec CP1=125.70 (tous autres < 15).
#   Cet outlier extrême monopolise un axe PCA entier et "attire" un cluster
#   entier → K=2 produit 874 vs 1 client (clustering dégénéré, silhouette
#   artificielle de 0.977 sans sens métier).
#   Écrêtage ±5σ : valeur conservatrice. Un client "normal" même très atypique
#   ne dépasse pas 5 écarts-types sur une feature scalée. Au-delà = erreur de
#   saisie ou donnée corrompue → on ramène à la borne ±5.
#   Appliqué UNIQUEMENT sur la copie pour le clustering (pas sur X_train/X_test
#   utilisés pour les modèles supervisés qui gèrent mieux les outliers via RF).

X_test_clipped = X_test_scaled.clip(lower=-5, upper=5)
n_clipped = int((X_test_scaled.abs() > 5).sum().sum())
print(f"\n  Écrêtage clustering (±5σ) : {n_clipped} valeur(s) écrêtée(s)")
print(f"  CP1 max avant écrêtage : {X_test_scaled.max().max():.2f} → après: {X_test_clipped.max().max():.2f}")

X_test_clipped.to_csv("data/train_test/X_test_scaled_no_smote.csv", index=False)
print(f"  ✓ X_test_scaled_no_smote.csv sauvegardé (écrêté, pour clustering uniquement)")

save_artifact(scaler, "models/scaler.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 11 — SMOTE (train uniquement)
# ─────────────────────────────────────────────────────────────────────────────
# SMOTE génère des exemples synthétiques de la classe minoritaire (churners)
# par interpolation entre voisins réels → évite le surapprentissage par copie.
# JAMAIS appliqué sur X_test : le test doit rester représentatif du réel.

log("ÉTAPE 11 — Rééquilibrage SMOTE (train uniquement)")

print(f"  Avant SMOTE : Fidèles={(y_train==0).sum()} | Churners={(y_train==1).sum()}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"  Après SMOTE : Fidèles={(y_train_res==0).sum()} | Churners={(y_train_res==1).sum()}")
print(f"  X_train_res : {X_train_res.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 12 — SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

log("ÉTAPE 12 — Sauvegarde")

os.makedirs("data/train_test", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

pd.DataFrame(X_train_res, columns=X_train_scaled.columns).to_csv(
    "data/train_test/X_train.csv", index=False)
X_test_scaled.to_csv("data/train_test/X_test.csv", index=False)
y_train_res.to_csv("data/train_test/y_train.csv", index=False)
y_test.to_csv("data/train_test/y_test.csv", index=False)

feature_names = list(X_train_scaled.columns)
save_artifact(feature_names, "models/feature_names.pkl")

print(f"\n  ✓ X_train     → data/train_test/X_train.csv  {X_train_res.shape} [SMOTE]")
print(f"  ✓ X_test      → data/train_test/X_test.csv   {X_test_scaled.shape}")
print(f"  ✓ X_test_scaled_no_smote.csv [écrêté ±5σ, clustering uniquement]")
print(f"  ✓ {X_train_res.shape[1]} features finales")
print(f"\n{'═'*60}")
print(f"  PREPROCESSING v3 TERMINÉ")
print(f"  ✓ ChurnRiskCategory supprimée (leakage r=0.882 confirmé)")
print(f"  ✓ LoyaltyLevel supprimée (proxy indirect Churn)")
print(f"  ✓ Outliers clustering écrêtés (±5σ)")
print(f"  ✓ Country LabelEncoder neutre (Target Enc → train_model.py)")
print(f"{'═'*60}")