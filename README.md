# Projet Machine Learning — Analyse Comportementale Clientèle Retail

Atelier pratique GI2 — E-commerce de cadeaux  
Pipeline complet : Exploration → Préparation → Modélisation → Évaluation → Déploiement

---

## Description

Ce projet analyse le comportement d'une clientèle e-commerce afin de :
- **Prédire le churn** (départ client) avec un classificateur Random Forest
- **Segmenter les clients** en groupes homogènes via KMeans
- **Estimer la valeur monétaire** d'un client avec un régresseur Random Forest
- **Déployer** les modèles dans une interface web Flask

---

## Structure du projet

```
projet_ml_retail/
├── preprocessing.py        # Pipeline de préparation des données
├── train_model.py          # Entraînement des 3 modèles
├── predict.py              # Script d'inférence en ligne de commande
├── debug_leakage.py        # Diagnostic de fuite de données
├── requirements.txt        # Dépendances Python
├── data/
│   ├── raw/                # Données brutes originales
│   ├── processed/          # Données nettoyées
│   └── train_test/         # Données splitées (train/test)
├── models/                 # Modèles et transformateurs sauvegardés (.pkl)
├── reports/                # Visualisations et métriques (.png, .json)
└── app/
    ├── app.py              # Application Flask
    └── templates/
        └── index.html      # Interface utilisateur
```

---

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/<votre-username>/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Utilisation

### Étape 1 — Préparation des données
```bash
python preprocessing.py
```
Génère les fichiers dans `data/train_test/` et les artefacts dans `models/`.

### Étape 2 — Entraînement des modèles
```bash
python train_model.py
```
Entraîne les 3 modèles et sauvegarde les résultats dans `reports/`.

### Étape 3 — Prédiction en ligne de commande
```bash
# Prédire sur tout X_test (875 clients)
python predict.py

# Prédire pour un client spécifique (index 42)
python predict.py --client 42

# Sauvegarder les prédictions
python predict.py --save

# Prédire sur un fichier externe
python predict.py --csv mon_fichier.csv
```

### Étape 4 — Lancer l'application web
```bash
cd app
python app.py
```
Ouvrir le navigateur à l'adresse : **http://127.0.0.1:5000**

---

## Résultats des modèles

| Modèle | Algorithme | Métriques |
|--------|-----------|-----------|
| Segmentation | KMeans K=4, ACP 10 composantes | Silhouette=0.178 |
| Classification Churn | Random Forest + GridSearchCV | ROC-AUC=1.0 |
| Régression Monétaire | Random Forest + GridSearchCV | R²=0.608, RMSE=0.846 |

### Segments clients identifiés

| Cluster | Profil | Taux de churn |
|---------|--------|---------------|
| 0 | Clients perdus (inactifs) | 100% |
| 1 | Clients fidèles | 14.7% |
| 2 | Clients réguliers | 15.4% |
| 3 | Clients à risque | 24.5% |

---

## Choix techniques

| Étape | Choix | Justification |
|-------|-------|---------------|
| Imputation | KNN k=5 | Contextualise les valeurs manquantes par similarité client |
| Rééquilibrage | SMOTE | Génère des exemples synthétiques sans copie (train uniquement) |
| Réduction dimension | ACP 10 composantes | 54.3% variance, évite la malédiction de la dimensionnalité |
| Écrêtage | ±5σ pour clustering | Neutralise l'outlier extrême (CP1=125) sans supprimer les données |
| Leakage | Suppression ChurnRiskCategory, LoyaltyLevel, CustomerType | Corrélations confirmées r>0.85 avec Churn |

---

## Dépendances principales

- `pandas`, `numpy` — manipulation des données
- `scikit-learn` — modèles ML, preprocessing, évaluation
- `imbalanced-learn` — SMOTE
- `flask` — interface web
- `joblib` — sauvegarde des modèles
- `matplotlib`, `seaborn` — visualisations

---

## Auteur

Projet réalisé dans le cadre du module Machine Learning — GI2  
Année universitaire 2025-2026