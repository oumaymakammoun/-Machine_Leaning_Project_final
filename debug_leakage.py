import pandas as pd
import numpy as np

X_train = pd.read_csv("data/train_test/X_train.csv")
y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()

# Corrélation de chaque feature avec Churn
corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
print("Top 20 features les plus corrélées avec Churn :")
print(corr.head(20).to_string())