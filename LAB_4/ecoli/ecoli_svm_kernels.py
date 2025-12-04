"""
Autorzy:
Dominik Ludwiński
Bartosz Dembowski

Opis:
    Skrypt porównuje kilka wariantów klasyfikatora SVM (SVC) na zbiorze Ecoli:
        - kernel liniowy (linear)
        - kernel RBF (z różnymi parametrami C i gamma)
        - kernel wielomianowy (poly)
        - kernel sigmoid

    Dla każdego wariantu obliczane są:
        - accuracy na zbiorze testowym
        - macro F1-score (średnia po klasach)

Instrukcja użycia:
    Wymagane pakiety Python:
        numpy
        pandas
        scikit-learn
        matplotlib

    Instalacja zależności:
        pip install numpy pandas scikit-learn matplotlib

    Uruchomienie
        python ecoli_svm_kernels.py

Źródło danych (Ecoli):
    UCI / Machine Learning Repository:
    https://archive.ics.uci.edu/ml/datasets/ecoli
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


# ===== wczytanie danych =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(CURRENT_DIR, "ecoli.csv")

df = pd.read_csv(csv_path)

# SITE = etykiety klas tekstowych (cp, im, imL, ...)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["SITE"])

# Używamy tych samych cech, co w pozostałych skryptach z Ecoli
features = ["MCG", "GVH"]
X = df[features].values

# Podział na train/test – tak samo jak w innych plikach
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3, stratify=y
)

# ===== definicja konfiguracji SVM do przetestowania =====
# Każda pozycja: (opis, obiekt SVC)
svm_configs = [
    (
        "linear, C=1.0",
        SVC(kernel="linear", C=1.0, decision_function_shape="ovr")
    ),
    (
        "RBF, C=1.0, gamma='scale'",
        SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovr")
    ),
    (
        "RBF, C=5.0, gamma='scale'",
        SVC(kernel="rbf", C=5.0, gamma="scale", decision_function_shape="ovr")
    ),
    (
        "RBF, C=1.0, gamma=0.5",
        SVC(kernel="rbf", C=1.0, gamma=0.5, decision_function_shape="ovr")
    ),
    (
        "poly, degree=3, C=1.0",
        SVC(kernel="poly", degree=3, C=1.0, gamma="scale",
            decision_function_shape="ovr")
    ),
    (
        "poly, degree=4, C=1.0",
        SVC(kernel="poly", degree=4, C=1.0, gamma="scale",
            decision_function_shape="ovr")
    ),
    (
        "sigmoid, C=1.0, gamma='scale'",
        SVC(kernel="sigmoid", C=1.0, gamma="scale",
            decision_function_shape="ovr")
    ),
]

print("=== SVM – porównanie kernel function (Ecoli, cechy: MCG, GVH) ===\n")
print(f"Liczba klas: {len(label_encoder.classes_)}")
print(f"Klasy: {list(label_encoder.classes_)}\n")

results = []

for desc, svm in svm_configs:
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append((desc, acc, f1_macro))

    print(f"Konfiguracja: {desc}")
    print(f"  accuracy (test):   {acc:.3f}")
    print(f"  macro F1 (test):   {f1_macro:.3f}")
    print("-" * 50)

# Krótkie podsumowanie w formie "tabelki" tekstowej
print("\n=== Podsumowanie – SVM kernels (Ecoli) ===")
print(f"{'Konfiguracja':40s} | {'acc':>5s} | {'F1_macro':>8s}")
print("-" * 65)
for desc, acc, f1_macro in results:
    print(f"{desc:40s} | {acc:5.3f} | {f1_macro:8.3f}")
