"""
Autorzy:
Dominik Ludwiński
Bartosz Dembowski

Opis:
    Skrypt trenuje klasyfikator SVM (SVC z kernelem liniowym)
    na zbiorze Ecoli. Wykorzystujemy dwie cechy:
        - MCG
        - GVH
    Dzięki temu możliwe jest:
        - zwizualizowanie danych wejściowych (MCG vs GVH),
        - narysowanie granic decyzyjnych SVM na zbiorze train/test,
        - ocenienie jakości klasyfikacji na podstawie metryk.

Instrukcja użycia:
    Wymagane pakiety Python:
        numpy
        pandas
        scikit-learn
        matplotlib

    Instalacja zależności:
        pip install numpy pandas scikit-learn matplotlib

    Uruchomienie
        python ecoli_svm.py

Źródło danych (Ecoli):
    UCI / Machine Learning Repository:
    https://archive.ics.uci.edu/ml/datasets/ecoli
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def visualize_classifier(classifier, X, y, title: str = "") -> None:
    """
    Rysuje granice decyzyjne klasyfikatora SVM dla danych 2D.

    Parametry
    ---------
    classifier : SVC
        Wytrenowany klasyfikator SVM (kernel linear).
    X : ndarray, shape (n_samples, 2)
        Dane wejściowe – dwie cechy numeryczne (MCG, GVH).
    y : ndarray, shape (n_samples,)
        Etykiety klas (zakodowane numerycznie).
    title : str
        Tytuł wykresu.

    Funkcja działa tak samo jak w drzewie:
        - tworzy siatkę punktów,
        - oblicza przewidywaną klasę,
        - rysuje tło i punkty danych.
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(9, 6))
    plt.title(title)

    from matplotlib.colors import ListedColormap
    # tło – 8 kolorów jak liczba klas
    bg_cmap = ListedColormap([
        "#ffccaa", "#aaddff", "#ccffcc", "#ffdddd",
        "#ddeeff", "#ffffcc", "#e0ccff", "#c0c0c0"
    ])
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=bg_cmap)

    # punkty – spójne z decision_tree
    scatter = plt.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap="tab10",
        edgecolors="black", linewidths=0.5,
        s=35, alpha=0.8
    )

    plt.xlabel("MCG")
    plt.ylabel("GVH")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.tight_layout()
    plt.show()


# ===== wczytanie danych z pliku CSV =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(CURRENT_DIR, "ecoli.csv")

df = pd.read_csv(csv_path)

# SITE = etykiety klas tekstowych → kodujemy je numerycznie
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["SITE"])

features = ["MCG", "GVH"]
X = df[features].values

# ===== opcjonalna wizualizacja danych wejściowych =====
# Ten wykres jest podobny do tego z drzewa, ale zostawiamy go
# żeby w skrypcie SVM również było widać rozkład klas.
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    X[:, 0], X[:, 1],
    c=y, cmap="tab10",
    edgecolors="black", linewidths=0.5,
    s=35, alpha=0.8
)
plt.xlabel("MCG")
plt.ylabel("GVH")
plt.title("Ecoli – Input Data (SVM)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(*scatter.legend_elements(), title="Class")
plt.tight_layout()
plt.show()

# ===== podział na train / test =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3, stratify=y
)

# ===== model: SVM – SVC z kernelem liniowym =====
# C = 1.0 to standardowa wartość, decision_function_shape="ovr"
# pozwala traktować problem wieloklasowo.
classifier = SVC(kernel="linear", C=1.0, decision_function_shape="ovr")
classifier.fit(X_train, y_train)

# ===== wizualizacja granic decyzyjnych =====
visualize_classifier(
    classifier, X_train, y_train, "Ecoli – SVC linear (Train)"
)
visualize_classifier(
    classifier, X_test, y_test, "Ecoli – SVC linear (Test)"
)

# ===== raport jakości =====
print("\n### ECOLI – SVC (linear) – TRAIN ###")
y_pred_train = classifier.predict(X_train)
print(classification_report(
    y_train,
    y_pred_train,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\n### ECOLI – SVC (linear) – TEST ###")
y_pred_test = classifier.predict(X_test)
print(classification_report(
    y_test,
    y_pred_test,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

# ===== przykładowa klasyfikacja pojedynczej próbki =====
example_sample = np.array([[0.5, 0.5]])  # [MCG, GVH]
pred_idx = classifier.predict(example_sample)[0]
pred_label = label_encoder.inverse_transform([pred_idx])[0]

print("\n=== Przykładowa klasyfikacja – SVC linear (Ecoli) ===")
print(f"Wejście (MCG, GVH): {example_sample[0]}")
print(f"Przewidziana klasa (indeks): {pred_idx}")
print(f"Przewidziana klasa (etykieta): {pred_label}")
