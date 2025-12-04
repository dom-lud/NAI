"""
Autorzy:
Dominik Ludwiński
Bartosz Dembowski

Opis:
    Skrypt trenuje klasyfikator drzewa decyzyjnego na zbiorze Ecoli.
    Do wizualizacji używane są tylko dwie cechy:
        - MCG
        - GVH
    Dzięki temu można zobaczyć w 2D:
        - rozkład danych wejściowych
        - granice decyzyjne modelu na zbiorze treningowym i testowym
    Na końcu wypisywane są metryki jakości klasyfikacji oraz prosty
    eksperyment z różnymi wartościami parametru max_depth.

Instrukcja użycia:
    python ecoli_decision_tree.py

Źródło danych (Ecoli):
    UCI / Machine Learning Repository:
    https://archive.ics.uci.edu/ml/datasets/ecoli
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def visualize_classifier(classifier, X, y, title: str = "") -> None:
    """
    Rysuje granice decyzyjne klasyfikatora dla danych 2D.

    Parametry
    ---------
    classifier : dowolny wytrenowany klasyfikator z metodą predict(X)
    X : ndarray, shape (n_samples, 2)
        Dane wejściowe – dokładnie dwie cechy numeryczne.
    y : ndarray, shape (n_samples,)
        Etykiety klas (zakodowane numerycznie).
    title : str
        Tytuł wykresu.

    Funkcja:
        - buduje gęstą siatkę punktów w przestrzeni (MCG, GVH),
        - dla każdego punktu liczy przewidywaną klasę,
        - wypełnia tło kolorami odpowiadającymi klasom,
        - nakłada na tło właściwe punkty treningowe/testowe.
    Dzięki temu widać jak classifier dzieli przestrzeń cech.
    """
    # zakres osi X/Y z niewielkim marginesem
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # regularna siatka punktów do wizualizacji granic decyzyjnych
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    # predykcja dla wszystkich punktów z siatki
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(9, 6))
    plt.title(title)

    from matplotlib.colors import ListedColormap
    # 8 kolorów tła – tyle ile klas w zbiorze Ecoli
    bg_cmap = ListedColormap([
        "#ffccaa", "#aaddff", "#ccffcc", "#ffdddd",
        "#ddeeff", "#ffffcc", "#e0ccff", "#c0c0c0"
    ])
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=bg_cmap)

    # punkty danych – kolory z tab10, lekka przezroczystość
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

# Kolumna SITE zawiera etykiety klas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["SITE"])

# Do wizualizacji bierzemy tylko dwie cechy numeryczne
features = ["MCG", "GVH"]
X = df[features].values

# ===== wizualizacja danych wejściowych =====
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    X[:, 0], X[:, 1],
    c=y, cmap="tab10",
    edgecolors="black", linewidths=0.5,
    s=35, alpha=0.8
)
plt.xlabel("MCG")
plt.ylabel("GVH")
plt.title("Ecoli – Input Data")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(*scatter.legend_elements(), title="Class")
plt.tight_layout()
plt.show()

# ===== podział na zbiory treningowy i testowy =====
# stratify=y -> proporcje klas są podobne w train i test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3, stratify=y
)

# ===== model: drzewo decyzyjne =====
classifier = DecisionTreeClassifier(max_depth=6, random_state=0)
classifier.fit(X_train, y_train)

# ===== wizualizacja granic decyzyjnych =====
visualize_classifier(
    classifier, X_train, y_train, "Ecoli – Decision Tree (Train)"
)
visualize_classifier(
    classifier, X_test, y_test, "Ecoli – Decision Tree (Test)"
)

# ===== raport jakości klasyfikacji =====
print("\n### ECOLI – DECISION TREE – TRAIN ###")
y_pred_train = classifier.predict(X_train)
print(classification_report(
    y_train,
    y_pred_train,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\n### ECOLI – DECISION TREE – TEST ###")
y_pred_test = classifier.predict(X_test)
print(classification_report(
    y_test,
    y_pred_test,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

# ===== prosty eksperyment: wpływ max_depth na accuracy =====
print("\n=== Effect of max_depth (Decision Tree, Ecoli) ===")
for depth in [3, 5, 7, 9, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"max_depth={depth}: acc={acc:.3f}")

# ===== przykładowa klasyfikacja pojedynczej próbki =====
# Wartości MCG/GVH dobrane "ze środka" rozkładu danych, tak aby
# reprezentowały typową obserwację ze zbioru.
example_sample = np.array([[0.5, 0.5]])  # [MCG, GVH]
pred_idx = classifier.predict(example_sample)[0]
pred_label = label_encoder.inverse_transform([pred_idx])[0]

print("\n=== Przykładowa klasyfikacja – Decision Tree (Ecoli) ===")
print(f"Wejście (MCG, GVH): {example_sample[0]}")
print(f"Przewidziana klasa (indeks): {pred_idx}")
print(f"Przewidziana klasa (etykieta): {pred_label}")
