"""
Autorzy:
Dominik Ludwiński
Bartosz Dembowski

Opis:
    Skrypt trenuje klasyfikator drzewa decyzyjnego do prognozowania
    jakości czerwonego wina. Problem został sprowadzony do klasyfikacji
    binarnej:
        - 0 – wino gorszej jakości (quality < 6)
        - 1 – wino dobrej jakości (quality >= 6)

    Do wizualizacji wykorzystujemy tylko dwie cechy:
        - alcohol
        - volatile acidity

    Pozwala to:
        - zobaczyć rozkład danych wejściowych w 2D,
        - narysować granice decyzyjne modelu (train/test),
        - ocenić jakość klasyfikacji (classification_report),
        - sprawdzić wpływ parametru max_depth na accuracy.

Instrukcja użycia:
    Wymagane pakiety Python:
        numpy
        pandas
        scikit-learn
        matplotlib

    Instalacja zależności:
        pip install numpy pandas scikit-learn matplotlib

    Uruchomienie
        python wine_decision_tree.py

Źródło danych (Wine Quality – Red):
    UCI Machine Learning Repository:
    https://archive.ics.uci.edu/ml/datasets/wine+quality
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def visualize_classifier(classifier, X, y, title: str = "") -> None:
    """
    Rysuje granice decyzyjne klasyfikatora dla danych 2D (wine).

    Parametry
    ---------
    classifier : wytrenowany klasyfikator z metodą predict(X)
    X : ndarray, shape (n_samples, 2)
        Dane wejściowe – dwie cechy numeryczne:
            [alcohol, volatile acidity].
    y : ndarray, shape (n_samples,)
        Etykiety klas binarnych: 0 – złe, 1 – dobre wino.
    title : str
        Tytuł wykresu.

    Funkcja:
        - buduje gęstą siatkę punktów,
        - dla każdego punktu liczy przewidywaną klasę,
        - koloruje tło według klasy,
        - nakłada punkty treningowe/testowe w czytelnych kolorach.
    """
    # zakres siatki
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    # predykcje na całej siatce
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(9, 6))
    plt.title(title)

    # tło – obszary decyzyjne
    from matplotlib.colors import ListedColormap
    bg_cmap = ListedColormap(["#ffccaa", "#aaddff"])
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=bg_cmap)

    # punkty danych
    class_0 = X[y == 0]  # bad wine
    class_1 = X[y == 1]  # good wine

    plt.scatter(
        class_0[:, 0], class_0[:, 1],
        s=35, alpha=0.6,
        color="red", edgecolors="black",
        label="bad wine"
    )
    plt.scatter(
        class_1[:, 0], class_1[:, 1],
        s=35, alpha=0.6,
        color="green", edgecolors="black",
        label="good wine"
    )

    plt.xlabel("alcohol")
    plt.ylabel("volatile acidity")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ====== wczytanie danych ======
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(CURRENT_DIR, "winequality-red.csv")

df = pd.read_csv(csv_path, sep=";")

# Etykieta: 0 – wino gorsze, 1 – wino dobre
df["target"] = (df["quality"] >= 6).astype(int)

features = ["alcohol", "volatile acidity"]
X = df[features].values
y = df["target"].values

# ====== wizualizacja danych wejściowych ======
class_0 = X[y == 0]
class_1 = X[y == 1]

plt.figure(figsize=(9, 6))
plt.scatter(
    class_0[:, 0], class_0[:, 1],
    s=35, alpha=0.6,
    color="red", edgecolors="black",
    label="bad wine"
)
plt.scatter(
    class_1[:, 0], class_1[:, 1],
    s=35, alpha=0.6,
    color="green", edgecolors="black",
    label="good wine"
)
plt.xlabel("alcohol")
plt.ylabel("volatile acidity")
plt.title("Wine Quality – Input Data")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ====== podział na zbiory ======
# stratify=y -> zachowujemy proporcje klas w train i test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5, stratify=y
)

# ====== model: drzewo decyzyjne ======
classifier = DecisionTreeClassifier(
    random_state=0,
    max_depth=5
)
classifier.fit(X_train, y_train)

# ====== wizualizacja granic decyzyjnych ======
visualize_classifier(classifier, X_train, y_train,
                     "Wine – Decision Tree (Train)")
visualize_classifier(classifier, X_test, y_test,
                     "Wine – Decision Tree (Test)")

# ====== raport jakości ======
print("\n### WINE – DECISION TREE – TRAIN ###")
y_pred_train = classifier.predict(X_train)
print(classification_report(y_train, y_pred_train))

print("\n### WINE – DECISION TREE – TEST ###")
y_pred_test = classifier.predict(X_test)
print(classification_report(y_test, y_pred_test))

# ====== wpływ parametru max_depth na accuracy ======
print("\n=== Effect of max_depth (Decision Tree, Wine) ===")
for depth in [2, 4, 6, 8, None]:
    clf = DecisionTreeClassifier(random_state=0, max_depth=depth)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"max_depth={depth}: acc={acc:.3f}")

# ====== przykładowa klasyfikacja pojedynczej próbki ======
# Przykładowe "realistyczne" wartości z rozkładu danych
example_sample = np.array([[11.0, 0.4]])  # [alcohol, volatile acidity]
pred = classifier.predict(example_sample)[0]

print("\n=== Przykładowa klasyfikacja – Decision Tree (Wine) ===")
print(f"Wejście [alcohol, volatile acidity]: {example_sample[0]}")
print(f"Przewidziana klasa (0=złe, 1=dobre): {pred}")
