"""
Autorzy:
Dominik Ludwiński
Bartosz Dembowski

Opis:
    Skrypt trenuje klasyfikator SVM (SVC z kernelem liniowym) do
    klasyfikacji czerwonego wina na dwie klasy:
        - 0 – wino gorszej jakości
        - 1 – wino dobrej jakości

    Do wizualizacji używane są dwie cechy:
        - alcohol
        - volatile acidity

    Skrypt:
        - dzieli dane na train/test (z zachowaniem proporcji klas),
        - trenuje SVC(kernel="linear"),
        - rysuje granice decyzyjne SVM (train i test),
        - wypisuje metryki klasyfikacji,
        - pokazuje przykład klasyfikacji pojedynczej próbki.

Instrukcja użycia:
    python wine_svm.py

Źródło danych (Wine Quality – Red):
    UCI Machine Learning Repository:
    https://archive.ics.uci.edu/ml/datasets/wine+quality
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def visualize_classifier(classifier, X, y, title: str = "") -> None:
    """
    Rysuje granice decyzyjne klasyfikatora SVM dla danych 2D (wine).

    Parametry
    ---------
    classifier : SVC
        Wytrenowany klasyfikator SVM z kernelem liniowym.
    X : ndarray, shape (n_samples, 2)
        Dane wejściowe [alcohol, volatile acidity].
    y : ndarray, shape (n_samples,)
        Etykiety klas binarnych (0/1).
    title : str
        Tytuł wykresu.

    Funkcja jest spójna z wersją z drzewa decyzyjnego – dzięki temu
    łatwiej wizualnie porównać oba modele.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(9, 6))
    plt.title(title)

    from matplotlib.colors import ListedColormap
    bg_cmap = ListedColormap(["#ffccaa", "#aaddff"])
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=bg_cmap)

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
df["target"] = (df["quality"] >= 6).astype(int)

features = ["alcohol", "volatile acidity"]
X = df[features].values
y = df["target"].values

# ====== podział na train / test ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5, stratify=y
)

# ====== model: SVM – SVC z kernelem liniowym ======
classifier = SVC(kernel="linear", C=1.0, decision_function_shape="ovr")
classifier.fit(X_train, y_train)

# ====== wizualizacja granic decyzyjnych ======
visualize_classifier(classifier, X_train, y_train,
                     "Wine – SVC linear (Train)")
visualize_classifier(classifier, X_test, y_test,
                     "Wine – SVC linear (Test)")

# ====== raport jakości ======
print("\n### WINE – SVC (linear) – TRAIN ###")
y_pred_train = classifier.predict(X_train)
print(classification_report(y_train, y_pred_train))

print("\n### WINE – SVC (linear) – TEST ###")
y_pred_test = classifier.predict(X_test)
print(classification_report(y_test, y_pred_test))

# ====== przykładowa klasyfikacja pojedynczej próbki ======
example_sample = np.array([[11.0, 0.4]])  # [alcohol, volatile acidity]
pred = classifier.predict(example_sample)[0]

print("\n=== Przykładowa klasyfikacja – SVC linear (Wine) ===")
print(f"Wejście [alcohol, volatile acidity]: {example_sample[0]}")
print(f"Przewidziana klasa (0=złe, 1=dobre): {pred}")
