import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Sieć neuronowa do klasyfikacji jakości wina (Wine Quality Dataset)

Autorzy: Dominik Ludwiński, Bartosz Dembowski


Instrukcja użycia:
- Umieść plik z danymi w tym samym folderze (np. winequality-red.csv).
- Uruchom:
  python wine_quality.py

Wymagane biblioteki:
- tensorflow
- pandas
- scikit-learn
"""

def load_data(file_path):
    """
    Wczytuje dane Wine Quality z pliku CSV (separator ';').

    :param file_path: Nazwa lub ścieżka do pliku CSV, 'winequality-red.csv'
    :return: DataFrame z kolumną 'quality' oraz cechami numerycznymi
    """
    wine = pd.read_csv(file_path, sep=";")
    print(wine.head(5))
    return wine


def build_model(input_shape):
    """
    Buduje prostą sieć neuronową typu MLP do klasyfikacji binarnej.

    Architektura:
    - warstwa wejściowa
    - Dense(32) + ReLU
    - Dense(16) + ReLU
    - Dense(1) + Sigmoid

    :param input_shape: liczba cech wejściowych
    :return: skompilowany model Keras
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    """
    Load Data
    """
    wine = load_data("winequality-red.csv")

    """
    Prepare X and y
    """
    X = wine.drop("quality", axis=1)
    y = (wine["quality"] >= 6).astype(int)  # 0 = słabe/średnie, 1 = dobre

    """
    Train-test split
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    """
    Feature scaling
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    """
    Model training
    """
    model = build_model(input_shape=x_train.shape[1])
    model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=32,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    """
    Predictions and evaluation
    """
    y_pred_probs = model.predict(x_test, verbose=0).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()