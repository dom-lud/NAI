import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Sieć neuronowa do klasyfikacji Iris (3 klasy) w TensorFlow/Keras.

Autorzy: Dominik Ludwiński, Bartosz Dembowski

Opis zbioru danych:
Iris dataset zawiera cztery cechy opisujące kwiaty:
- sepal length
- sepal width
- petal length
- petal width

Klasy (etykiety):
0 - Iris-setosa
1 - Iris-versicolor
2 - Iris-virginica

Instrukcja użycia:
- Umieść plik iris.csv w tym samym folderze.
- Uruchom:
  python iris.py

Wymagane biblioteki:
- TensorFlow
- Pandas
- Sklearn
"""

def load_data(file_path):
    """Wczytanie danych Iris z pliku CSV."""
    iris = pd.read_csv(file_path, delimiter=",")
    print(iris.head(5))
    return iris

def build_model(input_shape, num_classes):
    """Prosty MLP do klasyfikacji danych tabelarycznych."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    """
    Load Data
    """
    iris = load_data("iris.csv")

    """
    Prepare X, y
    """
    X = iris.drop("class", axis=1)
    y = iris["class"]

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
    Model preparation + training
    """
    model = build_model(input_shape=x_train.shape[1], num_classes=3)
    model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=20,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    """
    Predictions + metrics
    """
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = y_pred_probs.argmax(axis=1)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()