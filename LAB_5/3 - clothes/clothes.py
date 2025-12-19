import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

"""
Sieć neuronowa do rozpoznawania ubrań (Fashion-MNIST) w TensorFlow/Keras.

Autorzy: Dominik Ludwiński, Bartosz Dembowski

Instrukcja użycia:
- Uruchom:
  python clothes.py
- Dataset pobierze się automatycznie przy pierwszym uruchomieniu.

Wymagane biblioteki:
- Tensorflow
- matplotlib
- sklearn
- keras
"""

"""
Load Data
"""
clothes = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = clothes.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

"""
Data scaling
"""
train_images = train_images / 255.0
test_images = test_images / 255.0

"""
Model preparation
"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10)  # logits
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

"""
Fitting model
"""
model.fit(train_images, train_labels, epochs=10, verbose=1)

"""
Accuracy
"""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nAccuracy:", test_acc)

"""
Predictions
"""
y_logits = model.predict(test_images, verbose=0)
y_pred = y_logits.argmax(axis=1)

"""
Confusion matrix
"""
cm = confusion_matrix(test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Fashion-MNIST - Confusion matrix")

plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()