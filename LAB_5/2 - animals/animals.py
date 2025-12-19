import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
Sieć neuronowa do rozpoznawania obrazów z CIFAR-10 (10 klas) w TensorFlow/Keras.

Autorzy: Dominik Ludwiński, Bartosz Dembowski

Instrukcja użycia:
- Uruchom:
  python animals.py
- Dataset CIFAR-10 pobiera się automatycznie przy pierwszym uruchomieniu.

Wymagane biblioteki:
- Tensorflow
- matplotlib
- sklearn
- keras
- seaborn
"""

"""
Load data
"""
cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


"""
Data normalization
"""
train_images = train_images / 255.0
test_images = test_images / 255.0


"""
Model definition
"""
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10)
])


"""
Model compilation
"""
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


"""
Training
"""
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    verbose=1
)


"""
Accuracy
"""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Accuracy:", test_acc)


"""
Predictions
"""
y_pred = model.predict(test_images, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_labels.flatten()


"""
Confusion matrix
"""
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
plt.figure(figsize=(12, 9))
ax = sns.heatmap(confusion_mtx, annot=True, fmt="g")
ax.set(xticklabels=classes, yticklabels=classes)
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa klasa")
plt.title("CIFAR-10 - Confusion matrix")

plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()