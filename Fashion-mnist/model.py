import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import ssl
import certifi


ssl_context = ssl.create_default_context(cafile=certifi.where())

fmnist_data = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fmnist_data.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(75, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))



model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(75, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


weights, bias = model.layers[1].get_weights()


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=50, validation_split=0.15)

model.evaluate(x_test, y_test, verbose=0)


model.evaluate(x_test, y_test, verbose=0)


