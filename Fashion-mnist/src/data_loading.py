import ssl
import certifi
from tensorflow import keras

def load_data():
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    fmnist_data = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fmnist_data.load_data()
    return x_train, y_train, x_test, y_test