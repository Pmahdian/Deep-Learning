from tensorflow import keras

def build_model():
    model = keras.models.Sequential([
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1)
    ])
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()