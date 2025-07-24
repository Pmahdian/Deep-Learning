from tensorflow import keras

def build_model():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(75, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    
    model.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])
    return model

