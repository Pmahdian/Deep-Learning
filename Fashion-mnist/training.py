from model import build_model
from data_loading import load_data

def train_model():
    x_train, y_train, _, _ = load_data()
    model = build_model()
    history = model.fit(x_train, y_train, epochs=50, validation_split=0.15)
    return model, history