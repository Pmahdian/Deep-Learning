from model import build_model
from data_preprocessing import preprocess_data
from data_loading import load_data

def train_and_evaluate():
    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Build and compile model
    model = build_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    history = model.fit(
        processed_data['X_train'],
        processed_data['y_train'],
        validation_data=(processed_data['X_val'], processed_data['y_val']),
        epochs=10,
        batch_size=32
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(
        processed_data['X_test'],
        processed_data['y_test'],
        verbose=0
    )
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

if __name__ == "__main__":
    train_and_evaluate()