from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Split data
    X_train0, X_test, y_train0, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    X_train1, X_validation, y_train1, y_validation = train_test_split(
        X_train0, y_train0, test_size=0.25, random_state=42
    )
    
    # Scale data
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train1)
    X_validation_s = sc.transform(X_validation)
    X_test_s = sc.transform(X_test)
    
    return {
        'X_train': X_train_s,
        'X_val': X_validation_s,
        'X_test': X_test_s,
        'y_train': y_train1,
        'y_val': y_validation,
        'y_test': y_test
    }

if __name__ == "__main__":
    from data_loading import load_data
    data = load_data()
    processed_data = preprocess_data(data)