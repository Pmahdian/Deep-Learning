# Deep Learning Projects Repository 🧠

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-API-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green)

A collection of deep learning implementations covering fundamental concepts and practical applications.

## 📂 Repository Structure

### 1. California Housing Price Prediction
**Path**: `/California-Housing`  
**Description**:  
Predicts median house values in California using neural networks with:
- Data exploration and visualization
- Feature engineering
- Sequential model architecture
- Performance evaluation

**Key Files**:
- `data_processing.py` - Data loading and preprocessing
- `model.py` - Neural network implementation
- `training.py` - Model training pipeline
- `visualization.py` - EDA and results plotting

### 2. Fashion MNIST Classification
**Path**: `/Fashion-MNIST`  
**Description**:  
Image classifier for Fashion-MNIST dataset featuring:
- CNN implementation
- Training/validation workflows
- Model evaluation metrics
- Sample prediction visualization

**Key Files**:
- `data_loader.py` - Image data handling
- `cnn_model.py` - Convolutional network architecture
- `train.py` - Training procedures
- `evaluate.py` - Accuracy/loss metrics

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running Projects
```bash
# California Housing
cd California-Housing
python main.py

# Fashion MNIST
cd Fashion-MNIST
python train.py
```

## 🛠️ Technical Stack
- **Frameworks**: TensorFlow 2.x, Keras
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: scikit-learn

## 📊 Key Metrics
| Project | Test Accuracy | Loss |
|---------|--------------|------|
| California Housing | MAE: 0.45 | MSE: 0.38 |
| Fashion MNIST | 89.2% | 0.32 |

## 🤝 Contribution
Contributions welcome! Please:
1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## 📜 License
MIT License - See [LICENSE](LICENSE) for details


