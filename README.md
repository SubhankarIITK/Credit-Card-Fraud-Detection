# Credit Card Fraud Detection - ML

A machine learning project that develops a model to accurately detect fraudulent credit card transactions using historical data. The model analyzes transaction patterns to distinguish between normal and fraudulent activity, helping financial institutions flag suspicious behavior early and reduce potential risks.

## 📊 Project Overview

This project tackles the challenge of credit card fraud detection using machine learning techniques. The model is designed to:
- Detect fraudulent transactions with high precision
- Minimize false positives (flagging valid transactions as fraud)
- Maximize recall to catch as many fraud cases as possible
- Handle highly imbalanced datasets effectively

## 🎯 Key Challenges

- **Imbalanced Dataset**: Fraud cases represent only 0.02% of all transactions
- **High Precision Requirement**: Must minimize false positives to avoid customer inconvenience
- **High Recall Requirement**: Must detect as many fraud cases as possible to minimize financial losses
- **Feature Engineering**: Working with anonymized features (V1-V28) for privacy protection

## 📋 Dataset Description

The dataset contains **284,807 transactions** with **31 features**:

- **Time**: Seconds elapsed since the first transaction
- **V1-V28**: Anonymized features created using PCA transformation
- **Amount**: Transaction amount in euros
- **Class**: Target variable (0 = Normal, 1 = Fraudulent)

**Dataset Statistics**:
- Total transactions: 284,807
- Fraudulent transactions: 492 (0.173%)
- Normal transactions: 284,315 (99.827%)
- Outlier fraction: 0.0017

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and metrics

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/SubhankarIITK/Credit-Card-Fraud-Detection.git
cd credit-card-fraud-detection
```

2. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Download the dataset from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project directory as `creditcard.csv`.

## 🚀 Usage

Run the main script to train and evaluate the fraud detection model:

```python
python fraud_detection.py
```

The script will:
1. Load and explore the dataset
2. Analyze class distribution and transaction amounts
3. Visualize feature correlations
4. Train a Random Forest Classifier
5. Evaluate model performance with comprehensive metrics

## 📈 Model Performance

The Random Forest Classifier achieved the following results:

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.96% |
| **Precision** | 98.73% |
| **Recall** | 79.59% |
| **F1-Score** | 88.14% |
| **Matthews Correlation Coefficient** | 88.63% |

### Performance Analysis

- **High Precision (98.73%)**: Very few false alarms - when the model predicts fraud, it's correct 98.73% of the time
- **Good Recall (79.59%)**: Catches about 80% of actual fraud cases
- **Strong F1-Score (88.14%)**: Good balance between precision and recall
- **Excellent MCC (88.63%)**: Robust performance metric for imbalanced datasets

## 📊 Key Insights

1. **Transaction Amount Patterns**: Fraudulent transactions tend to have different amount distributions compared to normal transactions
2. **Feature Correlations**: Features V2 and V5 show negative correlation with transaction amounts
3. **Class Imbalance**: The dataset is highly imbalanced with only 0.173% fraudulent transactions
4. **Model Robustness**: Random Forest handles the imbalanced dataset well without additional sampling techniques

## 🔧 Project Structure

```
credit-card-fraud-detection/
│
├── README.md
├── fraud_detection.py
├── creditcard.csv
├── requirements.txt
└── outputs/
    ├── correlation_matrix.png
    ├── confusion_matrix.png
    └── evaluation_metrics.txt
```

## 🚀 Future Improvements

1. **Dataset Balancing**: Implement SMOTE or other resampling techniques
2. **Feature Engineering**: Create additional features from existing ones
3. **Advanced Models**: Experiment with XGBoost, Neural Networks, or ensemble methods
4. **Real-time Processing**: Implement streaming fraud detection
5. **Explainable AI**: Add SHAP or LIME for model interpretability

## 📝 Code Example

```python
# Basic usage example
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("creditcard.csv")

# Prepare features and target
X = data.drop(['Class'], axis=1)
y = data['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 📊 Evaluation Metrics Explained

- **Accuracy**: Overall correctness of the model
- **Precision**: Proportion of predicted fraud cases that are actually fraud
- **Recall**: Proportion of actual fraud cases that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **Matthews Correlation Coefficient**: Balanced measure for imbalanced datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by Kaggle and the Machine Learning Group of ULB
- Scikit-learn documentation and community
- Credit card fraud detection research papers and methodologies


