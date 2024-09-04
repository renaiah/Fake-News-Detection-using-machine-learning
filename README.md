
# Fake News Detection

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [License](#license)



## Introduction
The Fake News Detection project aims to identify and classify news articles as either "fake" or "real" using machine learning techniques. By analyzing the text content of news articles, the model helps in distinguishing between false information and authentic news. This project addresses the challenge of detecting fake news from a dataset containing both fake and true news articles.

## Features
News classification
- Classifies news articles as either fake or real.
Data Handling
- Processes and cleans text data for model training.
Model Performance
- Achieves high accuracy, precision, recall, and F1-score on the test data using various classification models.
Visualization
- Includes confusion matrix and classification reports for evaluating model performance.
## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/renaiah/credit-card-fraud-detection.git

    ```
2. Navigate to the project directory:
    ```bash
    cd credit-card-fraud-detection
    ```
3. Run the script:
    ```bash
    python credit_card_fraud_detection.py
    ```
4. Install the required libraries:
    ```bash
    pip install numpy pandas scikit-learn
    ```

## Usage
To use the Credit Card Fraud Detection model:

1. Ensure the dataset (Fake.csv and True.csv) is in the project directory.
2. Run the Streamlit app:
    ```bash
    python fake_news_detection.py
    ```
3. The script will preprocess the data, train the model, and display the evaluation metrics.

## Model Training
The model is trained using TensorFlow and Keras. Key details about the training process:

- **Model:** Logistic Regression, Decision Tree Classifier, Gradient Boosting Classifier, Random Forest Classifier
- **Dataset:** News articles labeled as fake or real, with features including the title, text, and date.
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn
- **Validation Accuracy:** ~99%

### Data Preparation
The dataset is prepared by combining fake and real news articles, cleaning the text data, and splitting it into training and testing sets. The data is then vectorized using TF-IDF to transform text data into numerical features.


### Training Process
Various classification models are trained on the preprocessed dataset using Scikit-learn. The training process includes:
- Splitting: The data is divided into training and test sets.
- Training: Models are trained using the training data.
- Evaluation: Models' performance is assessed with accuracy, precision, recall, and F1-score on the test data.

### Visualization
The script includes visualizations such as confusion matrices and classification reports to provide insights into the models' performance.

## Future Scope
Future improvements for this project could include:

- Model Expansion: Exploring additional models like Neural Networks for potentially improved accuracy.
- Data Augmentation: Implementing techniques like SMOTE for better handling of imbalanced datasets.
- Real-Time Implementation: Developing a real-time fake news detection system that can be integrated into news platforms.
- Feature Engineering: Creating new features from existing text data to enhance model performance.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Suggestions for improving the models or the codebase are highly appreciated.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
