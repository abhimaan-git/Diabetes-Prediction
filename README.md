# Diabetes Prediction with Machine Learning

This project aims to predict diabetes in individuals using machine learning techniques. The prediction is based on various health metrics and demographic information. The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

## Installation

To run this project, you need to have Python installed on your system along with the following libraries:

- NumPy
- pandas
- scikit-learn (sklearn)

You can install these libraries using pip:


pip install numpy pandas scikit-learn
```

Usage

1. Clone the repository to your local machine:


git clone https://github.com/abhimaan-git/diabetes-prediction.git
```

2. Navigate to the project directory:


cd diabetes-prediction
```

3. Run the `diabetes_prediction.py` script:

```
python diabetes_prediction.py
```

This script will perform the following tasks:

- Load and preprocess the dataset using NumPy and pandas.
- Standardize the data.
- Split the dataset into training and testing sets.
- Train a Support Vector Machine (SVM) classifier using scikit-learn.
- Evaluate the classifier's performance.
- Provide predictions for new data points.

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database, which contains various health metrics and demographic information for individuals. The dataset consists of the following columns:

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
9. Outcome (0 - No diabetes, 1 - Diabetes)

## Approach

- **Data Preprocessing**: Before training the model, the dataset is preprocessed to handle missing values and outliers.
- **Data Standardization**: The features are standardized to ensure that they have a mean of 0 and a standard deviation of 1.
- **Model Selection**: Support Vector Machine (SVM) classifier is chosen for its effectiveness in handling high-dimensional data and non-linear relationships.
- **Model Training**: The SVM classifier is trained on the preprocessed and standardized dataset.
- **Model Evaluation**: The performance of the trained model is evaluated using metrics such as accuracy, precision, recall, and F1-score.
- **Prediction**: Finally, the trained model is used to make predictions on new data points.

## Contributors

- John Doe (@john_doe)
- Jane Smith (@jane_smith)


## Acknowledgments

Special thanks to the contributors of the Pima Indians Diabetes Database on Kaggle for providing the dataset used in this project.
