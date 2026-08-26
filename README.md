# DATA-PREPROCESSING
# AIM
To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset, and applying feature scaling. 

## THEORY
Data preprocessing is an important step in data analytics and machine learning. It involves converting raw data into a clean, organized, and suitable format before applying machine learning algorithms. Real-world datasets may contain missing values, categorical data, and numerical features with different ranges. These issues can affect the performance and accuracy of machine learning models.

In this experiment, Python and the Scikit-learn library are used to perform the major steps of data preprocessing. Scikit-learn provides several useful functions and classes for preparing datasets efficiently.

The first step is handling missing values. Missing values occur when some entries in a dataset are unavailable or not recorded. They can be handled by removing incomplete records or replacing the missing values with suitable values such as the mean, median, or most frequent value.

The next step is encoding categorical data. Machine learning algorithms generally require numerical input. Therefore, categorical values such as "Male", "Female", "Yes", or "No" are converted into numerical representations using techniques such as Label Encoding or One-Hot Encoding.

After encoding, the dataset is divided into training and testing datasets. The training dataset is used to train the machine learning model, while the testing dataset is used to evaluate its performance on unseen data.

Finally, feature scaling is applied to numerical features when necessary. Features may have different ranges, such as age ranging from 18–60 and income ranging from thousands to lakhs. Scaling brings the features to a comparable range and can improve the performance of many machine learning algorithms.

Thus, data preprocessing helps transform raw data into a suitable form for developing accurate and reliable machine learning models.

## WORKING PRINCIPLE
The given dataset is first loaded into Python using suitable Scikit-learn and Python functions. The dataset is examined to identify missing values and different types of features.

Missing values are then handled using appropriate preprocessing techniques. Depending on the dataset, missing numerical values can be replaced using statistical values such as the mean or median, while categorical missing values can be replaced using the most frequent value.

Next, categorical features are converted into numerical values using encoding techniques. This makes the categorical information understandable to machine learning algorithms.

The preprocessed dataset is then divided into training and testing sets. The training data is used for developing the machine learning model, while the testing data is kept separately for evaluating the model.

Finally, numerical features are scaled so that features with different ranges can be brought to a comparable scale. After these preprocessing steps, the dataset becomes suitable for applying machine learning algorithms.


<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/0fcbb8c2-160b-4fc4-8a1e-5a466b3a4b79" />


# Procedure
1.	Import the required Python libraries. 
2.	Mount Google Drive and load the dataset using Pandas. 
3.	Display the first few records of the dataset. 
4.	Inspect the dataset using df.info() and df.shape. 
5.	Separate the independent variables (X) and dependent variable (Y). 
6.	Convert the independent variables into an array. 
7.	Identify and handle missing values using SimpleImputer with the mean strategy. 
8.	Encode the categorical Country column using LabelEncoder. 
9.	Apply One-Hot Encoding to convert categorical country values into dummy variables. 
10.	Encode the dependent variable Purchased using LabelEncoder. 
11.	Split the dataset into training and testing sets using train_test_split. 
12.	Apply Standard Scaler for feature scaling. 
13.	Display the preprocessed training and testing datasets. 

# Program


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Create a sample dataset
data = {
    'Age': [25, 30, None, 35, 28],
    'Salary': [30000, 45000, 50000, None, 40000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Separate input and output
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# Identify columns
numerical_features = ['Age', 'Salary']
categorical_features = ['Gender']

# Preprocessing numerical data
numeric_transformer = SimpleImputer(strategy='mean')

# Preprocessing categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Apply preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Data:")
print(X_train_scaled)

print("\nTesting Data:")
print(X_test_scaled)

print("\nPreprocessing completed successfully.")

# RESULT
Thus, data preprocessing using Python and Scikit-learn was performed successfully. Missing values were handled, categorical data was encoded, the dataset was divided into training and testing sets, and feature scaling was applied to prepare the dataset for further machine learning applications.
