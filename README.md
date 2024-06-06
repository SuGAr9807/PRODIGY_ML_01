# House Price Prediction with Linear Regression

## Introduction

This project aims to predict house prices using a simple linear regression model based on the square footage, number of bedrooms, and number of bathrooms.

## Dataset

The dataset used for this project is sourced from Kaggle's "House Prices: Advanced Regression Techniques" competition. It consists of two CSV files:

- `train.csv`: Training dataset containing information about houses including features and sale prices.
- `test.csv`: Testing dataset containing similar features but without the sale prices.

## Setup

Before running the code, ensure you have Python installed along with the required libraries:

- pandas
- scikit-learn
- matplotlib

You can install these libraries using pip:

```python
pip install pandas scikit-learn matplotlib
```

This section provides an overview of the project, the dataset used, and the required dependencies.

## Usage

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
```

Here, we import the necessary libraries for data manipulation, model training, evaluation, and visualization.

# Step 1: Load the dataset

```
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
```

This step loads the training and testing datasets from CSV files using pandas' read_csv() function.

# Step 2: Preprocess the data

## Select relevant features

```
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
```

## Remove rows with missing values

```
train_data = train_data.dropna(subset=features + ["SalePrice"])
test_data = test_data.dropna(subset=features)
```

Here, we preprocess the data by selecting relevant features and removing rows with missing values.

# Step 3: Split the data into training and testing sets

```
X_train = train_data[features]
y_train = train_data["SalePrice"]
X_test = test_data[features]
```

This step splits the data into features (X) and target variable (y) for both training and testing datasets.

# Step 4: Train the linear regression model

```
model = LinearRegression()
model.fit(X_train, y_train)
```

Here, we train a linear regression model using scikit-learn's LinearRegression() class.

# Step 5: Evaluate the model

```
y_train_pred = model.predict(X_train)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
print("Root Mean Squared Error (Train):", train_rmse)
```

This step evaluates the model's performance on the training set by calculating the root mean squared error (RMSE).

# Step 6: Make predictions on the test set

y_test_pred = model.predict(X_test)
Here, we use the trained model to make predictions on the test set.

## Visualize predictions

```
plt.scatter(X_test["GrLivArea"], y_test_pred, color="red", label="Predicted")
plt.scatter(X_train["GrLivArea"], y_train, color="blue", alpha=0.5, label="Actual")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.legend()
plt.title("Predicted vs Actual SalePrice")
plt.show()
```

This step visualizes the predicted vs actual sale prices for houses based on square footage using a scatter plot.

Results
The root mean squared error (RMSE) on the training set is printed to the console.
