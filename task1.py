import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load the dataset
train_data = pd.read_csv("D:\\Projects\\Krutva_Patel\\PRODIGY_ML_01\\train.csv")
test_data = pd.read_csv("D:\\Projects\\Krutva_Patel\\PRODIGY_ML_01\\test.csv")

# Step 2: Preprocess the data
# Select relevant features
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]

# Remove rows with missing values
train_data = train_data.dropna(subset=features + ["SalePrice"])
test_data = test_data.dropna(subset=features)

# Step 3: Split the data into training and testing sets
X_train = train_data[features]
y_train = train_data["SalePrice"]
X_test = test_data[features]

# Step 4: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_train_pred = model.predict(X_train)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
print("Root Mean Squared Error (Train):", train_rmse)

# Step 6: Make predictions on the test set
y_test_pred = model.predict(X_test)

# Visualize predictions
plt.scatter(X_test["GrLivArea"], y_test_pred, color="red", label="Predicted")
plt.scatter(X_train["GrLivArea"], y_train, color="blue", alpha=0.5, label="Actual")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.legend()
plt.title("Predicted vs Actual SalePrice")
plt.show()
