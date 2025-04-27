!pip install -q pandas matplotlib seaborn scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 1. Load the dataset
df = pd.read_csv('train.csv')  # Pastikan file 'train.csv' sudah diupload sebelumnya

# 2. Preprocessing and Data Cleaning

# a. Handling Missing Values:
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
df.drop(columns=cols_to_drop, inplace=True)
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
categorical_missing = df.select_dtypes(include='object').isnull().sum()
categorical_missing = categorical_missing[categorical_missing > 0]
for col in categorical_missing.index:
    df[col] = df[col].fillna(df[col].mode()[0])

# b. Label Encoding:
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# 3. Data Splitting:
X = df_encoded.drop("SalePrice", axis=1)
y = df_encoded["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Building and Evaluation

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate Linear Regression
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

# K-Nearest Neighbors Regression
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Evaluate KNN
knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
knn_r2 = r2_score(y_test, y_pred_knn)

# Print Model Performance
print(f"Linear Regression RMSE: {lr_rmse:.2f}")
print(f"Linear Regression R^2: {lr_r2:.2f}")
print(f"KNN Regression RMSE: {knn_rmse:.2f}")
print(f"KNN Regression R^2: {knn_r2:.2f}")

# Optional: Save the plot of `SalePrice` distribution to the output folder
import os
os.makedirs("output", exist_ok=True)
plt.figure(figsize=(8, 5))
sns.histplot(y, kde=True, color='skyblue')
plt.title("Distribusi SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Jumlah")
plt.grid(True)
output_path = "output/saleprice_distribution.png"
plt.savefig(output_path)
plt.show()
