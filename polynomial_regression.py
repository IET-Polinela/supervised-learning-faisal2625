
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Degree 2
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(features_scaled)

# Split untuk Degree 2
X_train_poly2, X_test_poly2, y_train_poly2, y_test_poly2 = train_test_split(X_poly2, target_clean, test_size=0.2, random_state=42)

# Model untuk Degree 2
model_poly2 = LinearRegression()
model_poly2.fit(X_train_poly2, y_train_poly2)
y_pred_poly2 = model_poly2.predict(X_test_poly2)

# Evaluasi Model Degree 2
mse_poly2 = mean_squared_error(y_test_poly2, y_pred_poly2)
r2_poly2 = r2_score(y_test_poly2, y_pred_poly2)
print("Degree 2 Polynomial Regression:")
print("MSE:", mse_poly2)
print("R2 Score:", r2_poly2)

# Degree 3
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly3.fit_transform(features_scaled)

# Split untuk Degree 3
X_train_poly3, X_test_poly3, y_train_poly3, y_test_poly3 = train_test_split(X_poly3, target_clean, test_size=0.2, random_state=42)

# Model untuk Degree 3
model_poly3 = LinearRegression()
model_poly3.fit(X_train_poly3, y_train_poly3)
y_pred_poly3 = model_poly3.predict(X_test_poly3)

# Evaluasi Model Degree 3
mse_poly3 = mean_squared_error(y_test_poly3, y_pred_poly3)
r2_poly3 = r2_score(y_test_poly3, y_pred_poly3)
print("Degree 3 Polynomial Regression:")
print("MSE:", mse_poly3)
print("R2 Score:", r2_poly3)
