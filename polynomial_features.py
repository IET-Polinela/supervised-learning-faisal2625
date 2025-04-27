
from sklearn.preprocessing import PolynomialFeatures

# Degree 2
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(features_scaled)

# Degree 3
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly3.fit_transform(features_scaled)

# Menyimpan hasil transformasi ke dalam variabel jika diperlukan
print("Polynomial features for degree 2:")
print(X_poly2)

print("Polynomial features for degree 3:")
print(X_poly3)
