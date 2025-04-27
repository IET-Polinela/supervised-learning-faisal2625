
from sklearn.metrics import mean_squared_error, r2_score
import os

# Degree 2
mse_poly2 = mean_squared_error(y_test_poly2, y_pred_poly2)
r2_poly2 = r2_score(y_test_poly2, y_pred_poly2)

# Degree 3
mse_poly3 = mean_squared_error(y_test_poly3, y_pred_poly3)
r2_poly3 = r2_score(y_test_poly3, y_pred_poly3)

# Cetak hasil ke layar
print("🔹 Polynomial Regression Degree 2")
print("MSE:", round(mse_poly2, 2))
print("R2 Score:", round(r2_poly2, 4))

print("\n🔹 Polynomial Regression Degree 3")
print("MSE:", round(mse_poly3, 2))
print("R2 Score:", round(r2_poly3, 4))

# Buat folder output jika belum ada
os.makedirs("output", exist_ok=True)

# Simpan hasil evaluasi ke dalam file teks di folder output
output_file = "output/poly_regression_results.txt"
with open(output_file, "w") as f:
    f.write("🔹 Polynomial Regression Degree 2\n")
    f.write(f"MSE: {round(mse_poly2, 2)}\n")
    f.write(f"R2 Score: {round(r2_poly2, 4)}\n")
    f.write("\n🔹 Polynomial Regression Degree 3\n")
    f.write(f"MSE: {round(mse_poly3, 2)}\n")
    f.write(f"R2 Score: {round(r2_poly3, 4)}\n")

print(f"\nHasil telah disimpan di {output_file}")
