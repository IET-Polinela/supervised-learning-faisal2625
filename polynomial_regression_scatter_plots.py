import seaborn as sns
import matplotlib.pyplot as plt
import os

# Debugging: Check current directory
print("Current working directory:", os.getcwd())

# Membuat folder output jika belum ada
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
print(f"Output folder exists: {os.path.exists(output_dir)}")

# Plot scatter Actual vs Predicted untuk Polynomial Degree 2 dan 3
plt.figure(figsize=(12, 4))

# Degree 2
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test_poly2, y=y_pred_poly2)
plt.title("Actual vs Predicted (Polynomial Degree 2)")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Degree 3
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test_poly3, y=y_pred_poly3)
plt.title("Actual vs Predicted (Polynomial Degree 3)")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Mengatur layout dan menampilkan plot
plt.tight_layout()

# Menyimpan gambar ke folder output
output_png_path = os.path.join(output_dir, "poly_regression_actual_vs_predicted.png")
print(f"Saving plot to {output_png_path}")
plt.savefig(output_png_path)  # Save the plot first

# Debugging: Check if file exists
if os.path.exists(output_png_path):
    print(f"Plot successfully saved to {output_png_path}")
else:
    print(f"Failed to save plot to {output_png_path}")

# Tampilkan plot secara interaktif
plt.show()

# Memberi tahu lokasi file output
print(f"Plot telah disimpan di {output_png_path}")
