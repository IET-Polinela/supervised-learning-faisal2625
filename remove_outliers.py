
# Gabungkan semua index outlier dari seluruh kolom
all_outlier_indices = set()
for idx_list in outliers.values():
    all_outlier_indices.update(idx_list)

# Dataset tanpa outlier
df_no_outlier = df_encoded.drop(index=all_outlier_indices)

# Menampilkan ukuran dataset sebelum dan sesudah penghapusan outlier
print(f"Original dataset shape: {df_encoded.shape}")
print(f"Dataset tanpa outlier: {df_no_outlier.shape}")
