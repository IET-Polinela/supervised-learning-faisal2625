
def detect_outliers_iqr(data):
    # Menyimpan indeks outlier untuk setiap kolom numerik
    outlier_indices = {}
    numeric_cols = data.select_dtypes(include=['number']).columns  # Menentukan kolom numerik
    for col in numeric_cols:
        # Menghitung Q1, Q3, dan IQR
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        # Menemukan data yang lebih kecil dari Q1 - 1.5 * IQR atau lebih besar dari Q3 + 1.5 * IQR
        outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]
        # Menyimpan indeks outlier untuk setiap kolom
        outlier_indices[col] = outliers.index
    return outlier_indices
