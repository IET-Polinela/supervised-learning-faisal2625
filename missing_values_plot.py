import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset from train.csv
df = pd.read_csv('train.csv')

# Mengisi missing values pada kolom 'LotFrontage' dengan median
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# Drop columns with too many missing values
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
df.drop(columns=cols_to_drop, inplace=True)

# Mengisi missing values pada kolom kategorikal dengan modus
categorical_missing = df.select_dtypes(include='object').isnull().sum()
categorical_missing = categorical_missing[categorical_missing > 0]

for col in categorical_missing.index:
    df[col] = df[col].fillna(df[col].mode()[0])

# Hitung missing value dari semua fitur
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

# Visualisasi
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_data.index, y=missing_data.values, palette="viridis")
plt.xticks(rotation=90)
plt.title("Jumlah Missing Value per Fitur")
plt.ylabel("Jumlah Missing")
plt.xlabel("Fitur")
plt.tight_layout()

# Create output folder if not exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Menyimpan visualisasi ke file gambar dalam folder output
output_file = os.path.join(output_dir, 'missing_values_plot.png')
plt.savefig(output_file)

# Tampilkan plot
plt.show()

print(f"Plot disimpan di: {output_file}")
