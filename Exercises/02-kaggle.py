# If needed:
# !pip install "kagglehub[pandas-datasets]"

import kagglehub
from kagglehub import KaggleDatasetAdapter

# kagglehub.login()  # uncomment if required on your environment

handle = "kunshbhatia/delhi-air-quality-dataset"
file_in_dataset = "delhi_air_quality.csv"  # adjust if the filename differs

df_kaggle = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    handle,
    file_in_dataset,
)

print("Kaggle (Delhi Air) shape:", df_kaggle.shape)
print(df_kaggle.head())
print(df_kaggle.isna().sum().sort_values(ascending=False).head(10))
