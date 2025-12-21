# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "blastchar/telco-customer-churn",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

pd.set_option("display.max_columns", None)
# print("First 5 records:", df.head())
# print(df.shape)
# print(df.columns)
# print(df.tail)
# print(df.sample(10))
# print(df.info())
# print(df.describe())
# print(df.describe(include="object"))
for col in df.columns:
    print(f"{col}: {df[col].uniqe()}")
print(df.nunique())

# print(df.isnull())
# print(df.isnull().sum())
# print(df.isnull().sum().sum())
# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# print(df.info())
# print(df.isnull().sum().sum())
# df['TotalCharges'].fillna(['TotalCarges'].median, inplace = True)
# df['TotalCharges'] = df['TotalCharges'].fillna(['TotalCarges'].median())
