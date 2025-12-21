# If needed:
# !pip install openml

import openml

d_irish = openml.datasets.get_dataset(451)  # "irish"
target_col = d_irish.default_target_attribute
X_irish, y_irish, cat_ind, names = d_irish.get_data(dataset_format="dataframe", target=target_col)

df_irish = X_irish.copy()
df_irish[target_col] = y_irish

print("OpenML 'irish' shape:", df_irish.shape)
print(df_irish.head())
print("Target:", target_col)
print("Categorical flags per feature:", dict(zip(names, cat_ind)))
print(df_irish.isna().sum().sort_values(ascending=False))

# Simple target distribution
df_irish[target_col].value_counts(dropna=False)
