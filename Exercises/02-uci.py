# If needed:
# !pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

heart = fetch_ucirepo(id=45)  # Heart Disease
X_uci = heart.data.features
y_uci = heart.data.targets

df_uci = X_uci.copy()
for c in y_uci.columns:
    df_uci[c] = y_uci[c]

print("UCI Heart shape:", df_uci.shape)
print(df_uci.head())
print("Targets:", list(y_uci.columns))
print(df_uci.isna().sum().sort_values(ascending=False).head(10))
