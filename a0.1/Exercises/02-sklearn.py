import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.datasets import fetch_california_housing
# calDB = fetch_california_housing()

# from sklearn.datasets import load_iris
# irisDB = load_iris()
# print(irisDB.items())
# print(irisDB.keys())
# print(irisDB.values())
# print(irisDB['data'][:5])
# print(irisDB['feature_names'])
# print(irisDB.target)
# print(irisDB.data.shape)
# print(irisDB.target_names)
# print(irisDB.target_names.tolist())
# print(irisDB.DESCR)
# print(irisDB.filename)
# X = irisDB.data
# y = irisDB.target
# print(X)
# print(y)
# df = pd.DataFrame(X, columns=irisDB.feature_names)
# df['target'] = irisDB.target
# print(df.shape)
# print(df.corr())

# from sklearn.datasets import make_moons
from sklearn.datasets import load_iris, fetch_california_housing, make_moons
# --- Iris (toy, tabular) ---
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
df_iris = pd.DataFrame(X_iris, columns=iris.feature_names)
df_iris["target"] = y_iris

print("Iris shape:", df_iris.shape)
print(df_iris.head())
print(df_iris.isna().sum().to_dict())

# --- California Housing (realistic, tabular) ---
cal = fetch_california_housing()
df_cal = pd.DataFrame(cal.data, columns=cal.feature_names)
df_cal["MedHouseVal"] = cal.target

print("\nCalifornia shape:", df_cal.shape)
print(df_cal.head())
print(df_cal.isna().sum().to_dict())

# --- Synthetic Moons (controlled patterns) ---
X_moon, y_moon = make_moons(n_samples=400, noise=0.15)
df_moon = pd.DataFrame(X_moon, columns=["x1","x2"])
df_moon["label"] = y_moon
print("\nMoons shape:", df_moon.shape)
print(df_moon.head())

# Quick visuals (optional)
plt.figure(figsize=(4,3))
plt.scatter(df_moon["x1"], df_moon["x2"], c=df_moon["label"])
plt.title("make_moons() scatter")
plt.tight_layout()
plt.show()
