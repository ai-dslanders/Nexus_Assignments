#!/usr/bin/env python
# coding: utf-8

# # 🍷 Mini-Project: Merge & Explore the Wine Quality Datasets
# 
# 
# > **Goal: Merge Wine Quality – Red and Wine Quality – White (UCI) into a single dataset, do a careful first-look exploration, and save the merged file to CSV.**
# 
# <p align="center">📢⚠️📂  </p>
# 
# <p align="center"> Please name your file using the format: <code>assignmentName_nickname.py/.ipynb</code> (e.g., <code>project1_ali.py</code>) and push it to GitHub with a clear commit message.</p>
# 
# <p align="center"> 🚨📝🧠 </p>

# In[2]:


# === Requirements ===
# pip install pandas matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# In[3]:


# ---------- 1) Load ----------
URL_RED   = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
URL_WHITE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red   = pd.read_csv(URL_RED, sep=";")
white = pd.read_csv(URL_WHITE, sep=";")


# In[4]:


# ---------- 2) Sanity checks ----------
print("Red shape:", red.shape, "White shape:", white.shape)
print("Columns equal? ->", list(red.columns) == list(white.columns))
print("Columns:", list(red.columns))


# In[5]:


# (Optional) strict schema assertion (search and read about assert in Python)
assert list(red.columns) == list(white.columns), "Column mismatch between red and white datasets."


# In[13]:


# ---------- 3) Tag source & merge ----------
red["type"] = "red"
white["type"] = "white"

df = pd.concat([red, white], ignore_index=True)
print("\nMerged shape:", df.shape)


# In[ ]:


# ---------- 4) Basic exploration ----------
print("\nDtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum().sort_values(ascending=False))
print("\nHead:\n", df.head())


# In[15]:


# Uniqueness & duplicates
dup_count = df.duplicated().unique()
print("\nDuplicate rows:", dup_count)

# Descriptive statistics (numeric)
num_cols = df.select_dtypes(include=[np.number]).columns
print("\nNumeric summary:\n", df[num_cols].describe().T)

# Target distributions
print("\nQuality distribution (overall):\n", df["quality"].count())
print("\nQuality distribution by type:\n", df.groupby("type")["quality"].count().sort_index())


# In[21]:


# ---------- 5) A few simple visuals (optional for report) ----------
# Histograms of numeric features (quick feel for ranges & skew)
top_vars = df.columns
n = min(4, len(top_vars))


fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharey=True)

for i, ax in enumerate(axes):
    if i < n:
        col = top_vars[i]
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        if i == 0:
            ax.set_ylabel("Count")
        else:
            ax.set_ylabel("")
    else:
        ax.axis("off")  # hide unused panels if top_vars has < 4

plt.tight_layout()
plt.show()


# In[22]:


# Boxplot of quality by type (class distribution spread)
plt.figure()
df.boxplot(column="quality", by="type")
plt.suptitle("")
plt.title("Quality by Wine Type")
plt.xlabel("Type")
plt.ylabel("Quality")
plt.tight_layout()
plt.show()


# In[23]:


# Correlation heatmap (numeric only)
corr = df[num_cols].corr()
plt.figure(figsize=(7, 6))
plt.imshow(corr, interpolation="nearest")
plt.title("Correlation Heatmap")
plt.colorbar()
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.tight_layout()
plt.show()


# In[24]:


# ---------- 6) Save ----------
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_file = OUT_DIR / "wine_quality_merged.csv"
df.to_csv(out_file, index=False)
print(f"\nSaved merged file to: {out_file.resolve()}")

# Quick verification of saved file
df_check = pd.read_csv(out_file)
print("Reloaded shape:", df_check.shape)

