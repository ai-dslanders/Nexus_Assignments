# If needed:
# !pip install datasets

import matplotlib.pyplot as plt
from datasets import load_dataset, load_dataset_builder
from collections import Counter

ds = load_dataset("imdb")

# Basic structure
print(ds)
print("Train rows:", ds["train"].num_rows, "Test rows:", ds["test"].num_rows)
print("Features:", load_dataset_builder("imdb").info.features)

# Label balance (train)
label_counts = Counter(ds["train"]["label"])
print("Label counts:", label_counts)

# Bar chart
plt.bar(["Negative","Positive"], [label_counts[0], label_counts[1]])
plt.title("IMDB Sentiment Distribution (Train)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Sample row
ds["train"][0]

