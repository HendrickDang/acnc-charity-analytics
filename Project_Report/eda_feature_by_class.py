# EDA: Mean feature values by charity size class
# Van Hoi (Hendrick) Dang

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("acnc_classification.csv")

# Map numeric target back to size labels for readability
size_labels = {0: "Small", 1: "Medium", 2: "Large"}
df["Charity_Size_Label"] = df["target"].map(size_labels)

# Select key financial features to compare across classes
features = ["Total_Revenue", "Total_Expenses", "Total_Assets"]
feature_labels = ["Avg Revenue", "Avg Expenses", "Avg Assets"]

# Compute mean values per class
means = df.groupby("Charity_Size_Label")[features].mean()

# Format y-axis ticks smartly: show $K for thousands, $M for millions
def currency_formatter(x, _):
    if x >= 1e6:
        return '$%.0fM' % (x / 1e6)
    elif x >= 1e3:
        return '$%.0fK' % (x / 1e3)
    else:
        return '$%.0f' % x

# Plot grouped bar chart with log scale to handle large value differences
classes = ["Small", "Medium", "Large"]
x = np.arange(len(feature_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

for i, cls in enumerate(classes):
    ax.bar(x + i * width, means.loc[cls], width, label=cls)

ax.set_xticks(x + width)
ax.set_xticklabels(feature_labels)
ax.set_ylabel("Mean Value (AUD, log scale)")
ax.set_title("Mean Financial Features by Charity Size")
ax.legend(title="Charity Size")
ax.set_yscale('log')
ax.set_ylim(bottom=1000, top=2e8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))

plt.tight_layout()
plt.savefig("eda_feature_by_class.png", dpi=300)
plt.show()

print("Saved as eda_feature_by_class.png")