# Rochak Bhusal

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("acnc_classification.csv")

# Determine which column contains the class labels
if "Charity_Size" in df.columns:
    class_col = "Charity_Size"
elif "target" in df.columns:
    mapping = {0: "Small", 1: "Medium", 2: "Large"}
    df["Charity_Size"] = df["target"].map(mapping)
    class_col = "Charity_Size"
else:
    raise ValueError("Neither 'Charity_Size' nor 'target' column found in the dataset.")

# Count class frequencies
class_counts = df[class_col].value_counts().sort_index()
class_percentages = (class_counts / class_counts.sum() * 100).round(2)

# Print results
print("Class Distribution:")
for label in class_counts.index:
    print(f"{label}: {class_counts[label]} ({class_percentages[label]}%)")

# Bar chart
plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar")
plt.title("Class Distribution of Charity Size")
plt.xlabel("Charity Size")
plt.ylabel("Number of Charities")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution_bar.png", dpi=300)

# Pie chart
plt.figure(figsize=(6, 6)) 
class_counts.plot(kind="pie", autopct="%1.1f%%", ylabel="")
plt.title("Class Distribution of Charity Size")
plt.tight_layout()
plt.savefig("class_distribution_pie.png", dpi=300)

print("\nFiles created:")
print("- class_distribution_bar.png")
print("- class_distribution_pie.png")