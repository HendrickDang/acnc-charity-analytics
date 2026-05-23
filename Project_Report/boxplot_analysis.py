
# Boxplot Visualization
# Rochak Bhusal

# boxplot showing the
# distribution of Total Revenue across
# Small, Medium, and Large charities.


import pandas as pd
import matplotlib.pyplot as plt


# Step 1: Load dataset

df = pd.read_csv("acnc_classification.csv")


# Step 2: Create boxplot

plt.figure(figsize=(8, 6))

# Create boxplot grouped by charity size
df.boxplot(column="Total_Revenue", by="Charity_Size")

# Titles and labels
plt.title("Total Revenue Distribution by Charity Size")
plt.suptitle("")  # removes default extra title
plt.xlabel("Charity Size")
plt.ylabel("Total Revenue")

# Save figure
plt.tight_layout()
plt.savefig("revenue_boxplot.png", dpi=300)

print("File created:")
print("- revenue_boxplot.png")