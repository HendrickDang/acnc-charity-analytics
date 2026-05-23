
# Correlation Analysis of Key Numerical Features
# Rochak Bhusal

#  script purpose:
# 1. Loads the final classification dataset
# 2. Selects important numerical features
# 3. Computes the correlation matrix
# 4. Saves the correlation matrix to a CSV file
# 5. Creates and saves a correlation heatmap


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Make sure acnc_classification.csv is in the same folder
df = pd.read_csv("acnc_classification.csv")

# Step 2: Define important numerical features
# These are financial and staffing-related variables
# that are expected to have meaningful relationships.
numeric_features = [
    "Total_Revenue",
    "Total_Expenses",
    "Total_Assets",
    "Net_Assets",
    "Staff_FullTime",
    "Staff_PartTime",
    "Staff_Volunteers",
    "Total_Staff",
    "Revenue_per_Staff",
    "Rev_Exp_Ratio"
]

# Step 3: Keep only columns that exist in the dataset
# This prevents errors if some columns are missing.
available_features = [col for col in numeric_features if col in df.columns]

print("Features used in correlation analysis:")
for feature in available_features:
    print("-", feature)


# Step 4: Compute correlation matrix
# Pearson correlation is used by default.
# Values range from:
# +1.0 = strong positive correlation
#  0.0 = no correlation
# -1.0 = strong negative correlation
corr_matrix = df[available_features].corr()

# Step 5: Save correlation matrix as CSV
corr_matrix.to_csv("correlation_matrix.csv")

# Step 6: Create heatmap
plt.figure(figsize=(10, 8))

# Display correlation values as a color grid
plt.imshow(corr_matrix, interpolation="nearest")
plt.colorbar()

# Set x-axis and y-axis labels
plt.xticks(range(len(available_features)), available_features, rotation=90)
plt.yticks(range(len(available_features)), available_features)

# Add title
plt.title("Correlation Heatmap of Key Numerical Features")

# Adjust spacing
plt.tight_layout()

# Save the heatmap as an image
plt.savefig("correlation_heatmap.png", dpi=300)


# Step 7: Print correlation matrix to terminal

print("\nCorrelation Matrix:")
print(corr_matrix.round(2))

# Step 8: Print generated files
print("\nFiles created:")
print("- correlation_matrix.csv")
print("- correlation_heatmap.png")