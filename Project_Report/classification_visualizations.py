
# Classification Visualizations
# Rochak Bhusal
#
# Script purpose:
# 1. Model accuracy comparison chart
# 2. Confusion matrix heatmap


import matplotlib.pyplot as plt
import numpy as np



# Step 1: Model Accuracy Comparison


# Accuracy results from classification models
models = ["Naive Bayes", "Decision Tree", "Random Forest"]
accuracies = [49.08, 86.85, 91.47]

# Print model accuracies
print("Model Accuracy Results:")
for model, accuracy in zip(models, accuracies):
    print(f"- {model}: {accuracy}%")

# Create bar chart
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)

# Labels and title
plt.ylabel("Accuracy (%)")
plt.xlabel("Classification Models")
plt.title("Model Accuracy Comparison")

# Save figure
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=300)



# Step 2: Confusion Matrix Heatmap

# Example confusion matrix
# Replace with actual values later if needed
cm = np.array([
    [8200, 300, 120],
    [500, 1400, 200],
    [150, 180, 1000]
])

classes = ["Small", "Medium", "Large"]

# Print confusion matrix
print("\nConfusion Matrix:")
print(cm)

# Create heatmap
plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.colorbar()

# Axis labels
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

# Add values inside matrix
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

# Labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")

# Save figure
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png", dpi=300)



# Final Output


print("\nVisualization files created successfully.")
print("- model_accuracy_comparison.png")
print("- confusion_matrix_heatmap.png")