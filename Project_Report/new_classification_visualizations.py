import matplotlib.pyplot as plt
import numpy as np

# Step 1: Model Accuracy Comparison

# Accuracy results from classification models
models = ["Naive Bayes", "Decision Tree", "SVM", "Random Forest"]
accuracies = [49.08, 86.85, 88.60, 91.47]

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
plt.show()

# Step 2: Confusion Matrix Heatmap

# Random Forest confusion matrix from test set
cm = np.array([
    [8183, 275, 42],
    [412,  1322, 60],
    [117,  81,  1078]
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
        plt.text(j, i, cm[i, j], ha="center", va="center")

# Labels and title
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix — Random Forest")

# Save figure
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png", dpi=300)
plt.show()

# Final Output

print("\nVisualization files created successfully.")
print("- model_accuracy_comparison.png")
print("- confusion_matrix_heatmap.png")