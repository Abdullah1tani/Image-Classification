import pandas as pd
import numpy as np

from models.evaluation_utils import evaluate_and_display

# Load PCA-reduced features and labels
data = np.load('../../features/cifar10_features_50d.npz')
test_labels = data['test_labels']

# Load predictions for both models
manual_preds = np.load('manual_dt_predictions.npz')['predictions']
sklearn_preds = np.load('sklearn_dt_predictions.npz')['predictions']

# Evaluate and display metrics for manual decision tree
manual_metrics = evaluate_and_display("Manual Decision Tree", test_labels, manual_preds)

print()

# Evaluate and display metrics for Scikit-Learn Decision Tree
sklearn_metrics = evaluate_and_display("Scikit-Learn Decision Tree", test_labels, sklearn_preds)

# Summarize findings in a table
summary_table = pd.DataFrame({
    "Model": ["Manual Decision Tree", "Scikit-Learn Decision Tree"],
    "Accuracy": [manual_metrics[0], sklearn_metrics[0]],
    "Precision": [manual_metrics[1], sklearn_metrics[1]],
    "Recall": [manual_metrics[2], sklearn_metrics[2]],
    "F1 Score": [manual_metrics[3], sklearn_metrics[3]]
})

# Display the summary table
print("\nSummary Table:")
print(summary_table)