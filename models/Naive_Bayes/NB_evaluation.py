import pandas as pd
import numpy as np

from models.evaluation_utils import evaluate_and_display

# Load training data and fit the manual decision tree model
data = np.load('../../features/cifar10_features_50d.npz')
test_labels = data['test_labels']

# Load predictions and evaluate
manual_preds = np.load('manual_nb_predictions.npz')['predictions']
scikit_preds = np.load('scikit_nb_predictions.npz')['predictions']

# Evaluate Manual Naive Bayes
manual_metrics = evaluate_and_display("Manual Naive Bayes", test_labels, manual_preds)

print()

# Evaluate Scikit-Learn Naive Bayes
scikit_metrics = evaluate_and_display("Scikit-Learn Naive Bayes", test_labels, scikit_preds)

# Summarize findings in a table
summary_table = pd.DataFrame({
    "Model": ["Manual Naive Bayes", "Scikit-Learn Naive Bayes"],
    "Accuracy": [manual_metrics[0], scikit_metrics[0]],
    "Precision": [manual_metrics[1], scikit_metrics[1]],
    "Recall": [manual_metrics[2], scikit_metrics[2]],
    "F1 Score": [manual_metrics[3], scikit_metrics[3]]
})

# Display the summary table
print("\nSummary Table:")
print(summary_table)