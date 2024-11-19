import numpy as np
import pandas as pd
from models.evaluation_utils import plot_confusion_matrix, evaluate_model

# Load predictions and labels
data = np.load('../../features/cifar10_features_50d.npz')
test_labels = data['test_labels']
manual_predictions = np.load('manual_dt_predictions.npz')['predictions']
sklearn_predictions = np.load('sklearn_dt_predictions.npz')['predictions']

# Evaluate both models
accuracy_manual, precision_manual, recall_manual, f1_manual, conf_matrix_manual = evaluate_model("Manual Decision Tree", test_labels, manual_predictions)
accuracy_sklearn, precision_sklearn, recall_sklearn, f1_sklearn, conf_matrix_sklearn = evaluate_model("Scikit-Learn Decision Tree", test_labels, sklearn_predictions)

plot_confusion_matrix(conf_matrix_manual, "Confusion Matrix - Manual Decision Tree")
plot_confusion_matrix(conf_matrix_sklearn, "Confusion Matrix - Scikit-Learn Decision Tree")

# Display results in a summary table
results_df = pd.DataFrame({
    'Model': ['Manual Decision Tree', 'Scikit-Learn Decision Tree'],
    'Accuracy': [accuracy_manual, accuracy_sklearn],
    'Precision': [precision_manual, precision_sklearn],
    'Recall': [recall_manual, recall_sklearn],
    'F1 Score': [f1_manual, f1_sklearn]
})
print("\nEvaluation Summary Table:")
print(results_df)

