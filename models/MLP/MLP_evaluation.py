#MLP Evaluation

import pandas as pd

from models.MLP.MLP_implementation import test_labels_tensor
from models.evaluation_utils import plot_confusion_matrix, mlp_evaluate_model

# Load test labels
test_labels_np = test_labels_tensor.numpy()

# Evaluate the original MLP model
accuracy, precision, recall, f1, conf_matrix = mlp_evaluate_model(mlp_model, test_loader, "Original MLP")

plot_confusion_matrix(conf_matrix, "Confusion Matrix - Original MLP")

# Save evaluation summary to display different depth and hidden size results
results_df = pd.DataFrame({
    'Model': ['Original MLP'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1]
})
print("\nEvaluation Summary Table:")
print(results_df)
