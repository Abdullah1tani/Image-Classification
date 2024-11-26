import numpy as np

from DT_manual_implementation import  SimpleDecisionTree
from sklearn.tree import DecisionTreeClassifier

# Load training data and fit the manual decision tree model
data = np.load('../../features/cifar10_features_50d.npz')
train_features, train_labels = data['train_features'], data['train_labels']
test_features, test_labels = data['test_features'], data['test_labels']

# Initialize train and test the SimpleDecisionTree model
dt_manual = SimpleDecisionTree(max_depth=50)
dt_manual.fit(train_features, train_labels)
manual_predictions = dt_manual.predict(test_features)

# Save predictions
np.savez('manual_dt_predictions.npz', predictions=manual_predictions)
print("Manual Decision Tree predictions saved.")

# Initialize train and test the DecisionTreeClassifier model
dt_sklearn = DecisionTreeClassifier(criterion='gini', max_depth=50, random_state=0)
dt_sklearn.fit(train_features, train_labels)
sklearn_predictions = dt_sklearn.predict(test_features)

# Save predictions for evaluation
np.savez('sklearn_dt_predictions.npz', predictions=sklearn_predictions)
print("Scikit-Learn Decision Tree predictions saved.")