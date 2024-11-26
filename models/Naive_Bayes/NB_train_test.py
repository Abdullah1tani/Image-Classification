import numpy as np

# import the manual naive bayes implementation and the Scikit’s Gaussian Naive Bayes classifier
from NB_Manual_Implementation import GaussianNaiveBayesManual
from sklearn.naive_bayes import GaussianNB

# Load the 50-dimensional feature vectors and labels from PCA-reduced file
data = np.load('../../features/cifar10_features_50d.npz')
train_features, train_labels = data['train_features'], data['train_labels']
test_features, test_labels = data['test_features'], data['test_labels']

# Initialize, train, and predict with the manual Gaussian Naive Bayes model
nb_manual = GaussianNaiveBayesManual()
nb_manual.fit(train_features, train_labels)
manual_predictions = nb_manual.predict(test_features)

# Save predictions for later use
np.savez('manual_nb_predictions.npz', predictions=manual_predictions)
print("Manual Gaussian Naive Bayes predictions saved.")

# Initialize, train, and predict with Scikit-Learn's Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(train_features, train_labels)
scikit_preds = gnb.predict(test_features)

# Save predictions for reuse
np.savez('scikit_nb_predictions.npz', predictions=scikit_preds)
print("\nScikit-Learn Gaussian Naive Bayes predictions saved.")