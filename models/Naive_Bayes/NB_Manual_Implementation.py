import numpy as np

class GaussianNaiveBayesManual:
    def __init__(self):
        # Initialize attributes
        self.labels = None
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, features, labels):
        # Identify unique labels
        self.labels = np.unique(labels)

        # Calculate mean, variance, and prior for each class
        for label in self.labels:
            features_label = features[labels == label]      # Array that contain all the features of a specific class (label)
            self.means[label] = np.mean(features_label, axis=0)
            self.variances[label] = np.var(features_label, axis=0) + 1e-9  # Small value to avoid division by zero
            self.priors[label] = features_label.shape[0] / features.shape[0]
        print("Gaussian Naive Bayes (manual) model training complete.\n")

    def predict(self, features_dataset):
        # Predict class for each sample in features dataset
        return np.array([self._predict(sample) for sample in features_dataset])

    def _predict(self, sample):
        # Calculate the posterior for each class and choose the one with the highest probability
        posteriors = []
        for label in self.labels:
            prior = np.log(self.priors[label])
            likelihood = -0.5 * (np.sum(np.log(2 * np.pi * self.variances[label])) + np.sum(((sample - self.means[label]) ** 2) / self.variances[label]))
            posteriors.append(prior + likelihood)
        return self.labels[np.argmax(posteriors)]
