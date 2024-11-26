import numpy as np

from sklearn.decomposition import PCA

# Load the extracted features (512-dimensional) from cifar10_features.npz
data = np.load('cifar10_features.npz')
train_features, train_labels = data['train_features'], data['train_labels']
test_features, test_labels = data['test_features'], data['test_labels']

# Check the initial dimensionality to ensure it's 512 for both train and test features.
assert train_features.shape[1] == 512, "Training features should be 512-dimensional."
assert test_features.shape[1] == 512, "Test features should be 512-dimensional."

# Apply PCA to reduce the feature vectors from 512 dimensions to 50 dimensions.
pca = PCA(n_components=50)

# Fit PCA on the training features and transform them to the reduced space.
train_features_50d = pca.fit_transform(train_features)

# Apply the same PCA transformation (from the training data) to the test features.
test_features_50d = pca.transform(test_features)

# Save the reduced-dimensional features (50 dimensions) and corresponding labels to a new .npz file.
np.savez('cifar10_features_50d.npz',
         train_features=train_features_50d,
         train_labels=train_labels,
         test_features=test_features_50d,
         test_labels=test_labels)

# confirm that PCA has been applied successfully.
print("PCA applied successfully. Features reduced to 50 dimensions and saved to 'cifar10_features_50d.npz'.")

# Print the shape of the reduced features.
# Train features should now have 50 dimensions instead of 512.
print(f"Train features shape after PCA: {train_features_50d.shape}")
print(f"Test features shape after PCA: {test_features_50d.shape}")