import numpy as np
import torch
from torch.utils.data import Subset

# Function to select the first `num_samples_per_class` images from each class
def select_subset(dataset, num_samples_per_class):
    # Initialize a dictionary to keep track of selected indices for each class
    class_indices = {i: [] for i in range(10)}  # CIFAR-10 has 10 classes, labeled 0-9

    # Iterate over the dataset to gather the required number of samples per class
    for index, (_, label) in enumerate(dataset):
        # Check if we have already selected enough samples for this class
        if len(class_indices[label]) < num_samples_per_class:
            class_indices[label].append(index)  # Add index to the list for this class
        # Stop if we have the required number of samples for each class
        if all(len(indices) >= num_samples_per_class for indices in class_indices.values()):
            break  # Stop iterating once we have the required samples for each class

    # Flatten the indices from all classes into a single list for creating the subset
    subset_indices = [index for indices in class_indices.values() for index in indices]
    return Subset(dataset, subset_indices)  # Return the subset with selected indices only

# Function to extract features from a dataset
def extract_features(data_loader, model):
    i = 0
    features = []
    labels = []
    with torch.no_grad():  # No gradient calculation needed for feature extraction
        for images, lbls in data_loader:
            output = model(images).squeeze()  # Get 512-dimensional feature vector
            print(f"successfully extracted features for batch {i}")
            i += 1
            features.append(output)
            labels.extend(lbls.numpy())       # Store corresponding labels
    return torch.cat(features).numpy(), np.array(labels)