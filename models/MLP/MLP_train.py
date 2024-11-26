import torch

from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from MLP_implementation import mlp_train, ThreeLayerMLP, IntermediateMLP, ShallowMLP, DeepMLP

# Load CIFAR-10 feature vectors (from PCA-reduced file with 50 dimensions)
data = np.load('../../features/cifar10_features_50d.npz')
train_features, train_labels = data['train_features'], data['train_labels']
test_features, test_labels = data['test_features'], data['test_labels']

# Convert data to PyTorch tensors
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Define DataLoader for batching
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize three layer mlp model
three_layer_mlp = ThreeLayerMLP(50, 512, 512, 10, "three_layer_mlp")

# Initialize mlp models for depth evaluation
shallow_layer_mlp = ShallowMLP(50, 512, 10)
intermediate_layer_mlp = IntermediateMLP(50, 512, 10)
deep_layer_mlp = DeepMLP(50, 512, 10)

# Initialize mlp models for size evaluation
small_layer_mlp = ThreeLayerMLP(50, 128, 64, 10, "small_layer_mlp")
moderate_layer_mlp = ThreeLayerMLP(50, 256, 128, 10, "moderate_layer_mlp")
large_layer_mlp = ThreeLayerMLP(50, 512, 256, 10, "large_layer_mlp")

# train the three layer mlp
mlp_train(three_layer_mlp, train_loader)

# train mlp models for depth evaluation
mlp_train(shallow_layer_mlp, train_loader)
mlp_train(intermediate_layer_mlp, train_loader)
mlp_train(deep_layer_mlp, train_loader)

# train mlp models for size evaluation
mlp_train(small_layer_mlp, train_loader)
mlp_train(moderate_layer_mlp, train_loader)
mlp_train(large_layer_mlp, train_loader)

# save the labels and features for the loader (we need it for the evaluation)
torch.save({'test_labels': test_labels_tensor, 'test_features': test_features_tensor}, 'test_data.pth')
print("Test data saved to test_data.pth.")
