import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader

from MLP_implementation import mlp_test, ThreeLayerMLP, ShallowMLP, IntermediateMLP, DeepMLP

# load the data
saved_data = torch.load('test_data.pth', weights_only=True)
test_labels = saved_data['test_labels']
test_features = saved_data['test_features']

# Recreate the DataLoader
test_dataset = TensorDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

# load weights into the models
three_layer_mlp.load_state_dict(torch.load('three_layer_mlp_weights.pth', weights_only=True))
shallow_layer_mlp.load_state_dict(torch.load('shallow_layer_mlp_weights.pth', weights_only=True))
intermediate_layer_mlp.load_state_dict(torch.load('intermediate_layer_mlp_weights.pth', weights_only=True))
deep_layer_mlp.load_state_dict(torch.load('deep_layer_mlp_weights.pth', weights_only=True))
small_layer_mlp.load_state_dict(torch.load('small_layer_mlp_weights.pth', weights_only=True))
moderate_layer_mlp.load_state_dict(torch.load('moderate_layer_mlp_weights.pth', weights_only=True))
large_layer_mlp.load_state_dict(torch.load('large_layer_mlp_weights.pth', weights_only=True))

# Set the models to evaluation mode
three_layer_mlp.eval()
shallow_layer_mlp.eval()
intermediate_layer_mlp.eval()
deep_layer_mlp.eval()
small_layer_mlp.eval()
moderate_layer_mlp.eval()
large_layer_mlp.eval()

# Evaluate every mlp model and plot confusion matrix
three_layer_metrics = mlp_test(three_layer_mlp, test_loader, test_labels, "Three Layer MLP")
print()
shallow_metrics = mlp_test(shallow_layer_mlp, test_loader, test_labels, "Shallow Layer MLP")
print()
intermediate_metrics = mlp_test(intermediate_layer_mlp, test_loader, test_labels, "Intermediate Layer MLP")
print()
deep_metrics = mlp_test(deep_layer_mlp, test_loader, test_labels, "Deep Layer MLP")
print()
small_metrics = mlp_test(small_layer_mlp, test_loader, test_labels, "Small Layer MLP")
print()
moderate_metrics = mlp_test(moderate_layer_mlp, test_loader, test_labels, "Moderate Layer MLP")
print()
large_metrics = mlp_test(large_layer_mlp, test_loader, test_labels, "Large Layer MLP")

# Save evaluation summary to display depth results
depth_evaluation_results = pd.DataFrame({
    'Model': ['Three Layer MLP', 'Shallow MLP', 'Intermediate MLP', 'Deep MLP'],
    'Accuracy': [three_layer_metrics[0], shallow_metrics[0], intermediate_metrics[0], deep_metrics[0]],
    'Precision': [three_layer_metrics[1], shallow_metrics[1],  intermediate_metrics[1], deep_metrics[1]],
    'Recall': [three_layer_metrics[2], shallow_metrics[2], intermediate_metrics[2], deep_metrics[2]],
    'F1 Score': [three_layer_metrics[3], shallow_metrics[3], intermediate_metrics[3], deep_metrics[3]]
})

print("\nDepth Evaluation Summary Table:")
print(depth_evaluation_results)

# Save evaluation summary to display different size results
size_evaluation_results = pd.DataFrame({
    'Model': ['Three Layer MLP', 'Small MLP', 'Moderate MLP', 'Large MLP'],
    'Accuracy': [three_layer_metrics[0], small_metrics[0], moderate_metrics[0], large_metrics[0]],
    'Precision': [three_layer_metrics[1], small_metrics[1],  moderate_metrics[1], large_metrics[1]],
    'Recall': [three_layer_metrics[2], small_metrics[2], moderate_metrics[2], large_metrics[2]],
    'F1 Score': [three_layer_metrics[3], small_metrics[3], moderate_metrics[3], large_metrics[3]]
})

print("\nSize Evaluation Summary Table:")
print(size_evaluation_results)

# Save evaluation summary to display all results
mlp_evaluation_results = pd.DataFrame({
    'Model': ['Three Layer MLP', 'Shallow MLP', 'Intermediate MLP', 'Deep MLP', 'Small MLP', 'Moderate MLP', 'Large MLP'],
    'Accuracy': [three_layer_metrics[0], shallow_metrics[0], intermediate_metrics[0], deep_metrics[0], small_metrics[0], moderate_metrics[0], large_metrics[0]],
    'Precision': [three_layer_metrics[1], shallow_metrics[1],  intermediate_metrics[1], deep_metrics[1], small_metrics[1],  moderate_metrics[1], large_metrics[1]],
    'Recall': [three_layer_metrics[2], shallow_metrics[2], intermediate_metrics[2], deep_metrics[2], small_metrics[2], moderate_metrics[2], large_metrics[2]],
    'F1 Score': [three_layer_metrics[3], shallow_metrics[3], intermediate_metrics[3], deep_metrics[3], small_metrics[3], moderate_metrics[3], large_metrics[3]]
})

print("\nAll Models Evaluation Summary Table:")
print(mlp_evaluation_results)