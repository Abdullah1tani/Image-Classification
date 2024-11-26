import torch
import torch.nn as nn
import torch.optim as optim

from models.evaluation_utils import evaluate_and_display

# Training function for mlp
def mlp_train(model, train_loader):
    print(f'Training {model.get_name()}:')

    # Loss function to compute the error
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # set learning rate and momentum
    epochs = 20  # Number of training epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)           # compute predictions
            loss = criterion(outputs, labels)   # measure difference between predictions and true labels
            loss.backward()
            optimizer.step()         # update weights
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Save the trained model weights
    torch.save(model.state_dict(), f'{model.get_name()}_weights.pth')
    print(f"{model.get_name()} model training complete and weights saved to {model.get_name()}_weights.pth.\n")

# Testing mlp
def mlp_test(model, test_loader, test_labels, name):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in test_loader:
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.numpy())

    # print accuracy, precision, recall and f1 score and plot confusion matrix
    return evaluate_and_display(name , test_labels, predictions)

# the three-layer MLP class contains 3 linear, 2 ReLU and 1 BatchNorm layer
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, name):
        super(ThreeLayerMLP, self).__init__()
        self.name = name

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.batch_norm = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def get_name(self):
        return self.name

# the shallow MLP class contains 2 linear, 1 ReLU and 0 BatchNorm layer
class ShallowMLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, output_size=10):
        super(ShallowMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    def get_name(self):
        return "shallow_layer_mlp"

# the intermediate MLP class contains 4 linear, 3 ReLU and 1 BatchNorm layer
class IntermediateMLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, output_size=10):
        super(IntermediateMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.batch_norm(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

    def get_name(self):
        return "intermediate_layer_mlp"

# the Deep MLP class contains 5 linear, 4 ReLU and 1 BatchNorm layer
class DeepMLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, output_size=10):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.batch_norm(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x

    def get_name(self):
        return "deep_layer_mlp"