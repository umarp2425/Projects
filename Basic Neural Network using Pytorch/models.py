import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Function to generate synthetic data based on number of output classes
def generate_data(num_classes, num_features=20, num_samples=1000):
    X, y = make_classification(
        n_samples=num_samples, n_features=num_features, n_informative=15, 
        n_classes=num_classes, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Neural network class
class FlexibleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FlexibleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden
        self.relu = nn.ReLU()                         # Activation
        self.fc2 = nn.Linear(hidden_size, output_size) # Hidden to output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Raw logits for classification

# 3. Training function
def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=20, batch_size=32):
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. Evaluation function
def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()
    print(f'Accuracy: {accuracy:.4f}')

# 5. Main script to run the model with flexible outcomes
def main(num_classes):
    # Generate data
    X_train, X_test, y_train, y_test = generate_data(num_classes=num_classes)
    
    # Preprocess the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Define model parameters
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = num_classes  # Flexible based on `num_classes`
    
    # Initialize model, criterion, and optimizer
    model = FlexibleNeuralNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print(f"Training model with {num_classes} output classes...")
    train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor)
    
    # Evaluate the model
    print(f"Evaluating model with {num_classes} output classes...")
    evaluate_model(model, X_test_tensor, y_test_tensor)

# Run with 2, 3, or more output classes
if __name__ == "__main__":
    num_classes = int(input("Enter the number of output classes (e.g., 2 for binary classification): "))
    main(num_classes)
