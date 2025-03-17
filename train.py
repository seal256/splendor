import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

class TwoHeadMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(TwoHeadMLP, self).__init__()
        self.shared_layer = nn.Linear(input_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.qval_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        shared_output = torch.relu(self.shared_layer(x))
        action_output = self.action_head(shared_output)
        qval_output = torch.sigmoid(self.qval_head(shared_output))
        return action_output, qval_output

def train_epoch(model, train_loader, criterion_action, criterion_qval, optimizer, device):
    model.train()
    running_loss = 0.0
    action_correct = 0
    action_total = 0
    
    for X_batch, y_action_batch, y_qval_batch in train_loader:
        X_batch, y_action_batch, y_qval_batch = X_batch.to(device), y_action_batch.to(device), y_qval_batch.to(device)
        optimizer.zero_grad()
        
        action_output, qval_output = model(X_batch)
        
        loss_action = criterion_action(action_output, y_action_batch)
        loss_qval = criterion_qval(qval_output.squeeze(), y_qval_batch)
        loss = loss_action + loss_qval
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        action_correct += torch.sum(torch.argmax(action_output, dim=1) == y_action_batch).item()
        action_total += y_action_batch.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_action_accuracy = action_correct / action_total
    
    return train_loss, train_action_accuracy

def validate(model, val_loader, criterion_action, criterion_qval, device):
    model.eval()
    total_loss = 0.0
    action_correct = 0
    action_total = 0
    
    with torch.no_grad():
        for X_batch, y_action_batch, y_qval_batch in val_loader:
            X_batch, y_action_batch, y_qval_batch = X_batch.to(device), y_action_batch.to(device), y_qval_batch.to(device)
            
            action_output, qval_output = model(X_batch)
            
            loss_action = criterion_action(action_output, y_action_batch)
            loss_qval = criterion_qval(qval_output.squeeze(), y_qval_batch)
            total_loss += (loss_action + loss_qval).item()
            
            action_correct += torch.sum(torch.argmax(action_output, dim=1) == y_action_batch).item()
            action_total += y_action_batch.size(0)
    
    total_loss = total_loss / len(val_loader)
    val_action_accuracy = action_correct / action_total
    
    return total_loss, val_action_accuracy

def train_loop(model, train_loader, val_loader, criterion_action, criterion_qval, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        train_loss, train_action_accuracy = train_epoch(model, train_loader, criterion_action, criterion_qval, optimizer, device)
        val_loss, val_action_accuracy = validate(model, val_loader, criterion_action, criterion_qval, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"train loss: {train_loss:.4f}, train accuracy: {train_action_accuracy:.4f}, "
              f"val loss: {val_loss:.4f}, val accuracy: {val_action_accuracy:.4f}")
    
    print("done!")

def train():
    np.random.seed(42)
    torch.manual_seed(42)

    input_size = 10
    hidden_size = 20
    num_actions = 3
    num_samples = 1000
    batch_size = 32
    num_epochs = 100
    seed = 42

    X = np.random.randn(num_samples, input_size)
    y_action = np.random.randint(0, num_actions, num_samples)
    y_qval = np.random.rand(num_samples)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'device: {device}')

    X_train, X_val, y_action_train, y_action_val, y_qval_train, y_qval_val = train_test_split(
        X, y_action, y_qval, test_size=0.1, random_state=seed
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_action_train = torch.tensor(y_action_train, dtype=torch.long)
    y_action_val = torch.tensor(y_action_val, dtype=torch.long)
    y_qval_train = torch.tensor(y_qval_train, dtype=torch.float32)
    y_qval_val = torch.tensor(y_qval_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_action_train, y_qval_train)
    val_dataset = TensorDataset(X_val, y_action_val, y_qval_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TwoHeadMLP(input_size, hidden_size, num_actions).to(device)
    criterion_action = nn.CrossEntropyLoss()
    criterion_qval = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_loop(model, train_loader, val_loader, criterion_action, criterion_qval, optimizer, num_epochs, device)

    return model

if __name__ == "__main__":

    model = train()