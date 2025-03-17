import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import numpy as np

STATE_LEN = 1052
NUM_ACTIONS = 43

class SplendorDataset(Dataset):
    def __init__(self, data_fname_prefix, state_len = STATE_LEN):
        """
        Args:
            data_fname_prefix (str): Prefix of the .npy files containing the states, actions, and rewards.
        """
        states = np.load(data_fname_prefix + "_states.npy", allow_pickle=True)
        actions = np.load(data_fname_prefix + "_actions.npy", allow_pickle=True)
        rewards = np.load(data_fname_prefix + "_rewards.npy", allow_pickle=True)

        states = np.unpackbits(states, axis=1, count=state_len)

        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (state, action, reward) where state is the input to the model,
                   action is the target action, and reward is the target reward.
        """
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]

        return state, action, reward


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

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    action_correct = 0
    action_total = 0
    
    for X_batch, y_action_batch, y_qval_batch in train_loader:
        X_batch, y_action_batch, y_qval_batch = X_batch.to(device), y_action_batch.to(device), y_qval_batch.to(device)
        optimizer.zero_grad()
        
        action_output, qval_output = model(X_batch)
        
        loss_action = F.cross_entropy(action_output, y_action_batch)
        loss_qval = F.mse_loss(qval_output.squeeze(), y_qval_batch)
        loss = loss_action + loss_qval
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        action_correct += torch.sum(torch.argmax(action_output, dim=1) == y_action_batch).item()
        action_total += y_action_batch.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_action_accuracy = action_correct / action_total
    
    return train_loss, train_action_accuracy

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    action_correct = 0
    action_total = 0
    
    with torch.no_grad():
        for X_batch, y_action_batch, y_qval_batch in val_loader:
            X_batch, y_action_batch, y_qval_batch = X_batch.to(device), y_action_batch.to(device), y_qval_batch.to(device)
            
            action_output, qval_output = model(X_batch)
            
            loss_action = F.cross_entropy(action_output, y_action_batch)
            loss_qval = F.mse_loss(qval_output.squeeze(), y_qval_batch)
            total_loss += (loss_action + loss_qval).item()
            
            action_correct += torch.sum(torch.argmax(action_output, dim=1) == y_action_batch).item()
            action_total += y_action_batch.size(0)
    
    total_loss = total_loss / len(val_loader)
    val_action_accuracy = action_correct / action_total
    
    return total_loss, val_action_accuracy

def print_weigths(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_norm = torch.norm(param, p=2).item()
            grad_norm = torch.norm(param.grad, p=2).item() if param.grad is not None else 0.0
            print(f"Layer {name}: weight norm: {weight_norm:.4f}, grad norm: {grad_norm:.4f}")


def train_loop(model, train_loader, val_loader, optimizer, device, num_epochs, verbose=True):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        train_loss, train_action_accuracy = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_action_accuracy = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"train loss: {train_loss:.4f}, train accuracy: {train_action_accuracy:.4f}, "
              f"val loss: {val_loss:.4f}, val accuracy: {val_action_accuracy:.4f}")
        if verbose:
            print_weigths(model)
            print()

    print("done!")

def train():
    seed = 1828
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = 32

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'device: {device}')

    dataset = SplendorDataset(data_fname_prefix='./data/train/iter0')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TwoHeadMLP(input_size=STATE_LEN, hidden_size=1000, num_actions=NUM_ACTIONS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    train_loop(model, train_loader, val_loader, optimizer, device, num_epochs=100)

    return model

if __name__ == "__main__":

    model = train()