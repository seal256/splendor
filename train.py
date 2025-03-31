from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from prepare_data import ALL_ACTIONS
STATE_LEN = 1052
NUM_ACTIONS = 43

class SplendorDataset(Dataset):
    def __init__(self, data_fname_prefix, state_len = STATE_LEN):
        """
        Args:
            data_fname_prefix (str): Prefix of the .npy files containing the states, actions, and rewards.
        
        Notes:
            keeps all the data in memory
        """
        states = np.load(data_fname_prefix + "_states.npy", allow_pickle=True)
        actions = np.load(data_fname_prefix + "_actions.npy", allow_pickle=True)
        rewards = np.load(data_fname_prefix + "_rewards.npy", allow_pickle=True)

        states = np.unpackbits(states, axis=1, count=state_len)

        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
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


class MLP(nn.Module):
    def __init__(self, input_size=STATE_LEN, hidden_size=512, num_actions=NUM_ACTIONS):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))        
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return x + self.block(x)


class ResNet(nn.Module):
    def __init__(self, input_size=STATE_LEN, hidden_size=512, num_blocks=3, num_actions=NUM_ACTIONS):
        super(ResNet, self).__init__()
        self.initial_fc = nn.Linear(input_size, hidden_size)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        self.final_fc = nn.Linear(hidden_size, num_actions)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.initial_fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.initial_fc.bias)
        nn.init.zeros_(self.final_fc.weight)
        nn.init.zeros_(self.final_fc.bias)
        
    def forward(self, x):
        x = F.relu(self.initial_fc(x))
        for block in self.res_blocks:
            x = block(x)        
        return F.softmax(self.final_fc(x), dim=-1)

def loss(output, target):
    return -torch.mean(torch.sum(target * torch.log(output + 1e-10), dim=1))
    # return F.kl_div(output, target)

def data_loss(data_loader, criterion):
    '''Computes irreducible entrpy of the target probability distibution'''
    total_loss = 0.0
    for _, y_action_batch, _ in data_loader:
        total_loss += criterion(y_action_batch, y_action_batch).item()
    
    return total_loss / len(data_loader)

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    classif_pred = []
    classif_correct = []
    
    for X_batch, y_action_batch, _ in data_loader:
        X_batch, y_action_batch = X_batch.to(device), y_action_batch.to(device)
        optimizer.zero_grad()
        
        action_output = model(X_batch)
        
        loss = criterion(action_output, y_action_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        classif_pred.extend(torch.argmax(action_output, dim=1).cpu().numpy())
        classif_correct.extend(torch.argmax(y_action_batch, dim=1).cpu().numpy())
    
    return total_loss / len(data_loader), classif_pred, classif_correct

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    classif_pred = []
    classif_correct = []
    
    with torch.no_grad():
        for X_batch, y_action_batch, _ in data_loader:
            X_batch, y_action_batch = X_batch.to(device), y_action_batch.to(device)
            
            action_output = model(X_batch)
            total_loss += criterion(action_output, y_action_batch).item()

            classif_pred.extend(torch.argmax(action_output, dim=1).cpu().numpy())
            classif_correct.extend(torch.argmax(y_action_batch, dim=1).cpu().numpy())

    return total_loss / len(data_loader), classif_pred, classif_correct

def print_weigths(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_norm = torch.norm(param, p=2).item()
            grad_norm = torch.norm(param.grad, p=2).item() if param.grad is not None else 0.0
            print(f"{name}: weight norm: {weight_norm:.4f}, grad norm: {grad_norm:.4f}")
    print()


def train_loop(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, verbose=True):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        train_loss, train_classif_pred, train_classif_correct = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_classif_pred, val_classif_correct = validate(model, val_loader, criterion, device)
        
        train_accuracy = accuracy_score(train_classif_correct, train_classif_pred)
        val_accuracy = accuracy_score(val_classif_correct, val_classif_pred)

        curr_time = datetime.now().strftime("%H:%M:%S")
        print(f"{curr_time} Epoch {epoch+1}/{num_epochs}, "
              f"train loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}, "
              f"val loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
        
        if verbose:
            print_weigths(model)
            print(classification_report(val_classif_correct, val_classif_pred, target_names = ALL_ACTIONS, zero_division=0))

    print("done!")

def train():
    seed = 1828
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = 128

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'device: {device}')

    model_path = './data/models/mlp.pth'
    model = MLP(hidden_size=512)
    # model_path = './data/models/resnet_10k.pth'
    # model = ResNet(hidden_size=512, num_blocks=3)
    # model.load_state_dict(torch.load(model_path))

    train_dataset = SplendorDataset(data_fname_prefix='./data/train/iter0')
    val_dataset = SplendorDataset(data_fname_prefix='./data/val/iter0')
    print(f'train set len: {len(train_dataset)} val set len: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-5)
    criterion = loss

    train_data_entropy = data_loss(train_loader, criterion)
    val_data_entropy = data_loss(val_loader, criterion)
    print(f'train data entropy: {train_data_entropy:.4f}, val data entropy: {val_data_entropy:.4f}')

    train_loop(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=20, verbose=True)

    torch.save(model.state_dict(), model_path)
    print(f'Final model is saved to {model_path}')

    sm_model_path = model_path[:-1] # without h at the end of the name
    sm = torch.jit.script(model)
    sm.save(sm_model_path)
    print(f'Jit script model is saved to {sm_model_path}')

# def export_model_with_torchscript():
#     model_path = './data/models/mlp_10k_bw.pth'
#     model = MLP(input_size=STATE_LEN, hidden_size=100, num_actions=NUM_ACTIONS)
#     model.load_state_dict(torch.load(model_path))
#     sm = torch.jit.script(model)
#     sm.save('./data/models/mlp_10k_bw.pt')

if __name__ == "__main__":

    train()

