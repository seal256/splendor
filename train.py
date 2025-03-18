from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

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

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss_action = 0.0
    total_loss_qval = 0.0
    action_correct = 0
    num_samples = 0
    
    for X_batch, y_action_batch, y_qval_batch in data_loader:
        X_batch, y_action_batch, y_qval_batch = X_batch.to(device), y_action_batch.to(device), y_qval_batch.to(device)
        optimizer.zero_grad()
        
        action_output, qval_output = model(X_batch)
        
        loss_action = F.cross_entropy(action_output, y_action_batch)
        loss_qval = F.mse_loss(qval_output.squeeze(), y_qval_batch)
        loss = loss_action + loss_qval
        
        loss.backward()
        optimizer.step()
        
        total_loss_action += loss_action.item()
        total_loss_qval += loss_qval.item()
        
        action_correct += torch.sum(torch.argmax(action_output, dim=1) == y_action_batch).item()
        num_samples += y_action_batch.size(0)
    
    num_batches = len(data_loader)
    action_accuracy = action_correct / num_samples
    
    return total_loss_action / num_batches, total_loss_qval / num_batches, action_accuracy

def validate(model, data_loader, device):
    model.eval()
    total_loss_action = 0.0
    total_loss_qval = 0.0
    action_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for X_batch, y_action_batch, y_qval_batch in data_loader:
            X_batch, y_action_batch, y_qval_batch = X_batch.to(device), y_action_batch.to(device), y_qval_batch.to(device)
            
            action_output, qval_output = model(X_batch)
            
            loss_action = F.cross_entropy(action_output, y_action_batch) # averaged over the batch
            loss_qval = F.mse_loss(qval_output.squeeze(), y_qval_batch)

            total_loss_action += loss_action.item()
            total_loss_qval += loss_qval.item()            
            action_correct += torch.sum(torch.argmax(action_output, dim=1) == y_action_batch).item()
            num_samples += y_action_batch.size(0)
    
    num_batches = len(data_loader)
    action_accuracy = action_correct / num_samples
    
    return total_loss_action / num_batches, total_loss_qval / num_batches, action_accuracy

def print_weigths(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_norm = torch.norm(param, p=2).item()
            grad_norm = torch.norm(param.grad, p=2).item() if param.grad is not None else 0.0
            print(f"{name}: weight norm: {weight_norm:.4f}, grad norm: {grad_norm:.4f}")
    print()


def train_loop(model, train_loader, val_loader, optimizer, device, num_epochs, verbose=True):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        train_loss_action, train_loss_qval, train_action_accuracy = train_epoch(model, train_loader, optimizer, device)
        val_loss_action, val_loss_qval, val_action_accuracy = validate(model, val_loader, device)
        
        curr_time = datetime.now().strftime("%H:%M:%S")
        print(f"{curr_time} Epoch {epoch+1}/{num_epochs}, "
              f"train loss action: {train_loss_action:.4f}, loss qval: {train_loss_qval:.4f}, accuracy: {train_action_accuracy:.4f}, "
              f"val loss action: {val_loss_action:.4f}, loss qval: {val_loss_qval:.4f}, accuracy: {val_action_accuracy:.4f}")
        if verbose:
            print_weigths(model)

    print("done!")

def train():
    seed = 1828
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = 128

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'device: {device}')

    model_path = './data/models/mlp_10k.pth'
    model = TwoHeadMLP(input_size=STATE_LEN, hidden_size=100, num_actions=NUM_ACTIONS)
    # model.load_state_dict(torch.load(model_path))

    dataset = SplendorDataset(data_fname_prefix='./data/train/iter0_10k')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    train_loop(model, train_loader, val_loader, optimizer, device, num_epochs=20, verbose=False)

    torch.save(model.state_dict(), model_path)
    print(f'Final model saved to {model_path}')


def model_predict(model, state_encoder, state, K=5):
    '''Retruns top K predicted actions and qvalue'''
    state_vec = state_encoder.state_to_vec(state)
    X = torch.tensor(state_vec, dtype=torch.float32)
    logits, qval = model.forward(X)
    top_actions = np.argsort(logits.detach().numpy())[-K:]
    logits = logits[top_actions]
    return top_actions, logits, qval.item()

def run_model():
    '''Aloows to inspect the moves predicted by the model'''
    from prepare_data import SplendorGameStateEncoder, ALL_ACTIONS
    from pysplendor.game import Trajectory, traj_loader
    from pysplendor.splendor import Action, CHANCE_PLAYER
    state_encoder = SplendorGameStateEncoder(2)
    STATE_LEN = 1052
    NUM_ACTIONS = 43

    model = TwoHeadMLP(STATE_LEN, 50, NUM_ACTIONS)
    model_path = './data/models/mlp_0.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    traj_file = './data/traj_dump_10k.txt'
    loader = traj_loader(traj_file)
    for _ in range(1100):
        next(loader)
    traj = next(loader) # pick one
    state = traj.initial_state.copy()
    rewards = traj.rewards

    for action in traj.actions:
        if state.active_player() != CHANCE_PLAYER: # ignore chance nodes
            top_actions, logits, qval = model_predict(model, state_encoder, state, K=5)
            print(state)
            suggested_actions = ' '.join([f'{ALL_ACTIONS[a]} ({l:.2f})' for a, l in zip(top_actions, logits)])
            print(f'predicted actions: {suggested_actions} qval: {qval:.3f} reward: {rewards[state.active_player()]}')
            print(f'actual action: {action}\n')

        state.apply_action(action)

if __name__ == "__main__":

    train()
    # run_model()
