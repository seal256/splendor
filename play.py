import torch
import numpy as np
import random

from pysplendor.game_state import GameState, CHANCE_PLAYER
from pysplendor.agents import RandomAgent, MCTSAgent, Agent
from pysplendor.splendor import SplendorGameState, Action
from pysplendor.game import run_one_game, Trajectory, traj_loader
from pysplendor.mcts import PolicyMCTS, ActionEncoder, Policy
from prepare_data import SplendorGameStateEncoder, ALL_ACTIONS, ACTION_ID
from train import TwoHeadMLP


class NNPolicy(Policy):
    def __init__(self, model, state_encoder):
        self.model = model
        self.state_encoder = state_encoder

    def predict(self, state):
        state_vec = self.state_encoder.state_to_vec(state)
        X = torch.tensor(state_vec, dtype=torch.float32)
        logits, _ = self.model.forward(X)
        return logits.detach().numpy()


class SplendorActionEncoder(ActionEncoder):
    def __init__(self):
        self.action_dict = ACTION_ID

    def action_id(self, action):
        return self.action_dict[str(action)]


class PolicyMCTSAgent(Agent):
    def __init__(self, **mcts_params):
        self.mcts_params = mcts_params

    def get_action(self, game_state: GameState):
        mcts = PolicyMCTS(game_state, **self.mcts_params)
        action = mcts.search()
        return action


def load_mlp_model(model_path):
    STATE_LEN = 1052
    NUM_ACTIONS = 43
    model = TwoHeadMLP(STATE_LEN, 100, NUM_ACTIONS)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    return model

def tournament(agents, num_games, verbose=True):
    results = []
    for round in range(num_games):
        print(f'\nRound {round}')
        game_state = SplendorGameState(len(agents))
        traj = run_one_game(game_state, agents, verbose)
        results.append(traj.rewards)
    
    for id in range(len(agents)):
        wins = sum([r[id] for r in results])
        print(f'palyer {id} wins: {wins}')

if __name__ == '__main__':
    random.seed(11)

    model = load_mlp_model('./data/models/mlp_10k_100e.pth')
    state_encoder = SplendorGameStateEncoder(2)
    nn_agent = PolicyMCTSAgent(policy=NNPolicy(model, state_encoder), action_encoder=SplendorActionEncoder(), policy_weight=0.5, iterations=1000)
    mcts_agent = MCTSAgent(iterations=1000)

    agents = [mcts_agent, nn_agent]
    game_state = SplendorGameState(len(agents))
    # traj = run_one_game(game_state, agents, verbose=True)
    tournament(agents, num_games=10)