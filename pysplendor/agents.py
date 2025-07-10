import random
import torch

from .game_state import GameState, Agent
from .splendor import ACTION_IDS, ACTIONS_STR
from .mcts import MCTS, Policy, PolicyMCTS

class RandomAgent(Agent):
    def __init__(self, name: str = 'random'):
        super().__init__(name)

    def get_action(self, game_state: GameState):
        legal_actions = game_state.get_actions()
        return random.choice(legal_actions)

class HumanPlayer(Agent):
    def __init__(self, name: str = 'human'):
        super().__init__(name)
    
    def get_action(self, game_state):
        valid_actions = [ACTIONS_STR[id] for id in game_state.get_actions()]
        print(f'Valid moves: {",".join(valid_actions)}')
        while True:
            action_str = input(f'player {game_state.active_player()} move: ')
            if action_str not in ACTION_IDS:
                print(f'Invalid move: {action_str}')
            elif action_str not in valid_actions:
                print(f'Forbidden move: {action_str}')
            else:
                return ACTION_IDS[action_str]

class MCTSAgent(Agent):
    def __init__(self, name: str = 'mcts'):
        super().__init__(name)
    
    def __init__(self, mcts_params = None):
        self.mcts_params = mcts_params

    def get_action(self, game_state: GameState):
        mcts = MCTS(game_state, self.mcts_params)
        action = mcts.search()
        return action

class ConstantPolicy(Policy):
    def __init__(self, probs):
        self.probs = probs

    def predict(self, game_state: GameState):
        action_probs = [self.probs[a] for a in game_state.get_actions()]
        return action_probs

class NNPolicy(Policy):
    def __init__(self, model, state_encoder):
        self.model = model
        self.state_encoder = state_encoder

    def predict(self, game_state: GameState):
        state_vec = self.state_encoder.state_to_vec(game_state)
        X = torch.tensor(state_vec, dtype=torch.float32)
        probs = self.model.forward(X).detach().numpy()
        action_probs = [probs[a] for a in game_state.get_actions()]
        return action_probs

class PolicyMCTSAgent(Agent):
    def __init__(self, policy, mcts_params, name: str = 'policy'):
        super().__init__(name)
        self.policy = policy
        self.mcts_params = mcts_params

    def get_action(self, game_state: GameState):
        mcts = PolicyMCTS(game_state, self.policy, self.mcts_params)
        action = mcts.search()
        return action

def load_mlp_model(model_path):
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    return model








    

