import torch
import numpy as np
import random, json
from concurrent.futures import ProcessPoolExecutor

from pysplendor.game_state import GameState, CHANCE_PLAYER
from pysplendor.agents import RandomAgent, MCTSAgent, Agent
from pysplendor.splendor import SplendorGameState, Action
from pysplendor.game import run_one_game, Trajectory, traj_loader
from pysplendor.mcts import MCTS, PVMCTS, Value, PolicyMCTS, Policy, MCTSParams
from prepare_data import SplendorGameStateEncoder, ALL_ACTIONS, ACTION_ID
from train import STATE_LEN, NUM_ACTIONS, MLP

class ConstantPolicy(Policy):
    def __init__(self, probs, action_ids):
        self.probs = probs
        self.action_ids = action_ids

    def predict(self, game_state: GameState):
        action_probs = [self.probs[self.action_ids.get(str(a))] for a in game_state.get_actions()]
        return action_probs

class NNPolicy(Policy):
    def __init__(self, model, state_encoder, action_ids):
        self.model = model
        self.state_encoder = state_encoder
        self.action_ids = action_ids

    def predict(self, game_state: GameState):
        state_vec = self.state_encoder.state_to_vec(game_state)
        X = torch.tensor(state_vec, dtype=torch.float32)
        probs = self.model.forward(X).detach().numpy()
        action_probs = [probs[self.action_ids.get(str(a))] for a in game_state.get_actions()]
        return action_probs

class AccumValue(Value):
    def __init__(self):
        super().__init__()
        # self.gamma = gamma
        self.score_norm = 15.0

    def predict(self, game_state: SplendorGameState):
        '''Returns accumulated reward up to the point'''
        # n = game_state.move_num()
        values = [player.points / self.score_norm for player in game_state.players]
        return values


class PolicyMCTSAgent(Agent):
    def __init__(self, policy, mcts_params):
        self.policy = policy
        self.mcts_params = mcts_params

    def get_action(self, game_state: GameState):
        mcts = PolicyMCTS(game_state, self.policy, self.mcts_params)
        action = mcts.search()
        return action

def load_mlp_model(model_path):
    model = MLP(STATE_LEN, 512, NUM_ACTIONS)
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

def run_tournament_round(agents):
    game_state = SplendorGameState(len(agents))
    traj = run_one_game(game_state, agents, verbose=False)
    return traj.rewards

def tournament_parallel(agents, num_games, num_workers=8):
    args = [agents for _ in range(num_games)]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_tournament_round, args))
    
    for id in range(len(agents)):
        wins = sum(r[id] for r in results)
        print(f'player {id} wins: {wins}')

def print_game_record(traj: Trajectory):
    '''Displays a recorded trajectory to console with some extra info. Debugging tool.'''

    # probs = [1/len(ACTION_ID)] * len(ACTION_ID)
    # mcts_params = MCTSParams()
    # const_policy = ConstantPolicy(probs, ACTION_ID)
    # model = load_mlp_model('./data/models/mlp.pth')
    # state_encoder = SplendorGameStateEncoder(2)
    # nn_policy = NNPolicy(model, state_encoder, ACTION_ID)
    # value_fun = AccumValue()

    game_state = traj.initial_state.copy()
    for n, action in enumerate(traj.actions):
        print(game_state)
        print(f'active_player: {game_state.active_player()} action: {action}')
        if traj.freqs:
            if game_state.active_player() != CHANCE_PLAYER:
                recorded_visits = traj.freqs[n]
                print(f'recorded: {recorded_visits}')

                # mcts = PolicyMCTS(game_state, const_policy, mcts_params)
                # mcts.search()
                # mcts_vistis = {str(action): count for action, count in  mcts.root_visits()}

                # pv_mcts = PVMCTS(game_state, const_policy, value_fun, mcts_params)
                # pv_mcts.search()
                # pv_mcts_vistis = {str(action): count for action, count in  pv_mcts.root_visits()}

                # probs = nn_policy.predict(game_state)
                # visits_str = ''.join(sorted([f'{a}: {recorded_visits[a]}, {mcts_vistis[a]}, {pv_mcts_vistis[a]}, {probs[n]:.4f}\t' for n, a in enumerate(mcts_vistis.keys())]))
                # print(f'recorded vs mcts: {visits_str}')
        print()
        game_state.apply_action(action)
    
    print(game_state)
    for id, r in enumerate(game_state.rewards()):
        print(f'player{id}: {r}')
    
def run_tournament():
    random.seed(11)

    model = load_mlp_model('./data/models/mlp_10k_bw.pth')
    state_encoder = SplendorGameStateEncoder(2)
    policy = NNPolicy(model, state_encoder, ACTION_ID)

    mcts_params = MCTSParams(iterations= 1000)
    nn_agent = PolicyMCTSAgent(policy, mcts_params)
    mcts_agent = MCTSAgent(mcts_params)

    agents = [nn_agent, mcts_agent]
    # game_state = SplendorGameState(len(agents))
    # game_state = SplendorGameState.from_json(json.load(open('/Users/seal/Downloads/state.json', 'rt')))
    # traj = run_one_game(game_state, agents, verbose=True)
    # tournament(agents, num_games=100, verbose=False)
    tournament_parallel(agents, num_games=100, num_workers=10)

if __name__ == '__main__':
    # run_tournament()

    tloader = traj_loader('data/traj_dump.txt')
    for _ in range(2):
        traj = next(tloader)
    print_game_record(traj)