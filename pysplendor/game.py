import json, random

from .splendor import SplendorGameState, Action, CHANCE_PLAYER


class Trajectory():
    def __init__(self, initial_state, actions, rewards, states=[]):
        self.initial_state = initial_state
        self.actions = actions # serialized to strings
        self.rewards = rewards
        self.states = states # for debug only, usually empty

    @classmethod
    def from_json(cls, data):
        initial_state = SplendorGameState.from_json(data['initial_state'])
        actions = [Action.from_str(a) for a in data['actions']]
        rewards = data['rewards']
        states = [SplendorGameState.from_json(s) for s in data['states']] if 'states' in data else []
        return cls(initial_state, actions, rewards, states)
    

def traj_loader(file_name):
    with open(file_name, 'rt') as fin:
        for line in fin:
            data = json.loads(line)
            traj = Trajectory.from_json(data)
            yield traj


def run_one_game(game_state, agents, verbose=False, save_states=False):
    trajectory = Trajectory(game_state.copy(), [], [])

    active_player = game_state.active_player()
    while not game_state.is_terminal():
        if active_player == CHANCE_PLAYER:
            legal_actions = game_state.get_actions()
            idx = random.randint(0, len(legal_actions) - 1)
            action = legal_actions[idx]
        else:
            action = agents[active_player].get_action(game_state)

        if verbose:
            print(f"\n{game_state}\n")
            print(f"selected action: {action}\n")

        trajectory.actions.append(action)
        game_state.apply_action(action)
        if save_states: # for debug only
            trajectory.states.append(game_state.copy())
        
        active_player = game_state.active_player()

    rewards = game_state.rewards()
    trajectory.rewards = rewards

    if verbose:
        print(game_state)
        print("Final scores:")
        for n in range(len(rewards)):
            print(f"player {n} score: {rewards[n]}")

    return trajectory

