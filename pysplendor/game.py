from .splendor import SplendorGameState, SplendorGameRules, Action
from .agents import Agent, HumanPlayer, RandomAgent, MCTSAgent
from .game_state import GameState

def run_one_game(game_state: GameState, agents: list[Agent]):
    active_player = game_state.active_player()
    while not game_state.is_terminal():
        if active_player is None: # chance game state
            action = game_state.get_actions()[0]
        else:
            action = agents[active_player].get_action(game_state)

        print(game_state)
        print(f'selected action: {action}\n')

        game_state.apply_action(action)
        active_player = game_state.active_player()

    print('Final scores:')
    for n, score in enumerate(game_state.rewards()):
        print(f'player {n} score: {score}')

def one_round():
    game_state = SplendorGameState(num_players = 2)
    agent = MCTSAgent(iterations=500)
    agent.get_action(game_state)

def profile():
    import cProfile
    cProfile.run('one_round()')

class Trajectory():
    def __init__(self, initial_state, actions, rewards):
        self.initial_state = initial_state
        self.actions = actions
        self.rewards = rewards

    @classmethod
    def from_json(cls, data):
        initial_state = SplendorGameState.from_json(data['initial_state'])
        actions = [Action.from_str(a) for a in data['actions']]
        rewards = data['rewards']
        return cls(initial_state, actions, rewards)


if __name__ == '__main__':
    agents = [RandomAgent(), MCTSAgent(iterations=5000)]
    names = [f'player{n}' for n in range(len(agents))]
    game_state = SplendorGameState(names, SplendorGameRules(len(agents)))
    run_one_game(game_state, agents)

    # one_round()
    # profile()