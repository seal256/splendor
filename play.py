from splendor_game import SplendorGameState, SplendorGameRules, Action
from agents import Agent, HumanPlayer, CultivatorPlayer, RandomAgent, MCTSAgent
from game import GameState

def game_round(game_state: GameState, agents: list[Agent]):
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
        print(f'player {n}: score: {score}')

if __name__ == '__main__':
    agents = [RandomAgent(), MCTSAgent()]
    names = [f'player{n}' for n in range(len(agents))]
    game_state = SplendorGameState(names, SplendorGameRules(len(agents)))
    game_round(game_state, agents)