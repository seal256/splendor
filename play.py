from pysplendor.agents import RandomAgent, MCTSAgent
from pysplendor.splendor import SplendorGameState
from pysplendor.game import run_one_game

if __name__ == '__main__':
    agents = [RandomAgent(), MCTSAgent(iterations=1000)]
    game_state = SplendorGameState(len(agents))
    traj = run_one_game(game_state, agents, verbose=True)
