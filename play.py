import argparse

from pysplendor.agents import RandomAgent, MCTSAgent, HumanPlayer, load_mlp_model, NNPolicy, PolicyMCTSAgent
from pysplendor.splendor import SplendorGameState, SplendorGameRules
from pysplendor.game import run_one_game, Trajectory
from pysplendor.mcts import MCTSParams
from prepare_features import SplendorGameStateEncoder

def human_vs_policy_mcts(model_path, iterations=500, win_points=15):
    '''Plays one game between a human player and an MCTS agent guided by a neural network policy.'''

    mcts_params = MCTSParams(iterations=iterations)
    model = load_mlp_model(model_path)
    state_encoder = SplendorGameStateEncoder(2)
    policy = NNPolicy(model, state_encoder)
    mcts_agent = PolicyMCTSAgent(policy, mcts_params)

    human_agent = HumanPlayer()
    agents = [mcts_agent, human_agent]

    rules = SplendorGameRules(len(agents))
    rules.win_points = win_points
    game_state = SplendorGameState(len(agents), rules)
    run_one_game(game_state, agents, verbose=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human vs Policy MCTS for Splendor')
    parser.add_argument('-m', '--model_path', type=str, default='/Users/seal/projects/splendor/data_0207_wp5/model_step_69_best.pt', help='Path to the trained model')
    parser.add_argument('-i', '--iterations', type=int, default=500, help='Number of MCTS iterations per move')
    parser.add_argument('-w', '--win_points', type=int, default=5, help='Points needed to win the game')
    
    args = parser.parse_args()
    
    human_vs_policy_mcts(
        model_path=args.model_path,
        iterations=args.iterations,
        win_points=args.win_points
    )
