import subprocess, json
import argparse, os
from dataclasses import asdict

from scripts.splendor_job import MCTSAgentConfig, PolicyMCTSAgentConfig, NNPolicyConfig, GameConfig, extract_stats
from pysplendor.game import traj_loader

BINARY_PATH = "./splendor"

def run_games_against_mcts(work_dir, name_suffix, step, num_games=1000, train=False, rotate_agents=True, overwrite=False):
    print(f'Running {name_suffix} games step {step}')

    traj_path = f'{work_dir}/traj_{name_suffix}_step_{step}.txt'
    if os.path.exists(traj_path):
        if overwrite:
            print(f'Overwriting trajectories: {traj_path}')
        else:
            print(f'Trajectories exist: {traj_path}')
            return traj_path
    
    model_name = f'{work_dir}/model_step_{step}.pt'
    agent_a = PolicyMCTSAgentConfig("a", train, NNPolicyConfig(model_name))
    agent_b = MCTSAgentConfig("b", train)
    config = GameConfig(agents=[agent_a, agent_b], num_games=num_games, rotate_agents=rotate_agents, dump_trajectories=traj_path)

    config_path = f'{work_dir}/{name_suffix}_step_{step}.json'
    json.dump(asdict(config), open(config_path, 'wt'))
    subprocess.run([BINARY_PATH, config_path], check=True)

    print(f'Created {traj_path}')
    return traj_path

def rate_against_mcts(work_dir, step_range, name_suffix='model_vs_mcts', overwrite=False):
    '''Runs series of games against a fixed MCTS baseline for a range of models'''

    for step in step_range:
        traj_file = run_games_against_mcts(work_dir, name_suffix, step, overwrite=overwrite)
        traj = list(traj_loader(traj_file))
        stats = extract_stats(traj)
        stat_file = f'{work_dir}/stats_{name_suffix}_step_{step}.json'
        json.dump(asdict(stats), open(stat_file, 'wt'))
        print(f'Created {stat_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rates a series of models (indexed by train steps) against MCTS baseline. Creates a file with JSON results for each step')
    parser.add_argument('--work_dir', type=str, required=True, help='Path to the working directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrites existing trajectory files in the working directory. By default only missing games are executed')
    parser.add_argument('--name_suffix', type=str, required=False, default='model_vs_mcts', help='Suffix for the trajctory file names in the working directory')
    parser.add_argument('--model', type=str, required=False, default=None, help='Path to the model to rate')
    parser.add_argument('--start_step', type=int, required=False, default=10, help='start step')
    parser.add_argument('--end_step', type=int, required=False, default=80, help='end step')
    parser.add_argument('--step_incr', type=int, required=False, default=1, help='step increment')
    args = parser.parse_args()

    step_range = list(range(args.start_step, args.end_step, args.step_incr))
    rate_against_mcts(args.work_dir, step_range, args.name_suffix, args.overwrite)


