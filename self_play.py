import os, json
from copy import deepcopy
from datetime import datetime
import subprocess
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from pysplendor.game import traj_loader
from train import train, create_random_model, TrainConfig
from prepare_data import prepare_data

POLICY_AGENT = {
            "type": "PolicyMCTSAgent",
            "iterations": 500,
            "exploration": 1.4,
            "max_chance_children": 1,
            "weighted_selection_moves":20,
            "p_noise_level":0.0,
            "alpha":1.0, # creates a uniform distribution
            "train":False,
            "policy" : {
                "type" : "NNPolicy",
                "model_path": "",
                "num_players": 2
            }
        }

MCTS_AGENT = {
            "type": "MCTSAgent",
            "iterations": 500,
            "exploration": 1.4,
            "max_chance_children": 1,
            "train":False,
        }


CONFIG = {
    "agents": [],
    "num_games": 10000,
    "num_workers": 9,
    "verbose": False,
    "save_freqs": True,
    "win_points": 5,
    "rotate_agents": False,
    "dump_trajectories": ""
}

BINARY_PATH = "./splendor"

def model_agent(model_path, name, train=False):
    agent = deepcopy(POLICY_AGENT)
    agent["policy"]["model_path"] = model_path
    agent["train"] = train
    agent["name"] = name    
    return agent

def mcts_agent(name, train=False):
    agent = deepcopy(MCTS_AGENT)
    agent["train"] = train
    agent["name"] = name    
    return agent

def game_config(model_a_path, model_b_path, traj_path, num_games=1000, train=False, rotate_agents=False):
    agent_a = model_agent(model_a_path, "a", train)
    agent_b = model_agent(model_b_path, "b", train)

    config = deepcopy(CONFIG)
    config["agents"] = [agent_a, agent_b]
    config["num_games"] = num_games
    config["rotate_agents"] = rotate_agents
    config["dump_trajectories"] = traj_path
    return config

def game_against_mcts_config(model_a_path, traj_path, num_games=1000, train=False, rotate_agents=False):
    agent_a = model_agent(model_a_path, "a", train)
    agent_b = mcts_agent("b", train)

    config = deepcopy(CONFIG)
    config["agents"] = [agent_a, agent_b]
    config["num_games"] = num_games
    config["rotate_agents"] = rotate_agents
    config["dump_trajectories"] = traj_path
    return config

def run_games(name_suffix, step, model_a_path, model_b_path, num_games=1000, train=False, rotate_agents=False):
    print(f'Running {name_suffix} games step {step}', flush=True)

    traj_path = f'{WORK_DIR}/traj_{name_suffix}_step_{step}.txt'
    config = game_config(model_a_path, model_b_path, traj_path, num_games, train, rotate_agents)
    config_path = f'{WORK_DIR}/{name_suffix}_step_{step}.json'
    json.dump(config, open(config_path, 'wt'))
    subprocess.run([BINARY_PATH, config_path], check=True)

    return traj_path # path to resulting trajectories

def run_games_against_mcts(name_suffix, step, model_a_path, num_games=1000, train=False, rotate_agents=False):
    print(f'Running {name_suffix} games step {step}')

    traj_path = f'{WORK_DIR}/traj_{name_suffix}_step_{step}.txt'
    config = game_against_mcts_config(model_a_path, traj_path, num_games, train, rotate_agents)
    config_path = f'{WORK_DIR}/{name_suffix}_step_{step}.json'
    json.dump(config, open(config_path, 'wt'))
    subprocess.run([BINARY_PATH, config_path], check=True)

    return traj_path # path to resulting trajectories

def agent_score(agent_name, traj_path):
    '''Returns the win rate of the first agent'''
    tloader = traj_loader(traj_path)
    first_player_score = 0
    total_score = 0
    for traj in tloader:
        agent_idx = traj.agent_names.index(agent_name)
        first_player_score += traj.rewards[agent_idx]
        total_score += sum(traj.rewards)
    return first_player_score / total_score
    

def self_play_steps():
    '''Only data from a previous iteration is used for self play and training of the next model'''

    best_model = '/Users/seal/projects/splendor/data_1405/model_wp3_best.pt'
    for step in range(5):
        print(f'\n\n---- Global step {step} ----\n')

        # run self play sessions with previous best model
        val_traj = run_games('val', step, best_model, best_model, 10000, train=False)
        train_traj = run_games('train', step, best_model, best_model, 50000, train=True)

        # preprocess data for model training
        train_dir = f'{WORK_DIR}/train_step_{step}'
        val_dir = f'{WORK_DIR}/val_step_{step}'
        prepare_data(train_traj, train_dir)
        prepare_data(val_traj, val_dir)

        # train new model        
        model_name_prefix = f'{WORK_DIR}/model_step_{step}'
        train(model_name_prefix, train_dir, val_dir)
        new_model = model_name_prefix + '_best.pt'

        # evaluate new model against the previous best one
        new_vs_best_traj = run_games('new_vs_best', step, new_model, best_model, 1000, train=False)
        new_model_win_rate = agent_score("a", new_vs_best_traj)
        print(f'New model win rate vs previous best model: {new_model_win_rate:.3f}')
        if (new_model_win_rate > 0.55):
            best_model = new_model
            print(f'New best model is: {best_model}')
        
        else:
            print('Stopping')
            break

@dataclass
class SelfPlayConfig:
    """Self-play training configuration"""

    model: str = None                       # Path to initial model (if None a new random model will be used)
    start_step: int = 0                     # Restarts process from the middle if needed
    train_games: int = 5000                 # New train games generated per iteration
    val_games: int = 500                    # New validation games generated per iteration (used for model overfitting control)
    train_buffer_size: int = 10             # Number of past iterations kept in training buffer
    train_epochs: int = 1                   # Model training epochs per iteration
    new_model_eval_games: int = 1000        # Evaluation games against current best model
    min_win_rate: float = 0.54              # Min win rate to adopt new model (0-1)
    max_iterations: int = 100               # Max total training iterations
    max_iters_without_improvement: int = 12 # Early stopping condition: Stop after this many non-improving iterations
    work_dir: str = "data"                  # Directory for output files
    train: TrainConfig = None               # 


def self_play_loop(config: SelfPlayConfig):
    print('Self play configuration:')
    print(json.dumps(asdict(config), indent=2))

    os.makedirs(config.work_dir, exist_ok=True)
    best_model = config.model 
    if not best_model:
        best_model = f'{config.work_dir}/random.pt'
        create_random_model(best_model)

    train_dirs = [f'{config.work_dir}/train_step_{step}' for step in range(config.start_step)]
    val_dirs = [f'{config.work_dir}/val_step_{step}' for step in range(config.start_step)]
    iters_without_improvement = 0

    for step in range(config.start_step, config.max_iterations):
        curr_time = datetime.now().strftime("%d.%m %H:%M:%S")
        print(f'\n\n### {curr_time} Iteration {step} ###\n', flush=True)

        # run self play sessions with previous best model
        print(f'Collecting new self play games using {best_model}', flush=True)
        val_traj = run_games('val', step, best_model, best_model, config.val_games, train=False, rotate_agents=False)
        train_traj = run_games('train', step, best_model, best_model, config.train_games, train=True, rotate_agents=False)

        # preprocess data for model training
        train_dir = f'{config.work_dir}/train_step_{step}'
        val_dir = f'{config.work_dir}/val_step_{step}'
        prepare_data(train_traj, train_dir)
        prepare_data(val_traj, val_dir)
        train_dirs.append(train_dir)
        val_dirs.append(val_dir)

        # train new model     
        if len(train_dirs) < config.train_buffer_size:
            continue
        print('Training new model', flush=True)
        train_config = deepcopy(config.train)
        train_config.train_dir = train_dirs[-config.train_buffer_size:]
        train_config.val_dir = val_dirs[-config.train_buffer_size:]
        train_config.model = best_model
        train_config.result_model_name = f'{config.work_dir}/model_step_{step}'
        train(train_config)
        new_model = model_name_prefix + '_best.pt'

        # evaluate new model against the previous best one
        new_vs_best_traj = run_games('new_vs_best', step, new_model, best_model, config.new_model_eval_games, train=False, rotate_agents=True)
        new_model_win_rate = agent_score("a", new_vs_best_traj)
        print(f'New model win rate vs previous best model: {new_model_win_rate:.3f}', flush=True)
        if (new_model_win_rate > config.min_win_rate):
            best_model = new_model
            print(f'New best model is: {best_model}', flush=True)
            iters_without_improvement = 0
        
        else:
            iters_without_improvement += 1
            print(f'Iterations without improvement: {iters_without_improvement}', flush=True)
            if iters_without_improvement >= config.max_iters_without_improvement:
                print('Stopping', flush=True)
                break

def rate_against_baseline():
    steps = []
    win_rates = []
    for step in range(21, 80):
        model_name = f'{WORK_DIR}/model_step_{step}_best.pt'
        trained_vs_mcts_traj = run_games_against_mcts('trained_vs_mcts', step, model_name, 1000, train=False, rotate_agents=True)
        model_win_rate = agent_score("a", trained_vs_mcts_traj)
        # print(step, model_win_rate)

        steps.append(step)
        win_rates.append(model_win_rate)

    print(steps)
    print(win_rates)


def load_config_from_json(file_path) -> SelfPlayConfig:
    with open(file_path) as f:
        data = json.load(f)
    return SelfPlayConfig(**data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-play configuration")
    parser.add_argument("-c", "--config-file", required=True, help="JSON configuration file")
    
    args = parser.parse_args()
    with open(args.c) as f:
        data = json.load(f)
    config = SelfPlayConfig(**data)

    self_play_loop(config)
    




