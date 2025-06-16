import os, json
from copy import deepcopy
import subprocess

from pysplendor.game import traj_loader
from train import train
from prepare_data import prepare_data

AGENT = {
            "type": "PolicyMCTSAgent",
            "iterations": 500,
            "exploration": 1.4,
            "max_choice_children": 1,
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

CONFIG = {
    "agents": [],
    "num_games": 10000,
    "num_workers": 9,
    "verbose": False,
    "save_freqs": True,
    "win_points": 3,
    "dump_trajectories": ""
}

BINARY_PATH = "./splendor"

def game_config(model_a_path, model_b_path, traj_path, num_games=1000, train=False):
    agent_a = deepcopy(AGENT)
    agent_a["policy"]["model_path"] = model_a_path
    agent_a["train"] = train
    agent_b = deepcopy(AGENT)
    agent_b["policy"]["model_path"] = model_b_path
    agent_b["train"] = train

    config = deepcopy(CONFIG)
    config["agents"] = [agent_a, agent_b]
    config["num_games"] = num_games
    config["dump_trajectories"] = traj_path
    return config

def run_games(name_suffix, step, model_a_path, model_b_path, num_games=1000, train=False, rotate_players=False):
    print(f'Running {name_suffix} games step {step}')

    traj_path = f'{WORK_DIR}/traj_{name_suffix}_step_{step}.txt'
    config = game_config(model_a_path, model_b_path, traj_path, num_games, train)
    config_path = f'{WORK_DIR}/{name_suffix}_step_{step}.json'
    json.dump(config, open(config_path, 'wt'))
    subprocess.run([BINARY_PATH, config_path], check=True)

    return traj_path # path to resulting trajectories

def first_agent_score(traj_path):
    '''Returns the win rate of the first agent'''
    tloader = traj_loader(traj_path)
    first_player_score = 0
    total_score = 0
    for traj in tloader:
        first_player_score += traj.rewards[0]
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
        new_model_win_rate = first_agent_score(new_vs_best_traj)
        print(f'New model win rate vs previous best model: {new_model_win_rate:.3f}')
        if (new_model_win_rate > 0.55):
            best_model = new_model
            print(f'New best model is: {best_model}')
        
        else:
            print('Stopping')
            break

def self_play_loop():
    # best_model = '/Users/seal/projects/splendor/data/models/random_2_512.pt'
    best_model = '/Users/seal/projects/splendor/data_1106/model_step_96_best.pt'
    start_step = 100

    games_per_update = 5000
    val_fraction = 0.1
    training_iterations = 10 # training_iterations * games_per_update trajectories will be used for each train iteration
    max_iterations = 200
    train_epochs = 1
    new_model_eval_games = 1000
    min_win_rate = 0.55
    max_iters_without_improvement = 10

    train_dirs = [f'{WORK_DIR}/train_step_{step}' for step in range(start_step)]
    val_dirs = [f'{WORK_DIR}/val_step_{step}' for step in range(start_step)]
    iters_without_improvement = 0

    for step in range(start_step, max_iterations):
        print(f'\n\n---- Iteration {step} ----\n')

        # run self play sessions with previous best model
        print(f'Collecting {games_per_update} new self play games')
        val_traj = run_games('val', step, best_model, best_model, games_per_update * val_fraction, train=False)
        train_traj = run_games('train', step, best_model, best_model, games_per_update, train=True)

        # preprocess data for model training
        train_dir = f'{WORK_DIR}/train_step_{step}'
        val_dir = f'{WORK_DIR}/val_step_{step}'
        prepare_data(train_traj, train_dir)
        prepare_data(val_traj, val_dir)
        train_dirs.append(train_dir)
        val_dirs.append(val_dir)

        # train new model     
        if len(train_dirs) < training_iterations:
            continue
        print('Training new model')
        model_name_prefix = f'{WORK_DIR}/model_step_{step}'
        train(model_name_prefix, train_dirs[-training_iterations:], val_dirs[-training_iterations:], train_epochs, best_model)
        new_model = model_name_prefix + '_best.pt'

        # evaluate new model against the previous best one
        new_vs_best_traj = run_games('new_vs_best', step, new_model, best_model, new_model_eval_games, train=False)
        new_model_win_rate = first_agent_score(new_vs_best_traj)
        print(f'New model win rate vs previous best model: {new_model_win_rate:.3f}')
        if (new_model_win_rate > min_win_rate):
            best_model = new_model
            print(f'New best model is: {best_model}')
            iters_without_improvement = 0
        
        else:
            iters_without_improvement += 1
            print(f'Iterations without improvement: {iters_without_improvement}')
            if iters_without_improvement >= max_iters_without_improvement:
                print('Stopping')
                break


if __name__ == '__main__':
    WORK_DIR = '/Users/seal/projects/splendor/data_1106'
    # os.mkdir(WORK_DIR)
    self_play_loop()
    # best_model = '/Users/seal/projects/splendor/data_1405/model_wp3_best.pt'
    # run_games('val', 0, best_model, best_model, 100, train=False)




