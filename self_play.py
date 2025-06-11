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

def run_games(name_suffix, step, model_a_path, model_b_path, num_games=1000, train=False):
    print(f'Running {name_suffix} games step {step}')

    traj_path = f'{WORK_DIR}/traj_{name_suffix}_step_{step}.txt'
    config = game_config(model_a_path, model_b_path, traj_path, num_games, train)
    config_path = f'{WORK_DIR}/{name_suffix}_step_{step}.json'
    json.dump(config, open(config_path, 'wt'))
    subprocess.run([BINARY_PATH, config_path], check=True)

    return traj_path # path to resulting trajectories

def first_agent_score(traj_path):
    '''Returns win rate of the first agent'''
    tloader = traj_loader(traj_path)
    first_player_score = 0
    total_score = 0
    for traj in tloader:
        first_player_score += traj.rewards[0]
        total_score += sum(traj.rewards)
    return first_player_score / total_score
    

def self_play_loop():
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

if __name__ == '__main__':
    WORK_DIR = '/Users/seal/projects/splendor/data_1405'
    # os.mkdir(WORK_DIR)
    self_play_loop()
    # best_model = '/Users/seal/projects/splendor/data_1405/model_wp3_best.pt'
    # run_games('val', 0, best_model, best_model, 100, train=False)




