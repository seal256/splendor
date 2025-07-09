import os, json
from copy import deepcopy
from datetime import datetime
import subprocess

from pysplendor.game import traj_loader
from train import train
from prepare_data import prepare_data

POLICY_AGENT = {
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

MCTS_AGENT = {
            "type": "MCTSAgent",
            "iterations": 500,
            "exploration": 1.4,
            "max_choice_children": 1,
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
    print(f'Running {name_suffix} games step {step}')

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

def self_play_loop():
    # best_model = '/Users/seal/projects/splendor/data/models/random_2_512.pt'
    best_model = f'{WORK_DIR}/model_step_9_best.pt'
    start_step = 10

    games_per_update = 5000
    val_fraction = 0.1
    training_iterations = 10 # training_iterations * games_per_update trajectories will be used for each train iteration
    max_iterations = 100
    train_epochs = 1
    new_model_eval_games = 1000
    min_win_rate = 0.54
    max_iters_without_improvement = 12

    train_dirs = [f'{WORK_DIR}/train_step_{step}' for step in range(start_step)]
    val_dirs = [f'{WORK_DIR}/val_step_{step}' for step in range(start_step)]
    iters_without_improvement = 0

    for step in range(start_step, max_iterations):
        curr_time = datetime.now().strftime("%d.%m %H:%M:%S")
        print(f'\n\n### {curr_time} Iteration {step} ###\n')

        # run self play sessions with previous best model
        print(f'Collecting new self play games using {best_model}')
        val_traj = run_games('val', step, best_model, best_model, games_per_update * val_fraction, train=False, rotate_agents=False)
        train_traj = run_games('train', step, best_model, best_model, games_per_update, train=True, rotate_agents=False)

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
        new_vs_best_traj = run_games('new_vs_best', step, new_model, best_model, new_model_eval_games, train=False, rotate_agents=True)
        new_model_win_rate = agent_score("a", new_vs_best_traj)
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

from math import log10
def win_rate_to_elo_rating(win_rate, ref_rating = 0):
    rating = ref_rating - 400 * log10(1 / win_rate - 1)
    return rating



from dataclasses import dataclass
import re
from typing import List

@dataclass
class JobReport:
    step: int
    player_a_score: float
    player_a_conf: float
    player_a_cards_mean: float
    player_a_cards_std: float
    game_length_avg: float
    game_length_std: float

def parse_job_report(file_path: str) -> List[JobReport]:
    with open(file_path, 'r') as file:
        content = file.read()
    
    reports = re.split(r'(?=\nseed:)', content.strip())
    results = []
    
    for report in reports:
        if not report.strip():
            continue
            
        # Extract player a stats
        player_a_match = re.search(
            r'player a: total score: [\d.]+,\s*mean score: ([\d.]+),\s*score conf interval: ([\d.]+),\s*cards mean: ([\d.]+),\s*cards std dev: ([\d.]+)',
            report
        )
        # Extract game length stats
        game_length_match = re.search(
            r'game length avg: ([\d.]+),\s*std dev: ([\d.]+)',
            report
        )
        # Extract step number
        step_match = re.search(
            r'traj_trained_vs_mcts_step_(\d+)\.txt',
            report
        )
        
        if all([player_a_match, game_length_match, step_match]):
            results.append(JobReport(
                step=int(step_match.group(1)),
                player_a_score=float(player_a_match.group(1)),
                player_a_conf=float(player_a_match.group(2)),
                player_a_cards_mean=float(player_a_match.group(3)),
                player_a_cards_std=float(player_a_match.group(4)),
                game_length_avg=float(game_length_match.group(1)),
                game_length_std=float(game_length_match.group(2))
            ))
    
    return results

import matplotlib.pyplot as plt
def plot_winrates():
    # random_vs_mcts_win_rate = 0.014 # Win rate of random player vs MCTS
    # base = win_rate_to_elo_rating(0.014) # Assuming random player has Elo rating 0

    resutls = parse_job_report('rate_report.txt')
    steps = [d.step for d in resutls]
    win_rates = [d.player_a_score for d in resutls]
    elos = [win_rate_to_elo_rating(w) for w in win_rates]
    err_bars = [d.player_a_conf for d in resutls]
    erros_elo_up = [win_rate_to_elo_rating(w + e) - win_rate_to_elo_rating(w) for w, e in zip(win_rates, err_bars)]
    erros_elo_down = [-win_rate_to_elo_rating(w - e) + win_rate_to_elo_rating(w) for w, e in zip(win_rates, err_bars)]
    # plt.plot(steps, win_rates, '-b')
    plt.errorbar(steps, elos, yerr=[erros_elo_down, erros_elo_up], fmt='-o', label='MCTS + action selection policy')
    plt.plot([steps[0], steps[-1]], [0, 0], '-k', label='MCTS')

    plt.xlabel('training step')
    # plt.ylabel('win rate')
    plt.ylabel('Elo rating wrt MCTS')
    plt.title('win points 5')
    plt.legend()
    plt.show()

def plot_cards():
    resutls = parse_job_report('rate_report.txt')
    steps = [d.step for d in resutls]
    cards = [d.player_a_cards_mean for d in resutls]
    err_bars = [d.player_a_cards_std for d in resutls]
    plt.errorbar(steps, cards, yerr=err_bars, fmt='-o', label='Purchased cards')
    
    plt.xlabel('training step')
    plt.ylabel('Cards')
    plt.title('win points 5')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    WORK_DIR = '/Users/seal/projects/splendor/data_0207_wp5'
    # os.mkdir(WORK_DIR)
    # self_play_loop()
    # rate_against_baseline()
    # plot_winrates()
    # plot_cards()

    random_vs_mcts_win_rate = 0.014 # Win rate of random player vs MCTS
    base = win_rate_to_elo_rating(0.014) # Assuming random player has Elo rating 0
    print(base)

    # best_model = '/Users/seal/projects/splendor/data_1405/model_wp3_best.pt'
    # run_games('val', 0, best_model, best_model, 100, train=False)




