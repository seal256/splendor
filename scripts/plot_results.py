import matplotlib.pyplot as plt
from math import log10
from typing import List
import json

from splendor_job import GameStats

def win_rate_to_elo_rating(win_rate, ref_rating = 0):
    rating = ref_rating - 400 * log10(1 / win_rate - 1)
    return rating

def plot_winrates(steps: List[int], stats: List[GameStats], player_idx = 0):
    # random_vs_mcts_win_rate = 0.014 # Win rate of random player vs MCTS
    # base = win_rate_to_elo_rating(0.014) # Assuming random player has Elo rating 0

    win_rates = [d.players[player_idx].win_rate for d in stats]
    elos = [win_rate_to_elo_rating(w) for w in win_rates]
    err_bars = [d.players[player_idx].confidence_interval for d in stats]
    erros_elo_up = [win_rate_to_elo_rating(w + e) - win_rate_to_elo_rating(w) for w, e in zip(win_rates, err_bars)]
    erros_elo_down = [-win_rate_to_elo_rating(w - e) + win_rate_to_elo_rating(w) for w, e in zip(win_rates, err_bars)]

    plt.figure()
    plt.errorbar(steps, elos, yerr=[erros_elo_down, erros_elo_up], fmt='-o', label='MCTS + action selection policy')
    plt.plot([steps[0], steps[-1]], [0, 0], '-k', label='MCTS')

    plt.xlabel('training step')
    plt.ylabel('Elo rating wrt MCTS')
    plt.title('win points 5')
    plt.legend()
    # plt.show()

def plot_cards(steps: List[int], stats: List[GameStats], player_idx = 0):
    cards = [d.players[player_idx].cards_mean for d in stats]
    err_bars = [d.players[player_idx].cards_std for d in stats]

    plt.figure()
    plt.errorbar(steps, cards, yerr=err_bars, fmt='-o', label='Purchased cards')
    
    plt.xlabel('training step')
    plt.ylabel('Cards')
    plt.title('win points 5')
    plt.legend()
    # plt.show()

def plot_game_len(steps: List[int], stats: List[GameStats], player_idx = 0):
    round_lens = [d.game_length_mean for d in stats]
    err_bars = [d.game_length_std for d in stats]

    plt.figure()
    plt.errorbar(steps, round_lens, yerr=err_bars, fmt='-o', label='Game length')
    
    plt.xlabel('training step')
    plt.ylabel('Game length')
    plt.title('win points 5')
    plt.legend()

def plot_cards_and_len(steps: List[int], stats: List[GameStats], player_idx=0):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cards 
    cards = [d.players[player_idx].cards_mean for d in stats]
    cards_err = [d.players[player_idx].cards_std for d in stats]
    ax1.errorbar(steps, cards, yerr=cards_err, fmt='-o', label='Purchased cards')
    ax1.set_xlabel('training step')
    ax1.set_ylabel('Cards')
    ax1.set_title('Purchased Cards')
    ax1.legend()
    
    # Plot game length
    round_lens = [d.game_length_mean for d in stats]
    len_err = [d.game_length_std for d in stats]
    ax2.errorbar(steps, round_lens, yerr=len_err, fmt='-o', label='Game length')
    ax2.set_xlabel('training step')
    ax2.set_ylabel('Game length')
    ax2.set_title('Game Length')
    ax2.legend()
    
    plt.tight_layout()

def load_stats(work_dir, step_range, name_suffix):
    stats = []
    for step in step_range:
        stat_file = f'{work_dir}/stats_{name_suffix}_step_{step}.json'
        stats.append(GameStats.from_json(json.load(open(stat_file))))
    return stats

def rates(win_rate, err):
    elo = win_rate_to_elo_rating(win_rate)
    elo_conf = (win_rate_to_elo_rating(win_rate + err) - win_rate_to_elo_rating(win_rate - err)) / 2
    print(f'elo = {elo:.1f} ({elo_conf:.1f})')

if __name__ == '__main__':
    # rates(0.99, 0.0039)

    work_dir = './data_0207_wp5'
    name_suffix = 'trained_vs_mcts'
    step_range = list(range(10, 80))

    stats = load_stats(work_dir, step_range, name_suffix)

    plot_winrates(step_range, stats)
    # plot_cards(step_range, stats)
    # plot_game_len(step_range, stats)
    plot_cards_and_len(step_range, stats)

    plt.show()