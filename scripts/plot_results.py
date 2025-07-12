
import matplotlib.pyplot as plt
from math import log10

def win_rate_to_elo_rating(win_rate, ref_rating = 0):
    rating = ref_rating - 400 * log10(1 / win_rate - 1)
    return rating


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

    plot_winrates()
    plot_cards()

    # random_vs_mcts_win_rate = 0.014 # Win rate of random player vs MCTS
    # base = win_rate_to_elo_rating(0.014) # Assuming random player has Elo rating 0
    # print(base)