import matplotlib.pyplot as plt
import numpy as np

def plot_game_length():
    # Plots game lengths distribution (2 player games) according to Spendee analysis:
    # https://spendee.mattle.online/lobby/forum/topic/mzXQmzjCBmyC56Dgx/splendor-strategy-data-analysis-part-1

    turns = list(range(20, 37))
    players = [13, 31, 112, 239, 489, 831, 1199, 1265, 1109, 878, 526, 303, 142, 68, 35, 15, 13]

def plot_purchased_cards():
    # Plots purchased cards distribution (2 player games) according to Spendee analysis:
    # https://spendee.mattle.online/lobby/forum/topic/mzXQmzjCBmyC56Dgx/splendor-strategy-data-analysis-part-1

    ratings = [1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    cards = [15.6, 14.9, 13.9, 12.1, 11.4, 11.7, 11.0, 9.3]

    coeff = np.polyfit(cards, ratings, 1)
    regression_line = np.poly1d(coeff)
    equation = f"R = {coeff[0]:.2f}C + {coeff[1]:.2f}"
    print(equation)

    plt.plot(cards, ratings, 'ob', label="Spendee server data")
    plt.plot(cards, regression_line(cards), '-r', label=equation)
    plt.ylabel('human Elo rating (R)')
    plt.xlabel('purchased cards (C)')

    # plt.plot(ratings, cards, 'ob', label="Spendee server data")
    # plt.xlabel('human Elo rating (R)')
    # plt.ylabel('purchased cards (C)')
    
    plt.legend()
    plt.show()
    


def human_ratings_distribution():
    # Read numbers from the text file
    with open('./assets/human_ratings_online.txt', 'r') as file:
        ratings = [float(line.strip()) for line in file if line.strip()]

    avg_rating = sum(ratings) / len(ratings)
    print(f'Average rating: {avg_rating:.0f}')

    plt.hist(ratings, bins=30)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == '__main__':
    human_ratings_distribution()
    # plot_purchased_cards()