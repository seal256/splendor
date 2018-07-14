from game import SplendorGameState, SplendorGameRules

def human_game():
    player_names = ['player1', 'player2']
    rules = SplendorGameRules(len(player_names))
    game = SplendorGameState(player_names, rules)

    while not game.check_win():
        for player_name in player_names:
            print(game)
            while True: # check for invalid action inputs
                action_str = input(player_name + ' move: ')
                try:
                    game.action(action_str)
                    break
                except AttributeError as err:
                    print('Invalid action {}: {}'.format(action_str, str(err)))

    print('best player:', game.best_player())

if __name__ == '__main__':
    human_game()
    