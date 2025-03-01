from splendor_game import SplendorGameState, SplendorGameRules, Action
from player import HumanPlayer, CultivatorPlayer

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
                    game.action(Action.from_str(action_str))
                    break
                except AttributeError as err:
                    print('Invalid action {}: {}'.format(action_str, str(err)))

    print('best player:', game.best_player())

def computer_game():
    player_names = ['player1', 'ai']
    players = [HumanPlayer(), CultivatorPlayer()]
    rules = SplendorGameRules(len(player_names))
    game = SplendorGameState(player_names, rules)

    while not game.check_win():
        for n, player in enumerate(players):
            print(game)
            while True: # check for invalid action inputs
                action = player.get_action(game)
                try:
                    game.action(action)
                    break
                except AttributeError as err:
                    print('Invalid action {}: {}'.format(str(action), str(err)))
                    if player.is_ai:
                        return
            if player.is_ai:
                print(player_names[n] + ' move: ' + str(action))


    print('best player:', game.best_player())


if __name__ == '__main__':
    computer_game()
    