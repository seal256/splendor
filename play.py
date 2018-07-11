from game import SplendorGameState, SplendorGameRules


if __name__ == '__main__':
    player_names = ['player1', 'player2']
    rules = SplendorGameRules(len(player_names))
    game = SplendorGameState(player_names, rules)

    while not game.check_win():
        for player_name in player_names:
            print(game)
            action_str = input(player_name + ' move: ')
            game.action(action_str)
            
    print('best player:', game.best_player())

    