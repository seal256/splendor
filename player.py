from splendor_game import SplendorGameState, SplendorPlayerState, GOLD_GEM, Action

class HumanPlayer:
    is_ai = False
    def get_action(self, game_state):
        player_name = game_state.players[game_state.player_to_move].name
        action_str = input(player_name + ' move: ')
        return Action.from_str(action_str)

class CultivatorPlayer:
    is_ai = True
    def get_action(self, game_state: SplendorGameState):
        player = game_state.players[game_state.player_to_move]
        gold = player.gems.get(GOLD_GEM)

        action_scores = []
        
        for level, cards in enumerate(game_state.cards):
            for pos, card in enumerate(cards):
                shortage = player.gems.shortage(card.price)
                gold_shortage = shortage.count() - gold
                if gold_shortage <= 0: # affordable card
                    action = Action(Action.purchase, None, (level, pos))
                    score = card.points
                    action_scores.append((score, action))
                else: 
                    gems = list(shortage.gems.keys())

                    if len(gems) > 3:
                        gems = gems[:3]
                    if len(gems) == 1 and shortage.get(gems[0]) > 1:
                        gems = [gems[0], gems[0]]
                    action = Action(Action.take, gems, None)
                    score = -gold_shortage
                    if card.points > 0 and gold_shortage < 3:
                        score += card.points + 1
                    action_scores.append((score, action))

        _, best_action = sorted(action_scores, reverse=True, key = lambda x: x[0])[0]
        return best_action








    

