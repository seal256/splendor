import random

from game import GameState, Agent
from splendor_game import SplendorGameState, SplendorPlayerState, GOLD_GEM, Action, ActionType
from mcts import MCTS

class RandomAgent(Agent):
    def get_action(self, game_state: GameState):
        legal_actions = game_state.get_actions()
        return random.choice(legal_actions)

class HumanPlayer(Agent):
    def get_action(self, game_state):
        player_name = game_state.players[game_state.player_to_move].name
        action_str = input(player_name + ' move: ')
        return Action.from_str(action_str)

class MCTSAgent(Agent):
    def __init__(self, **mcts_params):
        self.mcts_params = mcts_params

    def get_action(self, game_state: GameState):
        mcts = MCTS(game_state, **self.mcts_params)
        action = mcts.search()
        return action

class CultivatorPlayer:
    def get_action(self, game_state: SplendorGameState):
        player = game_state.players[game_state.player_to_move]
        gold = player.gems.get(GOLD_GEM)

        action_scores = []
        
        for level, cards in enumerate(game_state.cards):
            for pos, card in enumerate(cards):
                shortage = player.gems.shortage(card.price)
                gold_shortage = shortage.count() - gold
                if gold_shortage <= 0: # affordable card
                    action = Action(ActionType.purchase, level=level, pos=pos)
                    score = card.points
                    action_scores.append((score, action))
                else: 
                    gems = list(shortage.gems.keys())

                    if len(gems) > 3:
                        gems = gems[:3]
                    if len(gems) == 1 and shortage.get(gems[0]) > 1:
                        gems = [gems[0], gems[0]]
                    action = Action(ActionType.take, gems=gems)
                    score = -gold_shortage
                    if card.points > 0 and gold_shortage < 3:
                        score += card.points + 1
                    action_scores.append((score, action))

        _, best_action = sorted(action_scores, reverse=True, key = lambda x: x[0])[0]
        return best_action








    

