import random
from tqdm import tqdm
from copy import deepcopy

from .mcts import MCTS, GameState

class TicTacToe(GameState):
    def __init__(self):
        self.player = 0 # 0 or 1
        self.board = [None] * 9

    def get_actions(self):
        return [i for i, v in enumerate(self.board) if v is None]

    def active_player(self):
        return self.player

    def apply_action(self, action):
        self.board[action] = self.player
        self.player = 1 - self.player
        return self

    def is_terminal(self):
        return self._winner() is not None or len(self.get_actions()) == 0

    def _winner(self):
        lines = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c]:
                if self.board[a] is not None:
                    return self.board[a]
        return None
    
    def rewards(self):
        winner_id = self._winner()
        rewards = [0.0, 0.0]
        if winner_id is not None:
            rewards[winner_id] = 1.0
        else:
            rewards = [0.5, 0.5]
        return rewards
    
    def copy(self):
        return deepcopy(self)
    
    def display(self):
        symbols = {None: '.', 0: 'X', 1: 'O'}
        for i in range(0, 9, 3):
            print(' '.join(symbols[v] for v in self.board[i:i+3]))

def console_game():
    '''Allows to play with MCTS agent in terminal'''

    state = TicTacToe()
    mcts = MCTS(state)
    human_palyer_id = 0

    state.display()
    while not state.is_terminal():
        if state.player == human_palyer_id:  # Human's turn
            valid_actions = state.get_actions()
            action = int(input("Enter your action (0-8): "))
            while action not in valid_actions:
                action = int(input("Invalid action. Enter your action (0-8): "))
        else:
            print("Computer's turn...")
            action = mcts.search()
            # for child in mcts.root.children:
            #     print(f'Action: {child.action} Visits: {child.visits} Wins: {child.wins}')
        state.apply_action(action)
        mcts.apply_action(action)
        state.display()
   
    rewards = state.rewards()    
    print(f"Your reward is {rewards[human_palyer_id]}")

def random_agent_test():
    '''Checks that the MCTS implementation never looses in TicTacToe against a random agent'''
    
    mcts_palyer_id = 1 # mcts plays O-s

    for _ in tqdm(range(100)):
        state = TicTacToe()
        mcts = MCTS(state, iterations=400)
        while not state.is_terminal():
            if state.player == mcts_palyer_id:
                action = mcts.search()
            else:
                valid_actions = state.get_actions()
                action = random.choice(valid_actions)
            state.apply_action(action)
            mcts.apply_action(action)
            
        rewards = state.rewards()   
        assert rewards[mcts_palyer_id] >= 0.5, "MCTS agent shoulde never loose against a random agent"    
    print('Success!')


if __name__ == "__main__":
    # console_game()
    random_agent_test()