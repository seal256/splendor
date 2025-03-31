from copy import deepcopy
import random
from tqdm import tqdm
import pytest

from pysplendor.mcts import MCTS, MCTSParams
from pysplendor.agents import MCTSAgent, RandomAgent
from pysplendor.game import run_one_game

class TicTacToe:
    def __init__(self):
        self.player = 0  # 0 or 1
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
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
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

    def __str__(self):
        symbols = {None: '.', 0: 'X', 1: 'O'}
        result = ''
        for i in range(0, 9, 3):
            result += ' '.join(symbols[v] for v in self.board[i:i + 3]) + '\n'
        return result

def test_mcts_vs_random_agent():
    """Test that the MCTS implementation never loses in TicTacToe against a random agent."""

    # mcts_params = MCTSParams(iterations= 1000)
    agents = [RandomAgent(), MCTSAgent()]
    mcts_player_id = 1

    for _ in range(100):
        state = TicTacToe()
        traj = run_one_game(state, agents)
        reward = traj.rewards[mcts_player_id]
        if (reward < 0.5):
            state = traj.initial_state.copy()
            print(state)    
            for action in traj.actions:
                state.apply_action(action)
                print(state)
            print(f'rewards: {traj.rewards}')
        assert reward >= 0.5, "MCTS agent should never lose against a random agent"