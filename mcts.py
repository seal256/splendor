from abc import ABC, abstractmethod
from copy import deepcopy
import math
import random

class GameState(ABC):
    @abstractmethod
    def get_actions(self):
        '''Returns the list of legal actinos'''
        pass

    @abstractmethod
    def active_player(self) -> int:
        '''Returns the index of the active player (who's turn is now) or None for a chance game state. Can be used as an index for the rewards list.'''
        pass

    @abstractmethod
    def apply_action(self, action):
        '''Applies the action to the game state, modifying it'''
        pass

    def next_state(self, action):
        '''Returns next state keeping this object intact'''
        state = deepcopy(self)
        state.apply_action(action)
        return state

    @abstractmethod
    def is_terminal(self) -> bool:
        '''Returns true if the state is terminal'''
        pass

    @abstractmethod
    def rewards(self) -> list[int]:
        '''Returns a list of rewards for each player'''
        pass

class Node:
    def __init__(self, state: GameState, action=None, parent=None):
        self.state: GameState = state
        self.action = action # the action that was applied to parent and lead to this state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0 # wins for the player who's turn it is at the current state

class MCTS:
    def __init__(self, state: GameState, iterations=1000, exploration = 1.4):
        self.root = Node(deepcopy(state))
        self.iterations = iterations
        self.exploration = exploration

    def search(self):
        '''Grows the search tree and returns the best expected action'''
        for _ in range(self.iterations):
            self._search_iteration()
        return self.best_action()

    def _search_iteration(self):
        # Selection
        node = self.root
        while node.children:
            node = self._select_child(node)

        # Expansion
        if not node.state.is_terminal():
            for action in node.state.get_actions():
                next_state = node.state.next_state(action)
                child_node = Node(state=next_state, action=action, parent=node)
                node.children.append(child_node)
            node = random.choice(node.children)

        # Simulation
        rewards = self._rollout(node.state)

        # Backpropagation
        while node is not None:
            node.visits += 1
            if node.parent:
                active_player = node.parent.state.active_player() # active_player may be None if we are at a chance node
                if active_player is not None: 
                    node.wins += rewards[active_player]
            node = node.parent

    def _rollout(self, state: GameState):
        # Simulates a random playout from the given state
        state = deepcopy(state)
        while not state.is_terminal():
            actions = state.get_actions()
            action = random.choice(actions)
            state.apply_action(action)
        return state.rewards()

    def best_action(self):
        '''Returns best action at the root state'''
        return max(self.root.children, key=lambda c: c.visits).action

    def apply_action(self, action):
        '''Applies an actual game action and rebases the tree root. 
        
        Maintains a relevant part of the tree for next searches. Be careful to keep the same active player before making new searches.'''

        next_node = next((child for child in self.root.children if action == child.action), None)
        if next_node:
            self.root = next_node
            self.root.parent = None

        else: # unexplored action
            self.root = Node(self.root.state.next_state(action))

    def _select_child(self, node):
        '''Uses UCB-like criterion to select best node for rollout'''
        unexplored = [c for c in node.children if c.visits == 0]
        if unexplored:
            return random.choice(unexplored)
        return max(node.children, key=lambda c: c.wins / c.visits + self.exploration * math.sqrt(math.log(node.visits) / c.visits))


