import math
import random
from tqdm import tqdm

from game import GameState

class Node:
    def __init__(self, state: GameState, action=None, parent=None):
        self.state: GameState = state
        self.action = action # the action that was applied to parent and lead to this state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0 # wins for the player who's turn it is at the current state

    def get_state(self) -> GameState:
        '''Lazy state calculation'''
        if self.state is None:
            self.state = self.parent.state.next_state(self.action) 
        return self.state

class MCTS:
    def __init__(self, state: GameState, iterations=1000, exploration = 1.4):
        self.root = Node(state.copy())
        self.iterations = iterations
        self.exploration = exploration

    def search(self):
        '''Grows the search tree and returns the best expected action'''
        for _ in tqdm(range(self.iterations)):
            self._search_iteration()
        return self.best_action()

    def _search_iteration(self):
        # Selection
        node = self.root
        while node.children:
            node = self._select_child(node)

        # Expansion
        if not node.get_state().is_terminal():
            for action in node.get_state().get_actions():
                # state copying becomes expensive with a lot of node expansions
                # we will calculate the child state when it's needed
                child_node = Node(state=None, action=action, parent=node)
                node.children.append(child_node)
            node = random.choice(node.children)

        # Simulation
        rewards = self._rollout(node.get_state())

        # Backpropagation
        while node is not None:
            node.visits += 1
            if node.parent:
                active_player = node.parent.get_state().active_player() # active_player may be None if we are at a chance node
                if active_player is not None: 
                    node.wins += rewards[active_player]
            node = node.parent

    def _rollout(self, state: GameState):
        # Simulates a random playout from the given state
        state = state.copy()
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
            self.root = Node(self.root.get_state().next_state(action))

    def _select_child(self, node):
        '''Uses UCB-like criterion to select best node for rollout'''
        unexplored = [c for c in node.children if c.visits == 0]
        if unexplored:
            return random.choice(unexplored)
        return max(node.children, key=lambda c: c.wins / c.visits + self.exploration * math.sqrt(math.log(node.visits) / c.visits))


