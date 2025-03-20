import math
import random
from tqdm import tqdm

from .game_state import GameState, CHANCE_PLAYER

class Node:
    '''MCTS node (that doesn't store the state)'''

    def __init__(self, action=None, active_player=None, parent=None):
        self.action = action # the action that was applied to the parent and lead to this state
        self.parent = parent
        self.active_player = active_player
        self.children = []
        self.visits = 0
        self.wins = 0 # wins for the active_player (who's turn it is at the current step)
        self.p = 1.0 # estimated probability of the action

# This implementation is suited for memory intensive game setups: 
# game states are big, next state computation is relatively cheap.  
# In such conditions it is more efficient to store only action in each node, 
# and recompute state evolution on the fly, starting from the root on each iteration.
# This avoids excessive memory consumption and state copying cost.

class MCTS:
    '''MCTS algorithm'''
    
    def __init__(self, state: GameState, iterations=1000, exploration=1.4):
        self.root_state: GameState = state.copy()
        self.root = Node(active_player=state.active_player())
        self.iterations = iterations
        self.exploration = exploration
        
    def search(self):
        '''Grows the search tree and returns the best expected action'''
        for _ in tqdm(range(self.iterations)):
            self._search_iteration()
        return self.best_action()

    def _search_iteration(self):
        # Selection
        node: Node = self.root
        state = self.root_state.copy()
        while node.children and node.visits > 0:
            node = self._select_child(node)
            state.apply_action(node.action)

        # Expansion
        if not state.is_terminal() and not node.children and node.visits > 0:
            for action in state.get_actions():
                child_node = Node(action=action, active_player=state.active_player(), parent=node)
                node.children.append(child_node)
            node = random.choice(node.children)
            state.apply_action(node.action)

        # Simulation
        rewards = self._rollout(state)

        # Backpropagation
        while node is not None:
            node.visits += 1
            # active_player may be None if we are at a chance node
            if node.active_player is not None: 
                node.wins += rewards[node.active_player]
            node = node.parent
    
    def _rollout(self, state: GameState):
        # Simulates a random playout from the given state
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
        
        Maintains a relevant part of the tree for next searches.'''
        self.root_state = self.root_state.apply_action(action)

        next_node = next((child for child in self.root.children if action == child.action), None)
        if next_node:
            self.root = next_node
            self.root.parent = None

        else: # unexplored action
            self.root = Node(active_player=self.root_state.active_player())

    def _select_child(self, node):
        '''Uses UCB-like criterion to select best node for rollout'''
        unexplored = [c for c in node.children if c.visits == 0]
        if unexplored:
            return random.choice(unexplored)
        return max(node.children, key=lambda c: c.wins / c.visits + self.exploration * node.p * math.sqrt(math.log(node.visits) / c.visits))

class ActionEncoder:
    def action_id(self, action):
        pass

class Policy:
    def predict(self, state) -> list[float]:
        pass

class PolicyMCTS(MCTS):
    def __init__(self, state: GameState, policy: Policy, action_encoder: ActionEncoder, policy_weight=0.5, iterations=1000, exploration=1.4):
        super().__init__(state, iterations, exploration)
        self.policy = policy # action selection policy
        self.action_encoder = action_encoder
        self.policy_weight = policy_weight

    def _search_iteration(self):
       # Selection
        node: Node = self.root
        state = self.root_state.copy()
        while node.children and node.visits > 0:
            node = self._select_child(node)
            state.apply_action(node.action)

        # Expansion
        if not state.is_terminal() and not node.children and node.visits > 0:
            actions = state.get_actions()
            active_player = state.active_player()
            if active_player != CHANCE_PLAYER:
                probs = self._estimate_probs(state, actions)
            else:
                probs = [1] * len(actions)
            for action, p in zip(actions, probs):
                child_node = Node(action=action, active_player=active_player, parent=node)
                child_node.p = p
                node.children.append(child_node)
            node = self._select_child(node)
            state.apply_action(node.action)

        # Simulation
        rewards = self._rollout(state)

        # Backpropagation
        while node is not None:
            node.visits += 1
            # active_player may be None if we are at a chance node
            if node.active_player is not None: 
                node.wins += rewards[node.active_player]
            node = node.parent
    
    def _estimate_probs(self, state, actions):
        '''Returns probability estimates for selected actions'''
        probs = self.policy.predict(state)
        w = self.policy_weight
        action_probs = [(1 - w) + w * probs[self.action_encoder.action_id(a)] for a in actions] # all probabilities are artificially increased to avoid zeroing out
        s = sum(action_probs)
        action_probs = [p / s for p in action_probs]
        return action_probs

    def _select_child(self, node):
        '''Uses UCB-like criterion to select best node for rollout'''
        return max(node.children, key=lambda c: c.wins / (c.visits + 1) + self.exploration * node.p * math.sqrt(node.visits + 1) / (c.visits + 1))
  

