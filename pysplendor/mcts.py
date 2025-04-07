import math
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from dataclasses import dataclass

from .game_state import GameState, CHANCE_PLAYER

class Node:
    '''MCTS node (that doesn't store the state)'''

    def __init__(self, action=None, parent=None, acting_player=None):
        self.action = action  # the action that was applied to the parent and lead to this state
        self.parent = parent
        self.acting_player = acting_player  # active_player at the parent state
        self.children = []
        self.visits = 0
        self.wins = 0 # wins for the acting_player (who took the action)
        self.p = 1.0  # prior action probability

# This implementation is suited for memory intensive game setups: 
# game states are big, next state computation is relatively cheap.  
# In such conditions it is more efficient to store only action in each node, 
# and recompute state evolution on the fly, starting from the root on each iteration.
# This avoids excessive memory consumption and state copying cost.

@dataclass
class MCTSParams:
    iterations: int = 1000
    exploration: float = 1.4
    weighted_selection_moves: int = -1
    value_weight: float = 0.5

class MCTS:
    '''MCTS algorithm'''
    
    def __init__(self, state: GameState, params: MCTSParams = None):
        if params is None:
            params = MCTSParams()
        self.root_state = state.copy()
        self.root = Node()
        self.params = params
        
    def search(self):
        '''Grows the search tree and returns the best expected action'''
        for _ in range(self.params.iterations):
            self._search_iteration()
        return self.best_action()

    def _search_iteration(self):
        # Selection
        node = self.root
        state = self.root_state.copy()
        while node.children and node.visits > 0:  # visits > 0 assures we performed at least one rollout
            node = self._select_child(state, node)
            state.apply_action(node.action)

        # Expansion
        if not state.is_terminal() and not node.children and node.visits > 0:
            acting_player = state.active_player()
            for action in state.get_actions():
                child_node = Node(action=action, parent=node, acting_player=acting_player)
                node.children.append(child_node)
            random.shuffle(node.children)  # Optional step that simplifies selection phase
            if node.children:
                node = self._select_child(state, node)
                state.apply_action(node.action)

        # Simulation
        rewards = self._rollout(state)

        # Backpropagation
        while node is not None:
            node.visits += 1
            if not node.parent:  # skip root node
                break
            if node.acting_player == CHANCE_PLAYER:  # do not record wins, but downscale the rewards
                if node.parent.children:
                    rewards = [r / len(node.parent.children) for r in rewards]
            else:
                node.wins += rewards[node.acting_player]
            node = node.parent
    
    def best_action(self):
        '''Returns best action at the root state'''
        if not self.root.children:
            raise RuntimeError("No actions available")
        
        if (self.params.weighted_selection_moves == -1 or 
            self.root_state.move_num() > self.params.weighted_selection_moves):
            # select by max visits
            return max(self.root.children, key=lambda c: c.visits).action
        else:
            # select with probabilities proportional to visits
            visits = [c.visits for c in self.root.children]
            total = sum(visits)
            if total == 0:
                return random.choice(self.root.children).action
            probs = [v / total for v in visits]
            return random.choices(self.root.children, weights=probs, k=1)[0].action

    def root_visits(self) -> List[Tuple[Any, int]]:
        '''Returns children visits at the root state'''
        return [(child.action, child.visits) for child in self.root.children]

    def apply_action(self, action):
        '''Applies an actual game action and rebases the tree root.'''
        self.root_state.apply_action(action)
        
        next_node = next((child for child in self.root.children if child.action == action), None)
        if next_node:
            self.root = next_node
            self.root.parent = None
        else:  # unexplored action
            self.root = Node(action=action, acting_player=self.root_state.active_player())

    def _select_child(self, state, node):
        '''Selects child node according to UCB or random for chance nodes'''
        if state.active_player() == CHANCE_PLAYER:
            return random.choice(node.children)

        # Use the first unexplored child, assuming the children array is shuffled
        for child in node.children:
            if child.visits == 0:
                return child

        # Find the max UCB child
        log_parent_visits = math.log(node.visits)
        return max(node.children, key=lambda c: c.wins / c.visits + self.params.exploration * math.sqrt(log_parent_visits / c.visits))

    def _rollout(self, state: GameState):
        '''Simulates a random playout from the given state'''
        while not state.is_terminal():
            actions = state.get_actions()
            if not actions:
                break
            action = random.choice(actions)
            state.apply_action(action)
        return state.rewards()

class Policy(ABC):
    @abstractmethod
    def predict(self, game_state: GameState) -> List[float]:
        '''Returns probabilities of available actions in the order provided by game_state.get_actions()'''
        pass

class Value(ABC):
    @abstractmethod
    def predict(self, game_state: GameState) -> List[float]:
        '''Estimates the value of the game_state for each player'''
        pass

class PolicyMCTS(MCTS):
    def __init__(self, state: GameState, policy: Policy, params: MCTSParams = None):
        super().__init__(state, params)
        self.policy = policy # action selection policy

    def _search_iteration(self):
        # Selection
        node = self.root
        state = self.root_state.copy()
        while node.children and node.visits > 0:
            node = self._select_child(state, node)
            state.apply_action(node.action)

        # Expansion
        if not state.is_terminal() and not node.children and node.visits > 0:
            actions = state.get_actions()
            acting_player = state.active_player()
            probs = ([1.0] * len(actions) if acting_player == CHANCE_PLAYER 
                    else self.policy.predict(state))
            for action, p in zip(actions, probs):
                child_node = Node(action=action, parent=node, acting_player=acting_player)
                child_node.p = p
                node.children.append(child_node)
            random.shuffle(node.children)  # Optional step that simplifies selection phase
            if node.children:
                node = self._select_child(state, node)
                state.apply_action(node.action)

        # Simulation
        rewards = self._rollout(state)

        # Backpropagation
        while node is not None:
            node.visits += 1
            if not node.parent:  # skip root node
                break
            if node.acting_player == CHANCE_PLAYER:  # do not record wins, but downscale the rewards
                if node.parent.children:
                    rewards = [r / len(node.parent.children) for r in rewards]
            else:
                node.wins += rewards[node.acting_player]
            node = node.parent

    def _select_child(self, state, node):
        '''Selects child node according to UCB or random for chance nodes'''
        if state.active_player() == CHANCE_PLAYER:
            return random.choice(node.children)
        
        max_ucb = -1
        best_child = None
        parent_visits_sqrts = math.sqrt(node.visits)
        for child in node.children:
            exploitation_term = child.wins / child.visits if child.visits > 0 else 0
            exploration_term = child.p * parent_visits_sqrts / (child.visits + 1)
            ucb = exploitation_term + self.params.exploration * exploration_term
            if ucb > max_ucb:
                max_ucb = ucb
                best_child = child
        if best_child is None:
            raise RuntimeError("Unable to find best child!")
        return best_child
    
class PVMCTS(MCTS):
    def __init__(self, state: GameState, policy: Policy, value: Value, params: MCTSParams = None):
        super().__init__(state, params)
        self.policy = policy # action selection policy
        self.value = value # state value estimator

    def _search_iteration(self):
        # Selection
        node = self.root
        state = self.root_state.copy()
        while node.children and node.visits > 0:
            node = self._select_child(state, node)
            state.apply_action(node.action)

        # Expansion
        if not state.is_terminal() and not node.children and node.visits > 0:
            actions = state.get_actions()
            acting_player = state.active_player()
            probs = ([1.0] * len(actions) if acting_player == CHANCE_PLAYER 
                    else self.policy.predict(state))
            for action, p in zip(actions, probs):
                child_node = Node(action=action, parent=node, acting_player=acting_player)
                child_node.p = p
                node.children.append(child_node)
            random.shuffle(node.children)  # Optional step that simplifies selection phase
            if node.children:
                node = self._select_child(state, node)
                state.apply_action(node.action)

        # Simulation
        predicted_values = self.value.predict(state)
        rewards = self._rollout(state)
        vw = self.params.value_weight
        rewards = [(1 - vw) * r + vw * v for r, v in zip(rewards, predicted_values)]

        # Backpropagation
        while node is not None:
            node.visits += 1
            if not node.parent:  # skip root node
                break
            if node.acting_player == CHANCE_PLAYER:  # do not record wins, but downscale the rewards
                if node.parent.children:
                    rewards = [r / len(node.parent.children) for r in rewards]
            else:
                node.wins += rewards[node.acting_player]
            node = node.parent

    def _select_child(self, state, node):
        '''Selects child node according to UCB or random for chance nodes'''
        if state.active_player() == CHANCE_PLAYER:
            return random.choice(node.children)
        
        max_ucb = -1
        best_child = None
        parent_visits_sqrts = math.sqrt(node.visits)
        for child in node.children:
            exploitation_term = child.wins / child.visits if child.visits > 0 else 0
            exploration_term = child.p * parent_visits_sqrts / (child.visits + 1)
            ucb = exploitation_term + self.params.exploration * exploration_term
            if ucb > max_ucb:
                max_ucb = ucb
                best_child = child
        if best_child is None:
            raise RuntimeError("Unable to find best child!")
        return best_child