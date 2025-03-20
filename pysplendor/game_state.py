from abc import ABC, abstractmethod
from copy import deepcopy

CHANCE_PLAYER = -1

class GameState(ABC):
    @abstractmethod
    def get_actions(self):
        '''Returns the list of legal actions'''
        pass

    @abstractmethod
    def active_player(self) -> int:
        '''Returns the index of the active player (who's turn is now) or None for a chance game state. Can be used as an index for the rewards list.'''
        pass

    @abstractmethod
    def apply_action(self, action):
        '''Applies the action to thae game state, modifying it'''
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        '''Returns true if the state is terminal'''
        pass

    @abstractmethod
    def rewards(self) -> list[float]:
        '''Returns a list of rewards for each player'''
        pass

    @abstractmethod
    def copy(self):
        '''Copy of the state, that allows independent game continuation from the new object'''
        pass

class Agent(ABC):
    '''Game player'''

    @abstractmethod
    def get_action(game_state: GameState):
        pass

    # def is_stateless(self):
    #     return True
    
    # def apply_action(action):
    #     '''Stateful agents should implement this method to keep track of the game'''
    #     pass