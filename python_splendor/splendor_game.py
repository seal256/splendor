import random, copy, csv
from itertools import combinations

from game import GameState

GOLD_GEM = 5 # index of the gold gem
NUM_GEMS = 6
GEMS = tuple(range(NUM_GEMS))
GEM_STR = ('r', 'g', 'b', 'w', 'k', 'y') # red, green, blue, white, black, yellow(gold)
GEM_STR_TO_VAL = {val: idx for idx, val in enumerate(GEM_STR)}
CARD_LEVELS = 3

class ActionType:
    skip = 's' # skip the move
    take = 't' # take gems
    reserve = 'r' # reserve card
    purchase = 'p' # purchase table card
    purchase_hand = 'h' # purchase hand card
    new_table_card = 'c' # new table card from deck. Performed by randomness rather than any of the players

class GemSet(list):
    '''Count/price for each gem'''
    def __init__(self):
        super().__init__([0] * 6)

    def __str__(self):
        return ''.join([GEM_STR[gem] + str(count) for gem, count in enumerate(self) if count > 0])

class Noble:
    def __init__(self, points=0, price=None):
        self.points = points # number of win points
        self.price = price
        if price is None:
            self.price = GemSet()

    @staticmethod
    def parse_str(string):
        '''Parse from string (assumes serialization by self method str)'''
        assert len(string) >= 8
        points = int(string[1])
        num_gems = (len(string) - 4)//2
        assert num_gems >= 2 and num_gems <= 3
        price = GemSet()
        for n in range(num_gems):
            gem = GEM_STR_TO_VAL.get(string[2*n + 3])
            count = int(string[2*n + 4])
            price[gem] += count
        return points, price

    @classmethod
    def from_str(cls, string):
        '''Alternative constructor from string'''
        points, price = Noble.parse_str(string)
        return cls(points, price)

    def __str__(self):
        return '[' + str(self.points) + '|' + str(self.price) + ']'

class Card:
    def __init__(self, gem=None, points=0, price=None):
        self.gem = gem # title gem
        self.points = points # number of win points
        self.price = price 
        if price is None:
            self.price = GemSet()

    def __str__(self):
        return '[' + GEM_STR[self.gem] + str(self.points) + '|' + str(self.price) + ']'

def read_cards_from_csv(file_name):
    cards = [[], [], []]
    reader = csv.reader(open(file_name))
    next(reader, None) # skip header
    for line in reader:
        assert len(line) == 8
        card = Card()
        level = int(line[0]) - 1
        card.gem = GEM_STR_TO_VAL.get(line[1])
        card.points = int(line[2])
        for gem, amount in zip(GEMS[:-1], line[3:]):
            if len(amount) == 1:
                card.price[gem] += int(amount)
        cards[level].append(card)
    return tuple([tuple(cards[n]) for n in range(CARD_LEVELS)])

NOBLES = tuple(map(Noble.from_str, ['[3|r4g4]', '[3|g4b4]', '[3|b4w4]', '[3|w4k4]', '[3|k4r4]',
    '[3|r3g3b3]', '[3|b3g3w3]', '[3|b3w3k3]', '[3|w3k3r3]', '[3|k3r3g3]']))
CARDS = read_cards_from_csv('cards.csv')

def print_cards():
    for level, deck in enumerate(CARDS):
        print(f'level {level}')
        print(','.join([f'"{card}"' for card in deck]))
    
class SplendorPlayerState:
    def __init__(self, name):
        self.name = name
        self.card_gems = GemSet() # gems of acquired cards
        self.gems = GemSet() # gems on the table
        self.hand_cards = []
        self.points = 0 # winning points from all aquired cards

    def __str__(self):
        s = self.name + '|' + str(self.points) + '\n'
        s += 'card gems:' + str(self.card_gems) + '\n'
        s += 'gems:' + str(self.gems) + '\n'
        s += 'hand:'
        for card in self.hand_cards:
            s += str(card)
        s += '\n'
        return s
    
    def copy(self):
        cp = copy.copy(self) # shallow copy
        cp.card_gems = copy.copy(self.card_gems) # deep copy
        cp.gems = copy.copy(self.gems)
        cp.hand_cards = list(self.hand_cards)
        return cp

class Action:
    def __init__(self, action_type: ActionType, gems: list[str] = None, level: int = None, pos:int = None):
        self.type: ActionType = action_type
        self.gems: list[str] = gems # list of gem idx-s 
        self.level: int = level
        self.pos: int = pos

    def __str__(self):
        if self.type == ActionType.skip:
            return self.type
        if self.type == ActionType.take: 
            return self.type + ''.join([GEM_STR[g] for g in self.gems])
        elif self.type == ActionType.reserve or self.type == ActionType.purchase or self.type == ActionType.new_table_card:
            return f'{self.type}{self.level}n{self.pos}'
        elif self.type == ActionType.purchase_hand:
            return f'{self.type}{self.pos}'
        else:
            raise ValueError('Unknown action type')

    @classmethod
    def from_str(cls, action_str):
        try:
            action_type, gems, level, pos = Action.parse(action_str)
            return cls(action_type, gems, level, pos)
        except Exception:
            raise ValueError('Invalid action string {}'.format(action_str))

    @staticmethod
    def parse(action_str):
        action_type = action_str[0] # should be one of ACTIONS
        gems = None
        level = None
        pos = None

        if action_type == ActionType.skip:
            pass
        elif action_type == ActionType.take: 
            gems = [g for g in action_str[1:]]
        elif action_type == ActionType.reserve: 
            level, pos = Action.scan_level_pos(action_str)
        elif action_type == ActionType.purchase: 
            level, pos = Action.scan_level_pos(action_str) 
        elif action_type == ActionType.purchase_hand: 
            pos = (int(action_str[1:]),) # the 0-based index of a hand card
        elif action_type == ActionType.new_table_card:
            pos = (int(action_str[1:]),) # the 0-based index of a deck card
        else:
            raise ValueError('Invalid action type {} (in action {})'.format(action_type, action_str))
        
        return action_type, gems, level, pos

    @staticmethod
    def scan_level_pos(action_str):
        parts = action_str.split('n') # seperator between the card level and position
        assert len(parts) == 2
        level = int(parts[0])
        pos = int(parts[1])
        return level, pos

class SplendorGameRules:
    '''A set of constants that affect the game mechanics. Depend on the number of players'''

    def __init__(self, num_players):
        self.num_players = num_players
        self.max_open_cards = 4 # open cards on table
        self.max_hand_cards = 3 # max cards in player hand
        self.win_points = 15
        self.max_player_gems = 10
        self.max_nobles = self.num_players + 1
        self.max_gems_take = 3 # max gems to take
        self.max_same_gems_take = 2 # max same color gems to take
        self.min_same_gems_stack = 4 # min size of gem stack from which you can take 2 same color gems
        self.max_gold = 5
        self.max_gems = 7 # max same color gems on table (except gold)
        if self.num_players < 4:
            self.max_gems = 2 + self.num_players
 
class SplendorGameState(GameState):
    '''Knows game rules)'''

    def __init__(self, player_names, rules: SplendorGameRules):
        assert rules.num_players == len(player_names)

        self.rules = rules
        self.round = 0
        self.player_to_move = 0
        self.skips = 0 # number of players that skipped move in this round. If all players skipped, the game ends
        self.table_card_needed = False # indicates that previous player just purchased a table card
        self.deck_level = 0 # the level of the deck that will be used to select card if self.table_card_needed is True

        # init nobles
        self.nobles = random.sample(NOBLES, self.rules.max_nobles)

        # init decks and cards 
        self.decks = []
        self.cards = [] # open cards on table
        for level in range(CARD_LEVELS):
            cards = list(CARDS[level])
            random.shuffle(cards)
            open_cards = self.rules.max_open_cards
            self.decks.append(cards[:-open_cards])
            self.cards.append(cards[-open_cards:])

        # init gems
        self.gems = GemSet()
        for gem in GEMS[:-1]:
            self.gems[gem] = self.rules.max_gems
        self.gems[GOLD_GEM] = self.rules.max_gold

        # init players
        self.players = [SplendorPlayerState(name) for name in player_names]

    def copy(self):
        cp = copy.copy(self) # shallow copy
        # we don't need to copy cards or rules, but we need to reinit the lists
        cp.nobles = list(self.nobles)
        cp.decks = [list(self.decks[level]) for level in range(CARD_LEVELS)]
        cp.cards = [list(self.cards[level]) for level in range(CARD_LEVELS)]
        cp.gems = copy.copy(self.gems)
        cp.players = [player.copy() for player in self.players]
        return cp

    def __str__(self):
        s = 'round: ' + str(self.round) + ' player to move: ' + str(self.active_player()) + '\n'
        
        s += 'nobles: '
        for noble in self.nobles:
            s += str(noble) + ' '
        s += '\n'

        for n, card_list in enumerate(reversed(self.cards)):
            s += str(CARD_LEVELS - n) + ': '
            for card in card_list:
                if card:
                    s += str(card) + ' '
            s += '\n'

        s += 'gems: ' + str(self.gems) + '\n'

        for player in self.players:
            s += str(player)

        return s 

    def apply_action(self, action: Action):
        player = self.players[self.player_to_move]
        if self.player_to_move == 0:
            self.skips = 0 # resets in the beginning of each round

        if action.type == ActionType.skip:
            self.skips += 1
            self._increment_player_to_move()

        elif action.type == ActionType.take: 
            gems = action.gems
            unique_gems = list(set(gems))
            if len(gems) > self.rules.max_gems_take:
                raise ValueError('Can\'t take more than {} gems'.format(self.rules.max_gems_take))
            if len(unique_gems) == 1: # all same color
                if self.gems[unique_gems[0]] < self.rules.min_same_gems_stack:
                    raise ValueError('Should be at least {} gems in stack'.format(self.rules.min_same_gems_stack))
                if len(gems) != 1 and len(gems) > self.rules.max_same_gems_take: 
                    raise ValueError('Can\'t take more than {} identical gems'.format(self.rules.max_same_gems_take))
            if len(unique_gems) > 1 and len(unique_gems) != len(gems): 
                raise ValueError('You can either take all identical or all different gems')
            if sum(player.gems) + len(gems) > self.rules.max_player_gems:
                raise ValueError('Player can\'t have more than {} gems'.format(self.rules.max_player_gems))

            for gem in gems:
                if gem not in GEMS:
                    raise ValueError('Invalid gem {}'.format(gem))
                if gem == GOLD_GEM:
                    raise ValueError('You are not allowed to take gold ({}) gem'.format(GOLD_GEM))
                if self.gems[gem] == 0:
                    raise ValueError('Not inough {} gems on table'.format(gem))
                
                player.gems[gem] += 1
                self.gems[gem] -= 1
            
            self._increment_player_to_move()

        elif action.type == ActionType.reserve: 
            # level and pos indicate the position of the card to reserve on the table
            if action.level < 0 or action.level >= CARD_LEVELS:
                raise ValueError('Invalid deck level {}'.format(action.level))
            if action.pos < 0 or action.pos >= len(self.cards[action.level]):
                raise ValueError('Invalid card position {}'.format(action.pos))
            if len(player.hand_cards) >= self.rules.max_hand_cards:
                raise ValueError('Player can\'t reserve more than {} cards'.format(self.rules.max_hand_cards))

            card = self.cards[action.level].pop(action.pos)
            self._set_table_card_needed(action.level)
                
            # TODO: There's no option to blindly reserve from the deck
            # if action.pos == -1: # blind reserve from the deck
            #     if not self.decks[action.level]:
            #         raise ValueError('Deck {} is empty'.format(action.level))
            #     card = self.decks[action.level].pop()

            player.hand_cards.append(card)
            if self.gems[GOLD_GEM] > 0:
                player.gems[GOLD_GEM] += 1
                self.gems[GOLD_GEM] -= 1

            self._increment_player_to_move()

        elif action.type == ActionType.purchase: 
            # level and pos indicate the position of the card to purchase on the table
            if action.level < 0 or action.level >= CARD_LEVELS:
                raise ValueError('Invalid deck level {}'.format(action.level))
            if action.pos < 0 or action.pos >= self.rules.max_open_cards:
                raise ValueError('Invalid card position {}'.format(action.pos))

            card = self.cards[action.level].pop(action.pos) # takes the card from the table
            self._purchase_card(player, card)
            self._set_table_card_needed(action.level)
            self._get_noble(player) # try to get a noble
            self._increment_player_to_move()

        elif action.type == ActionType.purchase_hand: 
            # pos is the position of the card in hand
            if action.pos < 0 or action.pos >= len(player.hand_cards):
                raise ValueError('Invalid card position in hand {}'.format(action.pos))

            card = player.hand_cards.pop(action.pos) # remove card from hand
            self._purchase_card(player, card)
            self._get_noble(player) # try to get a noble
            self._increment_player_to_move()

        elif action.type == ActionType.new_table_card:
            # The game is in an intermediate state that requires to select a new random card from the deck.
            # This (strange) state is reuired to allow "choice" game nodes of search algorithms to work correctly

            if not self.table_card_needed:
                raise ValueError('Game does not require a new table card at the moment')
            if action.level < 0 or action.level >= CARD_LEVELS:
                raise ValueError('Invalid deck level {}'.format(action.level))
            if action.pos < 0 or action.pos >= len(self.decks[action.level]):
                raise ValueError('Invalid card position {}'.format(action.pos))
            
            new_card = self.decks[action.level].pop(action.pos)
            self.cards[action.level].append(new_card)
            self.table_card_needed = False
            # since this is not a player's action, we do not increment the active player's index

        else:
            raise ValueError('Invalid action type {}'.format(action.type))

    def _increment_player_to_move(self):
        self.player_to_move = (self.player_to_move + 1) % self.rules.num_players
        if self.player_to_move == 0: # round end
            self.round += 1

    def _set_table_card_needed(self, level: int):
        if self.decks[level]: # if the deck is not empty
            self.table_card_needed = True
            self.deck_level = level

    def _player_can_afford_card(self, player: SplendorPlayerState, card: Card) -> bool:
        '''Checks if the player can afford the given card'''

        gold_to_pay = 0
        for gem, price in enumerate(card.price):
            gem_to_pay = max(price - player.card_gems[gem], 0)
            if gem_to_pay > 0:
                gem_available = player.gems[gem]
                if gem_to_pay > gem_available:
                    gold_to_pay += gem_to_pay - gem_available

        return gold_to_pay <= player.gems[GOLD_GEM]

    def _purchase_card(self, player: SplendorPlayerState, card: Card):
        '''Performes the card purchase. Throws exception if the player can\'t afford the card'''

        gold_to_pay = 0
        for gem, price in enumerate(card.price):
            gem_to_pay = max(price - player.card_gems[gem], 0)
            if gem_to_pay > 0:
                gem_available = player.gems[gem]
                if gem_to_pay > gem_available:
                    gold_to_pay += gem_to_pay - gem_available
                    self.gems[gem] += gem_available
                    player.gems[gem] = 0
                else:
                    self.gems[gem] += gem_to_pay
                    player.gems[gem] = gem_available - gem_to_pay

        if gold_to_pay > 0:
            if gold_to_pay > player.gems[GOLD_GEM]:
                raise ValueError('Player can\'t afford the card')
            player.gems[GOLD_GEM] -= gold_to_pay
            self.gems[GOLD_GEM] += gold_to_pay

        player.card_gems[card.gem] += 1
        player.points += card.points

    def _get_noble(self, player: SplendorPlayerState):
        '''Attempts to acquire a noble card'''
        
        noble_list = []
        for n, noble in enumerate(self.nobles):
            can_afford = True
            for gem, price in enumerate(noble.price):
                if player.card_gems[gem] < price:
                    can_afford = False
                    break
            if can_afford:
                noble_list.append(n)
                break 
                # potentially more than one noble may be available for aquisition, 
                # but we ignore this possiblilty for simplicity
        
        if noble_list:
            noble = self.nobles.pop(noble_list[0])
            player.points += noble.points

    def get_actions(self) -> list[Action]:
        if self.table_card_needed:
            # move of the gods of randomness, no player is involved
            action_type = ActionType.new_table_card
            level = self.deck_level
            actions = [Action(action_type, level=level, pos=n) for n in range(len(self.decks[level]))]
            return actions
        
        actions = []

        # 0. skip the move
        actions.append(Action(ActionType.skip))

        player: SplendorPlayerState = self.players[self.player_to_move]

        # 1. Take new gems
        # two same gems
        action_type = ActionType.take
        if sum(player.gems) < self.rules.max_player_gems - self.rules.max_same_gems_take: 
            for gem in GEMS[:-1]:
                if self.gems[gem] >= self.rules.min_same_gems_stack:
                    actions.append(Action(action_type, [gem] * self.rules.max_same_gems_take))
                    
        # three distinct gems
        if sum(player.gems) < self.rules.max_player_gems - self.rules.max_gems_take:
            available_gems = [g for g in GEMS[:-1] if self.gems[g] > 0]
            for comb_gems in combinations(available_gems, self.rules.max_gems_take):
                actions.append(Action(action_type, comb_gems))

        # 2. Reserve a card
        if len(player.hand_cards) < self.rules.max_hand_cards:
            action_type = ActionType.reserve
            for level in range(CARD_LEVELS):
                for pos in range(len(self.cards[level])):
                    actions.append(Action(action_type, level=level, pos=pos))
                    

        # 3. Purchase a card from table
        action_type = ActionType.purchase
        for level in range(CARD_LEVELS):
            for pos, card in enumerate(self.cards[level]):
                if self._player_can_afford_card(player, card):
                    actions.append(Action(action_type, level=level, pos=pos))

        # 4. Purchase a card from the hand
        if player.hand_cards:
            action_type = ActionType.purchase_hand
            for pos, card in enumerate(player.hand_cards):
                if self._player_can_afford_card(player, card):
                    actions.append(Action(action_type, pos=pos))

        return actions

    def active_player(self):
        if self.table_card_needed:
            return None
        return self.player_to_move

    def is_terminal(self):
        # the game stops if all players skipped the move in the current round
        if self.skips >= len(self.players):
            return True
        
        # or one of them got the required number of win points
        for player in self.players:
            if player.points >= self.rules.win_points:
                return True
        
        return False

    def rewards(self):
        '''Returns the number of win points for each player'''
        # return [player.points for player in self.players]
        return [1.0 if player.points >= self.rules.win_points else 0.0 for player in self.players]

    def best_player(self):
        '''Returns name of best player'''
        scores = [(player.points, player.name) for player in self.players]
        return sorted(scores, reverse=True)[0]

if __name__ == '__main__':
    print_cards()

        



