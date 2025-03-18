import random, copy, csv, json
from itertools import combinations

from .game_state import GameState

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

    def unique(self):
        return sum([1 for x in self if x > 0])

    @classmethod
    def from_str(cls, input_str):
        '''Parses a string like "r2g1b4" and returns a GemSet object.'''
        gem_set = cls()
        pos = 0
        while pos < len(input_str):
            gem_char = input_str[pos]
            if gem_char in GEM_STR_TO_VAL:
                gem = GEM_STR_TO_VAL[gem_char]
                pos += 1

                # Check if the next character is a digit (count)
                count = 1  # Default count is 1 if no number is specified
                if pos < len(input_str) and input_str[pos].isdigit():
                    count = int(input_str[pos])  # Read at most 1 digit
                    pos += 1

                gem_set[gem] += count
            else:
                raise ValueError(f"Invalid gem character '{gem_char}' in GemSet input string")
        return gem_set
    
    @classmethod
    def from_json(cls, data: list[int]):
        '''Initializes from array of ints, no data validity checks'''
        gs = cls()
        if data:
            gs[:] = data
        return gs
    
class Noble:
    def __init__(self, points=0, price=None):
        self.points = points # number of win points
        self.price = price
        if price is None:
            self.price = GemSet()

    @classmethod
    def from_str(cls, input_str):
        assert input_str[0] == '[' and input_str[2] == '|' and input_str[-1] == ']'
        points = int(input_str[1])
        price = GemSet.from_str(input_str[3:-1])
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

    @classmethod
    def from_str(cls, input_str):
        assert input_str[0] == '[' and input_str[3] == '|' and input_str[-1] == ']'
        gem = GEM_STR_TO_VAL[input_str[1]]
        points = int(input_str[2])
        price = GemSet.from_str(input_str[4:-1])
        return cls(gem, points, price)

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
CARDS = read_cards_from_csv('./pysplendor/cards.csv')

def print_cards():
    for level, deck in enumerate(CARDS):
        print(f'level {level}')
        print(','.join([f'"{card}"' for card in deck]))
    
class SplendorPlayerState:
    def __init__(self, id):
        self.id = id
        self.card_gems = GemSet() # gems of acquired cards
        self.gems = GemSet() # gems on the table
        self.hand_cards = []
        self.points = 0 # winning points from all aquired cards

    def __str__(self):
        s = f'player {self.id} | points: {self.points}\n'
        s += f'card gems: {self.card_gems}\n'
        s += f'gems: {self.gems}\n'
        cards = ' '.join([str(card) for card in self.hand_cards])
        s += f'hand: {cards}\n'
        return s
    
    def copy(self):
        cp = copy.copy(self) # shallow copy
        cp.card_gems = copy.copy(self.card_gems) # deep copy
        cp.gems = copy.copy(self.gems)
        cp.hand_cards = list(self.hand_cards)
        return cp

    @classmethod
    def from_json(cls, player_data):
        player = cls(player_data['id'])
        player.card_gems = GemSet.from_json(player_data.get('card_gems'))
        player.gems = GemSet.from_json(player_data.get('gems'))
        player.hand_cards = [Card.from_str(card) for card in player_data.get('hand_cards', [])]
        player.points = player_data.get('points', 0)
        return player

class Action:
    def __init__(self, action_type: ActionType, gems: GemSet = None, level: int = None, pos:int = None):
        self.type: ActionType = action_type
        self.gems: GemSet = gems
        self.level: int = level
        self.pos: int = pos

    def __str__(self):
        if self.type == ActionType.skip:
            return self.type
        if self.type == ActionType.take: 
            return f'{self.type}{self.gems}'
        elif self.type == ActionType.reserve or self.type == ActionType.purchase or self.type == ActionType.new_table_card:
            return f'{self.type}{self.level}n{self.pos}'
        elif self.type == ActionType.purchase_hand:
            return f'{self.type}{self.pos}'
        else:
            raise ValueError('Unknown action type')

    @classmethod
    def from_str(cls, input_str):
        if not input_str:
            raise ValueError("Action string is empty")

        action_type = input_str[0]
        gems = None
        level = -1
        pos = -1

        if action_type == ActionType.take:
            gems = GemSet.from_str(input_str[1:])
        elif action_type in (ActionType.reserve, ActionType.purchase):
            separator_pos = input_str.find('n')
            if separator_pos == -1 or separator_pos == 0 or separator_pos == len(input_str) - 1:
                raise ValueError(f"Invalid format for level and position in action {input_str}")
            level = int(input_str[1:separator_pos])
            pos = int(input_str[separator_pos + 1:])
        elif action_type in (ActionType.purchase_hand, ActionType.new_table_card):
            pos = int(input_str[1:])

        return cls(action_type, gems, level, pos)

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
 
DEFAULT_RULES = {n: SplendorGameRules(n) for n in range(2, 5)}
CHANCE_PLAYER = -1

class SplendorGameState(GameState):
    '''Knows game rules)'''

    def __init__(self, num_players, rules: SplendorGameRules = None):
        self.rules = rules
        if rules is None:
            self.rules = DEFAULT_RULES[num_players]
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
        self.players = [SplendorPlayerState(n) for n in range(num_players)]

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
        s = f'round: {self.round} player to move: {self.active_player()}\n'
        nobles = ' '.join([str(n) for n in self.nobles])
        s += f'nobles: {nobles}\n'

        for n, card_list in enumerate(reversed(self.cards)):
            cards = ' '.join([str(c) for c in card_list])
            s += f"{CARD_LEVELS - n - 1}: {cards}\n"

        s += f'gems: {self.gems}\n'

        for player in self.players:
            s += str(player)

        return s 

    @staticmethod
    def from_json(data):
        num_players = len(data['players'])
        game_state = SplendorGameState(num_players)
        game_state.round = data.get('round', 0)
        game_state.player_to_move = data.get('player_to_move', 0)
        game_state.skips = data.get('skips', 0)
        game_state.table_card_needed = data.get('table_card_needed', False)
        game_state.deck_level = data.get('deck_level', 0)
        game_state.nobles = [Noble.from_str(noble) for noble in data['nobles']]
        game_state.cards = [[Card.from_str(card) for card in level] for level in data['cards']]
        game_state.decks = [[Card.from_str(card) for card in level] for level in data['decks']]
        game_state.gems = GemSet.from_json(data['gems'])
        game_state.players = [SplendorPlayerState.from_json(player) for player in data['players']]
        return game_state

    def apply_action(self, action: Action):
        player = self.players[self.player_to_move]
        if self.player_to_move == 0:
            self.skips = 0 # resets in the beginning of each round

        if action.type == ActionType.skip:
            self.skips += 1
            self._increment_player_to_move()

        elif action.type == ActionType.take:
            unique_gems_take = action.gems.unique()
            total_gems_take = sum(action.gems)

            if total_gems_take > self.rules.max_gems_take:
                raise ValueError(f"Can't take more than {self.rules.max_gems_take} gems")

            if unique_gems_take == 1:  # All same color
                color = next((n for n, gem in enumerate(action.gems) if gem > 0), None)
                if self.gems[color] < self.rules.min_same_gems_stack:
                    raise ValueError(f"Should be at least {self.rules.min_same_gems_stack} gems in stack")
                if action.gems[color] > self.rules.max_same_gems_take:
                    raise ValueError(f"Can't take more than {self.rules.max_same_gems_take} identical gems")

            if unique_gems_take > 1 and unique_gems_take != total_gems_take:
                raise ValueError("You can either take all identical or all different gems")

            if sum(player.gems) + total_gems_take > self.rules.max_player_gems:
                raise ValueError(f"Player can't have more than {self.rules.max_player_gems} gems")

            for gem, gem_count in enumerate(action.gems):
                if gem_count > 0:
                    if gem == GOLD_GEM:
                        raise ValueError("You are not allowed to take gold gem")
                    if gem_count > self.gems[gem]:
                        raise ValueError(f"Not enough {gem} gems on table")
                    player.gems[gem] += gem_count
                    self.gems[gem] -= gem_count

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
            # deck_level to add a new card is stored in the game state, not in the action
            if self.deck_level < 0 or self.deck_level >= CARD_LEVELS:
                raise ValueError('Invalid deck level {}'.format(self.deck_level))
            if action.pos < 0 or action.pos >= len(self.decks[self.deck_level]):
                raise ValueError('Invalid card position {}'.format(action.pos))
            
            new_card = self.decks[self.deck_level].pop(action.pos)
            self.cards[self.deck_level].append(new_card)
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
            return CHANCE_PLAYER
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

        



