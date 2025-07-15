import random, copy, csv, json

from .game_state import GameState, CHANCE_PLAYER

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

class GemSet:
    '''Count/price for each gem'''
    def __init__(self):
        self._counts = [0] * NUM_GEMS

    def __getitem__(self, gem):
        return self._counts[gem]

    def __setitem__(self, gem, count):
        self._counts[gem] = count

    def __str__(self):
        return ''.join([GEM_STR[gem] + str(count) for gem, count in enumerate(self._counts) if count > 0])

    def copy(self):
        new_gem_set = GemSet()
        new_gem_set._counts = self._counts.copy()
        return new_gem_set

    def unique(self):
        return sum([1 for x in self._counts if x > 0])

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
# CARDS = read_cards_from_csv('./assets/cards.csv')

NOBLES = tuple(map(Noble.from_str, ['[3|r4g4]', '[3|g4b4]', '[3|b4w4]', '[3|w4k4]', '[3|k4r4]', '[3|r3g3b3]', '[3|b3g3w3]', '[3|b3w3k3]', '[3|w3k3r3]', '[3|k3r3g3]']))
CARDS = (
        tuple(map(Card.from_str, ["[k0|r1g1b1w1]","[k0|r1g1b2w1]","[k0|r1b2w2]","[k0|r3g1k1]","[k0|r1g2]","[k0|g2w2]","[k0|g3]","[k1|b4]","[b0|r1g1w1k1]","[b0|r2g1w1k1]","[b0|r2g2w1]","[b0|r1g3b1]","[b0|w1k2]","[b0|g2k2]","[b0|k3]","[b1|r4]","[w0|r1g1b1k1]","[w0|r1g2b1k1]","[w0|g2b2k1]","[w0|b1w3k1]","[w0|r2k1]","[w0|b2k2]","[w0|b3]","[w1|g4]","[g0|r1b1w1k1]","[g0|r1b1w1k2]","[g0|r2b1k2]","[g0|g1b3w1]","[g0|b1w2]","[g0|r2b2]","[g0|r3]","[g1|k4]","[r0|g1b1w1k1]","[r0|g1b1w2k1]","[r0|g1w2k2]","[r0|r1w1k3]","[r0|g1b2]","[r0|r2w2]","[r0|w3]","[r1|w4]"])),
        tuple(map(Card.from_str, ["[k1|g2b2w3]","[k1|g3w3k2]","[k2|r2g4b1]","[k2|r3g5]","[k2|w5]","[k3|k6]","[b1|r3g2b2]","[b1|g3b2k3]","[b2|b3w5]","[b2|r1w2k4]","[b2|b5]","[b3|b6]","[w1|r2g3k2]","[w1|r3b3w2]","[w2|r4g1k2]","[w2|r5k3]","[w2|r5]","[w3|w6]","[g1|r3g2w3]","[g1|b3w2k2]","[g2|b2w4k1]","[g2|g3b5]","[g2|g5]","[g3|g6]","[r1|r2w2k3]","[r1|r2b3k3]","[r2|g2b4w1]","[r2|w3k5]","[r2|k5]","[r3|r6]"])),
        tuple(map(Card.from_str, ["[k3|r3g5b3w3]","[k4|r7]","[k4|r6g3k3]","[k5|r7k3]","[b3|r3g3w3k5]","[b4|w7]","[b4|b3w6k3]","[b5|b3w7]","[w3|r5g3b3k3]","[w4|k7]","[w4|r3w3k6]","[w5|w3k7]","[g3|r3b3w5k3]","[g4|b7]","[g4|g3b6w3]","[g5|g3b7]","[r3|g3b5w3k3]","[r4|g7]","[r4|r3g6b3]","[r5|r3g7]"]))
    )

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
        cp.card_gems = self.card_gems.copy() # deep copy
        cp.gems = self.gems.copy()
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
        elif self.type == ActionType.reserve or self.type == ActionType.purchase:
            return f'{self.type}{self.level}n{self.pos}'
        elif self.type == ActionType.purchase_hand or self.type == ActionType.new_table_card:
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

class ActionError:
    NONE = "All good"
    INVALID_ACTION_ID = "Invalid action id"
    CANNOT_TAKE_MORE_THAN_MAX_GEMS = "Can't take more than maximum allowed gems"
    NOT_ENOUGH_GEMS_IN_STACK = "Not enough gems in stack"
    CANNOT_TAKE_MORE_THAN_MAX_IDENTICAL_GEMS = "Can't take more than maximum identical gems"
    MUST_TAKE_ALL_IDENTICAL_OR_ALL_DIFFERENT = "Must take all identical or all different gems"
    PLAYER_CANNOT_HAVE_MORE_GEMS = "Player can't have more gems"
    CANNOT_TAKE_GOLD_GEM = "Can't take gold gem"
    NOT_ENOUGH_GEMS_ON_TABLE = "Not enough gems on table"
    INVALID_DECK_LEVEL = "Invalid deck level"
    INVALID_CARD_POSITION = "Invalid card position"
    PLAYER_CANNOT_RESERVE_MORE_CARDS = "Player can't reserve more cards"
    PLAYER_CANNOT_AFFORD_CARD = "Player can't afford the card"
    GAME_REQUIRES_NEW_TABLE_CARD = "Game requires new table card"
    GAME_DOES_NOT_NEED_NEW_TABLE_CARD = "Game doesn't need new table card"
    INVALID_ACTION_TYPE = "Invalid action type"

# full action list including chance node actions
ACTIONS_STR = ["s", # skip move
    "tr2","tg2","tb2","tw2","tk2","tr1g1b1","tr1g1w1","tr1g1k1","tr1b1w1","tr1b1k1","tr1w1k1","tg1b1w1","tg1b1k1","tg1w1k1","tb1w1k1", # take 
    "r0n0","r0n1","r0n2","r0n3","r1n0","r1n1","r1n2","r1n3","r2n0","r2n1","r2n2","r2n3", # reserve
    "p0n0","p0n1","p0n2","p0n3","p1n0","p1n1","p1n2","p1n3","p2n0","p2n1","p2n2","p2n3", # purchase
    "h0","h1","h2", # purchase from hand
    "c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30","c31","c32","c33","c34","c35","c36","c37","c38","c39" # new card (choice node actions)
]
# all possible actions that are avialable to players
PLAYER_ACTIONS_STR = ["s","tr2","tg2","tb2","tw2","tk2","tr1g1b1","tr1g1w1","tr1g1k1","tr1b1w1","tr1b1k1","tr1w1k1","tg1b1w1","tg1b1k1","tg1w1k1","tb1w1k1",
               "r0n0","r0n1","r0n2","r0n3","r1n0","r1n1","r1n2","r1n3","r2n0","r2n1","r2n2","r2n3",
               "p0n0","p0n1","p0n2","p0n3","p1n0","p1n1","p1n2","p1n3","p2n0","p2n1","p2n2","p2n3",
               "h0","h1","h2"]


ACTIONS = [Action.from_str(a) for a in ACTIONS_STR]
ACTION_IDS = {s: idx for idx, s in enumerate(ACTIONS_STR)}

def init_action_type_ids(actions: list[Action]) -> dict[str, list[int]]:
    """Initialize a dictionary mapping action types to their corresponding action IDs."""
    ids = {}
    for n, action in enumerate(actions):
        action_type = action.type
        if action_type not in ids:
            ids[action_type] = []
        ids[action_type].append(n)
    return ids

ACTION_TYPE_IDS = init_action_type_ids(ACTIONS)  # indices for all action types

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
        cp.gems = self.gems.copy()
        cp.players = [player.copy() for player in self.players]
        return cp

    def move_num(self):
        return self.round

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

    def apply_action(self, action_id: int):
        is_valid, error = self._verify_action(action_id)
        if not is_valid:
            raise ValueError(error)

        action = ACTIONS[action_id]
        player = self.players[self.player_to_move]

        if self.player_to_move == 0:
            self.skips = 0

        if action.type == ActionType.skip:
            self.skips += 1
            self._increment_player_to_move()

        elif action.type == ActionType.take:
            for gem in range(NUM_GEMS):
                gem_count = action.gems[gem]
                if gem_count > 0:
                    player.gems[gem] += gem_count
                    self.gems[gem] -= gem_count
            self._increment_player_to_move()

        elif action.type == ActionType.reserve:
            card = self.cards[action.level].pop(action.pos)
            self._set_table_card_needed(action.level)

            player.hand_cards.append(card)
            if self.gems[GOLD_GEM] > 0:
                player.gems[GOLD_GEM] += 1
                self.gems[GOLD_GEM] -= 1

            self._increment_player_to_move()

        elif action.type == ActionType.purchase:
            card = self.cards[action.level].pop(action.pos)
            self._purchase_card(player, card)
            self._set_table_card_needed(action.level)
            self._get_noble(player)
            self._increment_player_to_move()

        elif action.type == ActionType.purchase_hand:
            card = player.hand_cards.pop(action.pos)
            self._purchase_card(player, card)
            self._get_noble(player)
            self._increment_player_to_move()

        elif action.type == ActionType.new_table_card:
            new_card = self.decks[self.deck_level].pop(action.pos)
            self.cards[self.deck_level].append(new_card)
            self.table_card_needed = False

        else:
            raise ValueError(ActionError.INVALID_ACTION_TYPE)

    def _verify_action(self, action_id: int) -> tuple[bool, ActionError]:
        if action_id < 0 or action_id >= len(ACTIONS):
            return False, ActionError.INVALID_ACTION_ID

        action = ACTIONS[action_id]

        if self.table_card_needed:
            if action.type != ActionType.new_table_card:
                return False, ActionError.GAME_REQUIRES_NEW_TABLE_CARD
            if self.deck_level < 0 or self.deck_level >= len(self.decks):
                return False, ActionError.INVALID_DECK_LEVEL
            if action.pos < 0 or action.pos >= len(self.decks[self.deck_level]):
                return False, ActionError.INVALID_CARD_POSITION
            return True, ActionError.NONE
        else:
            if action.type == ActionType.new_table_card:
                return False, ActionError.GAME_DOES_NOT_NEED_NEW_TABLE_CARD

        if action.type == ActionType.skip:
            return True, ActionError.NONE

        player = self.players[self.player_to_move]

        if action.type == ActionType.take:
            unique_gems_take = action.gems.unique()
            total_gems_take = sum(action.gems)

            if total_gems_take > self.rules.max_gems_take:
                return False, ActionError.CANNOT_TAKE_MORE_THAN_MAX_GEMS
            if unique_gems_take == 1:
                gem_iter = next((i for i, x in enumerate(action.gems) if x > 0), None)
                color = gem_iter
                if self.gems[color] < self.rules.min_same_gems_stack:
                    return False, ActionError.NOT_ENOUGH_GEMS_IN_STACK
                if action.gems[gem_iter] > self.rules.max_same_gems_take:
                    return False, ActionError.CANNOT_TAKE_MORE_THAN_MAX_IDENTICAL_GEMS
            if unique_gems_take > 1 and unique_gems_take != total_gems_take:
                return False, ActionError.MUST_TAKE_ALL_IDENTICAL_OR_ALL_DIFFERENT
            if sum(player.gems) + total_gems_take > self.rules.max_player_gems:
                return False, ActionError.PLAYER_CANNOT_HAVE_MORE_GEMS

            for gem in range(NUM_GEMS):
                gem_count = action.gems[gem]
                if gem_count > 0:
                    if gem == GOLD_GEM:
                        return False, ActionError.CANNOT_TAKE_GOLD_GEM
                    if gem_count > self.gems[gem]:
                        return False, ActionError.NOT_ENOUGH_GEMS_ON_TABLE
            return True, ActionError.NONE

        if action.type == ActionType.reserve:
            if action.level < 0 or action.level >= len(self.cards):
                return False, ActionError.INVALID_DECK_LEVEL
            if action.pos < 0 or action.pos >= len(self.cards[action.level]):
                return False, ActionError.INVALID_CARD_POSITION
            if len(player.hand_cards) >= self.rules.max_hand_cards:
                return False, ActionError.PLAYER_CANNOT_RESERVE_MORE_CARDS
            return True, ActionError.NONE

        if action.type == ActionType.purchase:
            if action.level < 0 or action.level >= len(self.cards):
                return False, ActionError.INVALID_DECK_LEVEL
            if action.pos < 0 or action.pos >= len(self.cards[action.level]):
                return False, ActionError.INVALID_CARD_POSITION
            card = self.cards[action.level][action.pos]
            if not self._player_can_afford_card(player, card):
                return False, ActionError.PLAYER_CANNOT_AFFORD_CARD
            return True, ActionError.NONE

        if action.type == ActionType.purchase_hand:
            if action.pos < 0 or action.pos >= len(player.hand_cards):
                return False, ActionError.INVALID_CARD_POSITION
            card = player.hand_cards[action.pos]
            if not self._player_can_afford_card(player, card):
                return False, ActionError.PLAYER_CANNOT_AFFORD_CARD
            return True, ActionError.NONE

        return False, ActionError.INVALID_ACTION_TYPE

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

    # def get_actions(self) -> list[int]:
    #     actions = []
    #     for action in range(len(ACTIONS)):
    #         valid, err = self._verify_action(action)
    #         if valid:
    #             actions.append(action)
    #     return actions

    def get_actions(self) -> list[int]:
        """Returns a list of valid action IDs for the current game state."""
        actions = []

        if self.table_card_needed:
            # Move of the gods of randomness, no player is involved
            action_ids = ACTION_TYPE_IDS[ActionType.new_table_card]
            deck_size = len(self.decks[self.deck_level])
            actions.extend(action_ids[:deck_size])
            return actions

        player = self.players[self.player_to_move]

        # 1. Take new gems
        # Two same gems
        player_gems_sum = sum(player.gems)
        if player_gems_sum < self.rules.max_player_gems - self.rules.max_same_gems_take:
            action_ids = ACTION_TYPE_IDS[ActionType.take]
            for gem in range(NUM_GEMS - 1):  # Exclude gold gem
                if self.gems[gem] >= self.rules.min_same_gems_stack:
                    actions.append(action_ids[gem])  # take same gems come first in the action list

        # Three distinct gems
        if player_gems_sum <= self.rules.max_player_gems - self.rules.max_gems_take:
            action_ids = ACTION_TYPE_IDS[ActionType.take]
            for n in range(NUM_GEMS - 1, len(action_ids)):  # distinct gem combinations start after single gem takes
                action = ACTIONS[action_ids[n]]
                can_take = True
                for gem in range(NUM_GEMS - 1):
                    if action.gems[gem] > self.gems[gem]:
                        can_take = False
                        break
                if can_take:
                    actions.append(action_ids[n])

        # 2. Reserve a card
        if len(player.hand_cards) < self.rules.max_hand_cards:
            for action_id in ACTION_TYPE_IDS[ActionType.reserve]:
                action = ACTIONS[action_id]
                if action.pos < len(self.cards[action.level]):
                    actions.append(action_id)

        # 3. Purchase a card from the table
        for level in range(len(self.cards)):
            for pos in range(len(self.cards[level])):
                if self._player_can_afford_card(player, self.cards[level][pos]):
                    actions.append(ACTION_TYPE_IDS[ActionType.purchase][level * self.rules.max_open_cards + pos])

        # 4. Purchase a card from the hand
        if player.hand_cards:
            for pos in range(len(player.hand_cards)):
                if self._player_can_afford_card(player, player.hand_cards[pos]):
                    actions.append(ACTION_TYPE_IDS[ActionType.purchase_hand][pos])

        # 0. Skip the move
        if not actions:
            actions.append(ACTION_TYPE_IDS[ActionType.skip][0])

        return actions

    def active_player(self):
        if self.table_card_needed:
            return CHANCE_PLAYER
        return self.player_to_move

    def is_terminal(self):
        # The game stops if all players skipped the move in the current round
        if self.skips >= len(self.players):
            return True

        # Or one of them got the required number of win points, but only after ensuring all players have had equal turns
        if self.player_to_move == 0:  # Ensures all players have had equal turns
            for player in self.players:
                if player.points >= self.rules.win_points:
                    return True

        return False

    def rewards(self):
        scores = [0.0] * len(self.players)
        winners = self.get_winners()
        for id in winners:
            scores[id] = 1.0 / len(winners)
        return scores

    def get_winners(self):
        max_points = max([p.points for p in self.players])
        if max_points < self.rules.win_points: # no winners
            return []
        
        candidate_ids = [player.id for player in self.players if player.points >= max_points]

        # If there's only one candidate, they are the winner
        if len(candidate_ids) <= 1:
            return candidate_ids

        # In case of a tie, the player with the fewest development cards wins
        card_counts = [sum(self.players[player_id].card_gems) for player_id in candidate_ids] # Sum of all card gems represents the number of development cards
        min_cards = min(card_counts)
        winners = [id for id in candidate_ids if card_counts[id] <= min_cards] # there still may be a tie though

        return winners

if __name__ == '__main__':
    print_cards()

        



