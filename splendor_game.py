import random
import csv
from itertools import combinations

from game import GameState

GOLD_GEM = 'y'
GEMS = ('r', 'g', 'b', 'w', 'k', GOLD_GEM) # red, green, blue, white, black, yellow(gold)
CARD_LEVELS = 3

class ActionType:
    take = 't' # take gems
    reserve = 'r' # reserve card
    purchase = 'p' # purchase table card
    purchase_hand = 'h' # purchase hand card
    new_table_card = 'c' # new table card from deck. Performed by randomness rather than any of the players
   
class GemSet:
    '''Set of gems with count (or price) each'''
    def __init__(self):
        self.gems = {} # gem name (one of GEMS) : count

    def add(self, gem, count):
        new_count = self.gems.get(gem, 0) + count
        assert new_count >= 0
        self.gems[gem] = new_count

    def get(self, gem):
        return self.gems.get(gem, 0)

    def __setitem__(self, gem, count):
        assert count >= 0
        self.gems[gem] = count

    def items(self):
        return self.gems.items()

    def __str__(self):
        s = ''
        for gem in GEMS: # preserve gem order
            if gem in self.gems:
                count = self.get(gem)
                if count > 0:
                    s += gem + str(count)
        return s

    def shortage(self, price):
        '''Returns list of additional gems required. Gold is not taken into account.'''
        shortage = GemSet()
        for gem, count in price.items():
            assert gem is not GOLD_GEM
            diff = count - self.get(gem)
            if diff > 0:
                shortage.add(gem, diff)
        return shortage

    def count(self):
        '''Total count of all gems in set''' 
        total = 0
        for _, count in self.items():
            total += count
        return total

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
            gem = string[2*n + 3]
            count = int(string[2*n + 4])
            price.add(gem, count)
        return points, price

    @classmethod
    def from_str(cls, string):
        '''Alternative constructor from string'''
        points, price = Noble.parse_str(string)
        return cls(points, price)

    def __str__(self):
        return '[' + str(self.points) + '|' + str(self.price) + ']'

class Card:
    def __init__(self, gem='', points=0, price=None):
        self.gem = gem # title gem
        self.points = points # number of win points
        self.price = price 
        if price is None:
            self.price = GemSet()

    def __str__(self):
        return '[' + self.gem + str(self.points) + '|' + str(self.price) + ']'

def read_cards_from_csv(file_name):
    cards = [[], [], []]
    reader = csv.reader(open(file_name))
    next(reader, None) # skip header
    for line in reader:
        assert len(line) == 8
        card = Card()
        level = int(line[0]) - 1
        card.gem = line[1]
        card.points = int(line[2])
        for gem, amount in zip(GEMS[:-1], line[3:]):
            if len(amount) == 1:
                card.price.add(gem, int(amount))
        cards[level].append(card)
    return tuple([tuple(cards[n]) for n in range(CARD_LEVELS)])

NOBLES = tuple(map(Noble.from_str, ['[3|r4g4]', '[3|g4b4]', '[3|b4w4]', '[3|w4k4]', '[3|k4r4]',
    '[3|r3g3b3]', '[3|b3g3w3]', '[3|b3w3k3]', '[3|w3k3r3]', '[3|k3r3g3]']))
CARDS = read_cards_from_csv('cards.csv')

class SplendorPlayerState:
    def __init__(self, name):
        self.name = name
        self.cards = GemSet()
        self.gems = GemSet()
        self.hand_cards = []
        self.points = 0
        self.gem_count = 0

    def __str__(self):
        s = self.name + '|' + str(self.points) + '\n'
        s += 'cards:' + str(self.cards) + '\n'
        s += 'gems:' + str(self.gems) + '\n'
        s += 'hand:'
        for card in self.hand_cards:
            s += str(card)
        s += '\n'
        return s

    def can_afford_card(self, card):
        '''Checks if the player can afford the given card'''

        shortage = self.gems.shortage(card.price)
        gold = self.gems.get(GOLD_GEM) # gold available 
        gold_to_pay = shortage.count()
        return gold < gold_to_pay
    
    def purchase_card(self, card):
        '''Performes the card purchase. Returns false if player can\'t afford the card'''

        shortage = self.gems.shortage(card.price)
        gold = self.gems.get(GOLD_GEM) # gold available 
        gold_to_pay = shortage.count()
        if gold < gold_to_pay:
            return False

        # if player can afford the card, perform the payment
        if gold_to_pay > 0:
            self.gems.add(GOLD_GEM, -gold_to_pay)
        for gem, price in card.price.items():
            if gem in shortage.gems: # have to pay by gold => new gem count is zero
                self.gems[gem] = 0
            else:
                self.gems.add(gem, -price)
        self.gem_count -= card.price.count()

        self.cards.add(card.gem, 1)
        self.points += card.points
        return True

    def get_noble(self, nobles):
        '''Attempts to acquire a noble card'''
        
        noble_list = []
        for n, noble in enumerate(nobles):
            can_afford = True
            for gem, price in noble.price.items():
                if self.cards.get(gem) < price:
                    can_afford = False
                    break
            if can_afford:
                noble_list.append(n)
                break 
                # potentially more than one noble may be available for aquisition, 
                # but we ignore this possiblilty for simplicity
        
        if noble_list:
            noble = nobles.pop(noble_list[0])
            self.points += noble.points

class Action:
    def __init__(self, action_type: ActionType, gems: list[str] = None, level: int = None, pos:int = None):
        self.type: ActionType = action_type
        self.gems: list[str] = gems # list of gem symbols 
        self.level: int = level
        self.pos: int = pos

    def __str__(self):
        if self.type == ActionType.take: 
            return self.type + ''.join(self.gems)
        elif self.type == ActionType.reserve or self.type == ActionType.purchase or self.type == ActionType.new_table_card:
            return f'{self.type}{self.level}n{self.pos}'
        elif self.type == ActionType.purchase_hand:
            return f'{self.type}{self.level}'
        else:
            raise ValueError('Unknown action type')

    @classmethod
    def from_str(cls, action_str):
        try:
            action_type, gems, level, pos = Action.parse(action_str)
            return cls(action_type, gems, level, pos)
        except Exception:
            raise AttributeError('Invalid action string {}'.format(action_str))

    @staticmethod
    def parse(action_str):
        action_type = action_str[0] # should be one of ACTIONS
        gems = None
        level = None
        pos = None

        if action_type == ActionType.take: 
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
            raise AttributeError('Invalid action type {} (in action {})'.format(action_type, action_str))
        
        return action_type, gems, level, pos

    @staticmethod
    def scan_level_pos(action_str):
        parts = action_str.split('n') # seperator between the card level and position
        assert len(parts) == 2
        level = int(parts[0]) - 1
        pos = int(parts[1]) - 1
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
        self.num_moves = 0
        self.player_to_move = 0
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
            self.gems.add(gem, self.rules.max_gems)
        self.gems.add(GEMS[-1], self.rules.max_gold)

        # init players
        self.players = [SplendorPlayerState(name) for name in player_names]

    def __str__(self):
        s = 'move:' + str(self.num_moves) + ' player:' + str(self.player_to_move) + '\n'
        
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

        s += 'gems:' + str(self.gems) + '\n'

        for player in self.players:
            s += str(player)

        return s 

    def apply_action(self, action: Action):
        player = self.players[self.player_to_move]

        if action.type == ActionType.take: 
            gems = action.gems
            unique_gems = list(set(gems))
            if len(gems) > self.rules.max_gems_take:
                raise AttributeError('Can\'t take more than {} gems'.format(self.rules.max_gems_take))
            if len(unique_gems) == 1: # all same color
                if self.gems.get(unique_gems[0]) < self.rules.min_same_gems_stack:
                    raise AttributeError('Should be at least {} gems in stack'.format(self.rules.min_same_gems_stack))
                if len(gems) != 1 and len(gems) > self.rules.max_same_gems_take: 
                    raise AttributeError('Can\'t take more than {} identical gems'.format(self.rules.max_same_gems_take))
            if len(unique_gems) > 1 and len(unique_gems) != len(gems): 
                raise AttributeError('You can either take all identical or all different gems')
            if player.gem_count + len(gems) > self.rules.max_player_gems:
                raise AttributeError('Player can\'t have more than {} gems'.format(self.rules.max_player_gems))

            for gem in gems:
                if gem not in GEMS:
                    raise AttributeError('Invalid gem {}'.format(gem))
                if gem == GOLD_GEM:
                    raise AttributeError('You are not allowed to take gold ({}) gem'.format(GOLD_GEM))
                if self.gems.get(gem) == 0:
                    raise AttributeError('Not inough {} gems on table'.format(gem))
                
                player.gems.add(gem, 1)
                self.gems.add(gem, -1)
            
            self._increment_player_to_move()

        elif action.type == ActionType.reserve: 
            # level and pos indicate the position of the card to reserve on the table
            if action.level < 0 or action.level >= CARD_LEVELS:
                raise AttributeError('Invalid deck level {}'.format(action.level))
            if action.pos < 0 or action.pos >= len(self.cards[action.level]):
                raise AttributeError('Invalid card position {}'.format(action.pos))
            if len(player.hand_cards) >= self.rules.max_hand_cards:
                raise AttributeError('Player can\'t reserve more than {} cards'.format(self.rules.max_hand_cards))

            card = self.cards[action.level][action.pos]
            self._set_table_card_needed(action.level)
                
            # TODO: There's no option to blindly reserve from the deck
            # if action.pos == -1: # blind reserve from deck
            #     if not self.decks[action.level]:
            #         raise AttributeError('Deck {} is empty'.format(action.level))
            #     card = self.decks[action.level].pop()

            player.hand_cards.append(card)
            if self.gems.get(GOLD_GEM) > 0:
                player.gems.add(GOLD_GEM, 1)
                self.gems.add(GOLD_GEM, -1)

            self._increment_player_to_move()

        elif action.type == ActionType.purchase: 
            # level and pos indicate the position of the card to purchase on the table
            if action.level < 0 or action.level >= CARD_LEVELS:
                raise AttributeError('Invalid deck level {}'.format(action.level))
            if action.pos < 0 or action.pos >= self.rules.max_open_cards:
                raise AttributeError('Invalid card position {}'.format(action.pos))

            card = self.cards[action.level].pop(action.pos) # takes the card from the table
            if not player.purchase_card(card):
                raise AttributeError('Player can\'t afford the card')
            self._set_table_card_needed(action.level)

            player.get_noble(self.nobles) # try to get a noble
            self._increment_player_to_move()

        elif action.type == ActionType.purchase_hand: 
            # pos is the position of the card in hand
            if action.pos < 0 or action.pos >= len(player.hand_cards):
                raise AttributeError('Invalid card position in hand {}'.format(action.pos))

            card = player.hand_cards[action.pos]
            if not player.purchase_card(card):
                raise AttributeError('Player can\'t afford the card')
            player.hand_card.pop(action.pos) # remove card from hand

            player.get_noble(self.nobles) # try to get a noble
            self._increment_player_to_move()

        elif action.type == ActionType.new_table_card:
            # The game is in an intermediate state that requires to select a new random card from the deck.
            # This (strange) state is reuired to allow "choice" game nodes of search algorithms to work correctly

            if not self.table_card_needed:
                raise AttributeError('Game does not require a new table card at the moment')
            if action.level < 0 or action.level >= CARD_LEVELS:
                raise AttributeError('Invalid deck level {}'.format(action.level))
            if action.pos <= 0 or action.pos >= len(self.decks[action.level]):
                raise AttributeError('Invalid card position {}'.format(action.pos))
            
            new_card = self.decks[action.level].pop(action.pos)
            self.cards[action.level].append(new_card)
            self.table_card_needed = False
            # since this is not a player's action, we do not increment the active player's index

        else:
            raise AttributeError('Invalid action type {}'.format(action.type))

    def _increment_player_to_move(self):
        self.player_to_move = (self.player_to_move + 1) % self.rules.num_players
        if self.player_to_move == 0: # round end
            self.num_moves += 1

    def _set_table_card_needed(self, level):
        if self.decks[level]: # if the deck is not empty
            self.table_card_needed = True
            self.deck_level = level

    def get_actions(self) -> list[Action]:
        if self.table_card_needed:
            # move of the gods of randomness, no player is involved
            action_type = ActionType.new_table_card
            level = self.deck_level
            actions = [Action(action_type, gems=None, level=level, pos=n) for n in range(len(self.decks[level]))]
            return actions
        
        actions = []
        player: SplendorPlayerState = self.players[self.player_to_move]

        # 1. Take new gems
        # 2 same gems
        action_type = ActionType.take
        if player.gem_count < self.rules.max_player_gems - self.rules.max_same_gems_take: 
            for gem, count in self.gems.items():
                if count >= self.rules.min_same_gems_stack:
                    actions.append(Action(action_type, [gem] * self.rules.max_same_gems_take))
                    
        # 3 distinct gems
        if player.gem_count < self.rules.max_player_gems - self.rules.max_gems_take:
            available_gems = [g for g in GEMS[:-1] if self.gems.get(g) > 0]
            for comb_gems in combinations(available_gems, self.rules.max_gems_take):
                actions.append(Action(action_type, comb_gems))

        # 2. Reserve a card
        if len(player.hand_cards) < self.rules.max_hand_cards:
            action_type = ActionType.reserve
            for level in range(CARD_LEVELS):
                for pos in range(len(self.cards[level])):
                    actions.append(Action(action_type, gems=None, level=level, pos=pos))
                    

        # 3. Purchase a card from table
        action_type = ActionType.purchase
        for level in range(CARD_LEVELS):
            for pos, card in enumerate(self.cards[level]):
                if player.can_afford_card(card):
                    actions.append(Action(action_type, gems=None, level=level, pos=pos))

        # 4. Purchase a card from the hand
        if player.hand_cards:
            action_type = ActionType.purchase_hand
            for pos in range(len(player.hand_cards)):
                actions.append(Action(action_type, gems=None, level=None, pos=pos))

        return actions

    def active_player(self):
        if self.table_card_needed:
            return None
        return self.player_to_move

    def is_terminal(self):
        for player in self.players:
            if player.points >= self.rules.win_points:
                return True
        return False

    def rewards(self):
        '''Returns the number of win points for each player'''
        return [player.score for player in self.players]

    def best_player(self):
        '''Returns name of best player'''
        scores = [(player.score, player.name) for player in self.players]
        return sorted(scores, reverse=True)[0]



        



