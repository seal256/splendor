import random
import csv

from game import GameState

GOLD_GEM = 'y'
GEMS = ('r', 'g', 'b', 'w', 'k', GOLD_GEM) # red, green, blue, white, black, yellow(gold)
ACTIONS = ('t', 'r', 'p', 'h') # take gems, reserve card, purchase card, purchase hand card
CARD_LEVELS = 3

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
    take = ACTIONS[0] # take gems
    reserve = ACTIONS[1] # reserve card
    purchase = ACTIONS[2] # purchase table card
    purchase_hand = ACTIONS[3] # purchase hand card

    def __init__(self, action_type, gems, pos):
        self.type = action_type
        self.gems = gems
        self.pos = pos

    def __str__(self):
        if self.type == Action.take: 
            return Action.take + ''.join(self.gems)
        else:
            return self.type.join(map(str, self.pos))

    @classmethod
    def from_str(cls, action_str):
        try:
            action_type, gems, pos = Action.parse(action_str)
            return cls(action_type, gems, pos)
        except Exception:
            raise AttributeError('Invalid action string {}'.format(action_str))

    @staticmethod
    def parse(action_str):
        action_type = action_str[0] # should be one of ACTIONS
        gems = None
        pos = None

        if action_type == Action.take: 
            gems = [g for g in action_str[1:]]
        elif action_type == Action.reserve: 
            pos = Action.scan_pos(action_str) # (level, pos)
        elif action_type == Action.purchase: 
            pos = Action.scan_pos(action_str) 
        elif action_type == Action.purchase_hand: 
            assert len(action_str) == 2
            pos = (int(action_str[1]) - 1,) # single int -- position of hand card
        else:
            raise AttributeError('Invalid action type {} (in action {})'.format(action_type, action_str))
        
        return action_type, gems, pos

    @staticmethod
    def scan_pos(action_str):
        assert len(action_str) == 3
        level = int(action_str[1]) - 1
        pos = int(action_str[2]) - 1
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
    
    def __init__(self, player_names, rules):
        assert rules.num_players == len(player_names)

        self.rules = rules
        self.num_moves = 0
        self.player_to_move = 0

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

    def new_table_card(self, level, pos):
        '''Put new card on table if player reserved/purchased card'''
        new_card = None
        if self.decks[level]:
            new_card = self.decks[level].pop()
        self.cards[level][pos] = new_card

    def action(self, action):
        player = self.players[self.player_to_move]

        if action.type == Action.take: 
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

        elif action.type == Action.reserve: 
            level, pos = action.pos
            if level < 0 or level >= CARD_LEVELS:
                raise AttributeError('Invalid deck level {}'.format(level + 1))
            if pos < -1 or pos >= len(self.cards[level]):
                raise AttributeError('Invalid card position {}'.format(pos + 1))
            if len(player.hand_cards) >= self.rules.max_hand_cards:
                raise AttributeError('Player can\'t reserve more than {} cards'.format(self.rules.max_hand_cards))

            card = None
            if pos >= 0:
                card = self.cards[level][pos]
                if card is None:
                    raise AttributeError('Card already taken')
                self.new_table_card(level, pos)
            if pos == -1: # blind reserve from deck
                if not self.decks[level]:
                    raise AttributeError('Deck {} is empty'.format(level + 1))
                card = self.decks[level].pop()
            player.hand_cards.append(card)
            if self.gems.get(GOLD_GEM) > 0:
                player.gems.add(GOLD_GEM, 1)
                self.gems.add(GOLD_GEM, -1)

        elif action.type == Action.purchase: 
            level, pos = action.pos 
            if level < 0 or level >= CARD_LEVELS:
                raise AttributeError('Invalid deck level {}'.format(level + 1))
            if pos < 0 or pos >= self.rules.max_open_cards:
                raise AttributeError('Invalid card position {}'.format(pos + 1))

            card = self.cards[level][pos]
            if not player.purchase_card(card):
                raise AttributeError('Player can\'t afford the card')
            self.new_table_card(level, pos)

            player.get_noble(self.nobles) # try to get a noble

        elif action.type == Action.purchase_hand: 
            pos, = action.pos # position of card in hand
            if pos < 0 or pos >= len(player.hand_cards):
                raise AttributeError('Invalid card position in hand {}'.format(pos + 1))

            card = player.hand_cards[pos]
            if not player.purchase_card(card):
                raise AttributeError('Player can\'t afford card')
            player.hand_card.pop(pos) # remove card from hand

            player.get_noble(self.nobles) # try to get noble

        else:
            raise AttributeError('Invalid action type {}'.format(action.type))

        self.player_to_move = (self.player_to_move + 1) % self.rules.num_players
        if self.player_to_move == 0: # round end
            self.num_moves += 1

    def check_win(self):
        for player in self.players:
            if player.points >= self.rules.win_points:
                return True
        return False

    def best_player(self):
        '''Returns name of best player'''
        scores = [(player.score, player.name) for player in self.players]
        return sorted(scores, reverse=True)[0]



        



