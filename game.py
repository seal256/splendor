from random import sample, shuffle, seed
import csv

# http://www.spacecowboys.fr/img/games/splendor/details/rules/Rules_Splendor_US.pdf
# in our implementation hand cards are visible to everyone

GEMS = ('r', 'g', 'b', 'w', 'k', 'y') # red, green, blue, white, black, yellow(gold)
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

    def items(self):
        return self.gems.items()

    def __str__(self):
        s = ''
        for gem in GEMS: # preserve gem order
            if gem in self.gems:
                count = self.get(gem)
                if count  > 0:
                    s += gem + str(count)
        return s

class Noble:
    def __init__(self, points=0, price=None):
        self.points = points # number of win points
        self.price = price
        if price is None:
            self.price = GemSet()

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

NOBLES = ()
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
        '''Returns false if player can\'t afford card'''
        gems_to_pay = GemSet()

        gold = self.gems.get('y') # gold available 
        for gem, price in card.price.items():
            to_pay = max(0, price - self.cards.get(gem))
            available = self.gems.get(gem) 
            if available < to_pay:
                pay_by_gold = to_pay - available # may be covered by gold
                if gold < pay_by_gold:
                    return False
                gems_to_pay.add('y', pay_by_gold)
                to_pay -= pay_by_gold
                gold -= pay_by_gold
            gems_to_pay.add(gem, to_pay)

        # if everything goes smooth, do actual payment
        for gem, to_pay in gems_to_pay.items():
            self.gems.add(gem, -to_pay)
            self.gem_count -= to_pay

        self.cards.add(card.gem, 1)
        self.points += card.points
        return True

class Action:
    take = ACTIONS[0] # take gems
    reserve = ACTIONS[1] # reserve card
    purchase = ACTIONS[2] # purchase table card
    purchase_hand = ACTIONS[3] # purchase hand card

    def __init__(self, action_str):
        self.parse(action_str)

    def parse(self, action_str):
        self.type = action_str[0] # should be one of ACTIONS
        self.gems = None
        self.pos = None

        if self.type == Action.take: 
            self.gems = [g for g in action_str[1:]]
        elif self.type == Action.reserve: 
            self.pos = self.scan_pos(action_str) # (level, pos)
        elif self.type == Action.purchase: 
            self.pos = self.scan_pos(action_str) 
        elif self.type == Action.purchase_hand: 
            assert len(action_str) == 2
            self.pos = int(action_str[1]) # single int -- number of hand card
        else:
            raise AttributeError('Invalid action type {} (in action {})'.format(self.type, action_str))

    def scan_pos(self, action_str):
        assert len(action_str) == 3
        level = int(action_str[1])
        pos = int(action_str[2])
        return level, pos

class SplendorGameRules:
    def __init__(self, num_players):
        seed(1828)
        self.num_players = num_players
        self.max_open_cards = 4 # open cards on table
        self.max_hand_cards = 3 # max cards in player hand
        self.win_points = 15
        self.max_player_gems = 10
        self.max_nobles = self.num_players + 1
        self.max_gems_take = 3 # max gems to take
        self.max_same_gems_take = 2 # max same color gems to take
        self.max_gold = 5
        self.max_gems = 7 # max same color gems on table (except gold)
        if self.num_players < 4:
            self.max_gems = 2 + self.num_players
 
class SplendorGameState:
    '''Knows game rules)'''
    def __init__(self, player_names, rules):
        assert rules.num_players == len(player_names)

        self.rules = rules
        self.num_moves = 0
        self.player_to_move = 0

        # init nobles TODO
        #self.nobles = sample(NOBLES, self.rules.max_nobles)
        self.nobles = ()

        # init decks and cards 
        self.decks = []
        self.cards = [] # open cards on table
        for level in range(CARD_LEVELS):
            cards = list(CARDS[level])
            shuffle(cards)
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

    def get_table_card(self, level, pos):
        return self.cards[level-1][pos-1]

    def new_table_card(self, level, pos):
        '''Put new card on table if player reserved/purchased card'''
        new_card = None
        if len(self.decks[level]) > 0:
            new_card = self.decks[level].pop()
        self.cards[level-1][pos-1] = new_card


    def action(self, action_str):
        action = Action(action_str)
        player = self.players[self.player_to_move]

        if action.type == Action.take: 
            gems = action.gems
            unique_gems = set(gems)
            if len(gems) > self.rules.max_gems_take:
                raise AttributeError('Invalid gem amount {} (in action {})'.format(len(gems), action_str))
            if len(unique_gems) == 1 and len(gems) != self.rules.max_same_gems_take: 
                raise AttributeError('Invalid amount of identical gems (in action {})'.format(action_str))
            if len(unique_gems) > 1 and len(unique_gems) != len(gems): 
                raise AttributeError('Invalid amount of identical gems (in action {})'.format(action_str))
            if player.gem_count + len(gems) > self.rules.max_player_gems:
                raise AttributeError('Player will receive more than {} gems (in action {})'.format(self.rules.max_player_gems, action_str))

            for gem in gems:
                if gem not in GEMS:
                    raise AttributeError('Invalid gem ({}) in action {}'.format(gem, action_str))
                if gem == 'y':
                    raise AttributeError('You are not allowed to take gold gem (in action {})'.format(action_str))
                if self.gems.get(gem) == 0:
                    raise AttributeError('Not inough {} gems on table (in action {})'.format(gem, action_str))
                
                player.gems.add(gem, 1)
                self.gems.add(gem, -1)

        elif action.type == Action.reserve: 
            level, pos = action.pos # pos=0 means blind reserve from deck
            assert level > 0 and level <= CARD_LEVELS
            assert pos >= 0 and pos < len(self.cards[level])

            if len(player.hand) >= self.rules.max_hand_cards:
                raise AttributeError('Player can\'t reserve with {} cards in hand (in action {})'.format(len(player.hand), action_str))

            if pos > 0:
                card = self.get_table_card(level, pos)
                player.hand_cards.append(card)
                self.new_table_card(level, pos)
            if pos == 0:
                assert len(self.decks[level]) > 0
                new_card = self.decks[level].pop()
                player.hand_cards.append(new_card)

        elif action.type == Action.purchase: 
            level, pos = action.pos 
            assert level > 0 and level <= CARD_LEVELS
            assert pos > 0 and pos < self.rules.max_open_cards

            card = self.get_table_card(level, pos)
            if not player.purchase_card(card):
                raise AttributeError('Player can\'t afford card (in action {})'.format(action_str))
            self.new_table_card(level, pos)

        elif action.type == Action.purchase_hand: 
            pos = action.pos # position of card in hand
            assert pos > 0 and pos < self.rules.max_hand_cards

            card = player.hand_cards[pos]
            if not player.purchase_card(card):
                raise AttributeError('Player can\'t afford card (in action {})'.format(action_str))
            player.pop(pos) # remove card from hand
        else:
            raise AttributeError('Invalid action type {} (in action {})'.format(action.type, action_str))

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
        


        



