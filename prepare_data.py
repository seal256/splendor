import json
import numpy as np

from pysplendor.game import Trajectory, traj_iter

from pysplendor.splendor import SplendorGameState, SplendorPlayerState, Action, GemSet, Card, Noble, NUM_GEMS, CARD_LEVELS, DEFAULT_RULES, CHANCE_PLAYER

    
# all possible actions that are avialable to players
ALL_ACTIONS = ["s","tr2","tg2","tb2","tw2","tk2","tr1g1b1","tr1g1w1","tr1g1k1","tr1b1w1","tr1b1k1","tr1w1k1","tg1b1w1","tg1b1k1","tg1w1k1","tb1w1k1",
               "r0n0","r0n1","r0n2","r0n3","r1n0","r1n1","r1n2","r1n3","r2n0","r2n1","r2n2","r2n3",
               "p0n0","p0n1","p0n2","p0n3","p1n0","p1n1","p1n2","p1n3","p2n0","p2n1","p2n2","p2n3",
               "h0","h1","h2"]
ACTION_ID = {a: id for id, a in enumerate(ALL_ACTIONS)}


NUM_PLAYERS = 2
RULES = DEFAULT_RULES.get(NUM_PLAYERS)
# Only valid for 2 player games!

MAX_GEMS = RULES.max_gems 
def gems_to_vec(gems: GemSet, max_gems = MAX_GEMS):
    vec = [0] * (NUM_GEMS * (max_gems + 1))
    for gem, count in enumerate(gems):
        vec[gem * max_gems + count] = 1
    return vec

MAX_CARD_POINTS = 5
def card_to_vec(card: Card):
    gem = [0] * NUM_GEMS
    gem[card.gem] = 1
    points = [0] * (MAX_CARD_POINTS + 1)
    points[card.points] = 1
    price = gems_to_vec(card.price)
    return gem + points + price

CARD_VEC_LEN = len(card_to_vec(Card.from_str("[k0|r1g1b1w1]")))


def noble_to_vec(noble: Noble):
    # a noble always gives you 3 points, no need to encode them
    return gems_to_vec(noble.price)

NOBLE_VEC_LEN = len(noble_to_vec(Noble.from_str("[3|r3g3b3]")))

MAX_HAND_CARDS = RULES.max_hand_cards
WIN_POINTS = RULES.win_points
MAX_CARDS = 6 # this is arbitrary and can be exceeded!
def player_to_vec(player: SplendorPlayerState):
    card_gems = gems_to_vec(player.card_gems, max_gems=MAX_CARDS)
    table_gems = gems_to_vec(player.gems)
    hand_cards = [0] * MAX_HAND_CARDS * CARD_VEC_LEN
    for n, card in enumerate(player.hand_cards):
        hand_cards[n * CARD_VEC_LEN: (n+1) * CARD_VEC_LEN] = card_to_vec(card)
    points = [0] * (WIN_POINTS + 1)
    points[player.points] = 1
    
    return card_gems + table_gems + hand_cards + points


MAX_OPEN_CARDS = RULES.max_open_cards
MAX_NOBLES = RULES.max_nobles
def state_to_vec(state: SplendorGameState):
    assert NUM_PLAYERS == len(state.players)
    # Encodes only open game information (e.g. cards in decks are not encoded)

    nobles = [0] * (NOBLE_VEC_LEN * MAX_NOBLES)
    for n, noble in enumerate(state.nobles):
        nobles[n * NOBLE_VEC_LEN: (n+1) * NOBLE_VEC_LEN] = noble_to_vec(noble)

    table_cards = [0] * (CARD_VEC_LEN * MAX_OPEN_CARDS * CARD_LEVELS)
    for level in range(CARD_LEVELS):
        for ncard, card in enumerate(state.cards[level]):
            pos = (level * MAX_OPEN_CARDS + ncard) * CARD_VEC_LEN
            table_cards[pos: pos + CARD_VEC_LEN] = card_to_vec(card)

    players = []
    active_player = state.active_player()
    for n in range(NUM_PLAYERS):
        player_idx = (n + active_player) % NUM_PLAYERS # always start feature vector from the active player
        players += player_to_vec(state.players[player_idx])

    table_gems = gems_to_vec(state.gems)

    return nobles + table_cards + players + table_gems


def prepare_data(traj_file, data_fname_prefix):
    '''Creates 3 files with states, actions and rewards'''
    states = []
    actions = []
    rewards = []
    for traj in traj_iter(traj_file):
        state = traj.initial_state.copy()
        for action in traj.actions:
            if state.active_player() != CHANCE_PLAYER: # ignore chance nodes
                state_vec = state_to_vec(state)
                states.append(np.packbits(state_vec)) # compressed bytes
                actions.append(ACTION_ID[action]) # ints
                rewards.append(traj.rewards[state.active_player()]) # 0 or 1 

            state.apply_action(Action.from_str(action))

    np.save(data_fname_prefix + "_states.npy", np.array(states))
    np.save(data_fname_prefix + "_actions.npy", np.array(actions, dtype=np.uint8))
    np.save(data_fname_prefix + "_rewards.npy", np.array(rewards, dtype=np.uint8))



if __name__ == '__main__':
    prepare_data('./data/traj_dump.txt', './data/train/iter0')
