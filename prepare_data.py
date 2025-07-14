import time
import numpy as np
import argparse

from pysplendor.game_state import CHANCE_PLAYER
from pysplendor.game import Trajectory, traj_loader
from pysplendor.splendor import SplendorGameState, SplendorPlayerState, PLAYER_ACTIONS_STR, GemSet, Card, Noble, NUM_GEMS, CARD_LEVELS, DEFAULT_RULES

class SplendorGameStateEncoder:
    '''Compresses game state into a "one-hot" style vector of 0-s and 1-s'''

    def __init__(self, num_players):
        self.num_players = num_players
        self.rules = DEFAULT_RULES.get(self.num_players)
        
        self.max_card_points = 5 # The maximum number of win points on splenodr cards
        self.max_cards = 6  # The maximum number of cards of one color that the player can aquire. In rare cases this can be exceeded!
        self.card_vec_len = len(self._card_to_vec(Card.from_str("[k0|r1g1b1w1]")))
        self.noble_vec_len = len(self._noble_to_vec(Noble.from_str("[3|r3g3b3]")))

    def _gems_to_vec(self, gems: GemSet, max_gems=None):
        if max_gems is None:
            max_gems = self.rules.max_gems
        vec = [0] * (NUM_GEMS * (max_gems + 1))
        for gem, count in enumerate(gems):
            count = min(max_gems, count) # this affects cards or gold counts exceeding the limit
            vec[gem * (max_gems + 1) + count] = 1
        return vec

    def _card_to_vec(self, card: Card):
        gem = [0] * NUM_GEMS
        gem[card.gem] = 1
        points = [0] * (self.max_card_points + 1)
        points[card.points] = 1
        price = self._gems_to_vec(card.price)
        return gem + points + price

    def _noble_to_vec(self, noble: Noble):
        # A noble always gives you 3 points, no need to encode them
        return self._gems_to_vec(noble.price)

    def _player_to_vec(self, player: SplendorPlayerState):
        card_gems = self._gems_to_vec(player.card_gems, max_gems=self.max_cards)
        table_gems = self._gems_to_vec(player.gems)
        hand_cards = [0] * self.rules.max_hand_cards * self.card_vec_len
        for n, card in enumerate(player.hand_cards):
            hand_cards[n * self.card_vec_len: (n+1) * self.card_vec_len] = self._card_to_vec(card)
        points = [0] * (self.rules.win_points + 1)
        points[min(player.points, self.rules.win_points)] = 1
        
        return card_gems + table_gems + hand_cards + points

    def state_to_vec(self, state: SplendorGameState):
        assert self.num_players == len(state.players)
        # Encodes only open game information (e.g., cards in decks are not encoded)

        nobles = [0] * (self.noble_vec_len * self.rules.max_nobles)
        for n, noble in enumerate(state.nobles):
            nobles[n * self.noble_vec_len: (n+1) * self.noble_vec_len] = self._noble_to_vec(noble)

        table_cards = [0] * (self.card_vec_len * self.rules.max_open_cards * CARD_LEVELS)
        for level in range(CARD_LEVELS):
            for ncard, card in enumerate(state.cards[level]):
                pos = (level * self.rules.max_open_cards + ncard) * self.card_vec_len
                table_cards[pos: pos + self.card_vec_len] = self._card_to_vec(card)

        players = []
        active_player = state.active_player()
        for n in range(self.num_players):
            player_idx = (n + active_player) % self.num_players  # Always start feature vector from the active player
            players += self._player_to_vec(state.players[player_idx])

        table_gems = self._gems_to_vec(state.gems)

        return nobles + table_cards + players + table_gems

def prepare_data(traj_file, data_fname_prefix, only_winner_moves = False, num_players = 2):
    '''Creates 3 files with states, actions and rewards
    
    Args:
        traj_file: Path to the trajectory file containing game record data
        data_fname_prefix: Prefix to use for generated output files
        only_winner_moves: Use only moves of the winning player (this makes _rewards files contain only 1-s)
    '''

    states = []
    actions = []
    rewards = []
    state_encoder = SplendorGameStateEncoder(num_players)

    start = time.perf_counter()
    num_states = 0
    num_traj = 0
    for traj in traj_loader(traj_file):
        state = traj.initial_state.copy()
        for move_num in range(len(traj.actions)):
            action = traj.actions[move_num]
            if state.active_player() != CHANCE_PLAYER: # ignore chance nodes 
                reward = traj.rewards[state.active_player()]
                if only_winner_moves and reward < 0.5: # select only winner moves
                    continue
                state_vec = state_encoder.state_to_vec(state)
                states.append(np.packbits(state_vec)) # compressed bytes
                # actions.append(ACTION_ID[str(action)]) # ints
                rewards.append(reward) # 0 or 1 
                if traj.freqs:
                    freq_vec = [0.0] * len(PLAYER_ACTIONS_STR)
                    sum_count = sum([x[1] for x in traj.freqs[move_num]])
                    for action_id, count in traj.freqs[move_num]:
                        freq_vec[action_id] = count / sum_count                    
                    actions.append(freq_vec) # probability distribution over all actions
                
                num_states += 1

            state.apply_action(action)

        num_traj += 1

    np.save(data_fname_prefix + "_states.npy", np.array(states))
    np.save(data_fname_prefix + "_actions.npy", np.array(actions, dtype=np.float32))
    np.save(data_fname_prefix + "_rewards.npy", np.array(rewards, dtype=np.float32))

    elapsed = time.perf_counter() - start
    print(f'Processed {num_traj} trajectories, {num_states} states in {elapsed:.2f} sec, {num_traj/elapsed:.2f} traj/sec')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare training data from trajectory files.')
    parser.add_argument('--traj_file', type=str, required=True, help='Path to the trajectory file containing raw data')
    parser.add_argument('--data_prefix', type=str, required=True, help='Prefix to use for generated output files')
    parser.add_argument('--only_winner_moves', action='store_true', help='Use only moves of the winning player (makes rewards files contain only 1s)')
    args = parser.parse_args()

    prepare_data(traj_file=args.traj_file, data_fname_prefix=args.data_prefix, only_winner_moves=args.only_winner_moves)