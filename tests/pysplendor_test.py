import subprocess
import os

from pysplendor.game import Trajectory, traj_iter
from pysplendor.splendor import SplendorGameState, CARD_LEVELS, Action

BINARY_PATH = "./splendor"
CONFIG_PATH = "tests/test_config.json"
TRAJ_DUMP_PATH = "data/traj_dump.txt"

def compare_states(state1: SplendorGameState, state2: SplendorGameState) -> bool:
    if state1.round != state2.round:
        raise RuntimeError("Round mismatch")
    if state1.player_to_move != state2.player_to_move:
        raise RuntimeError("Player to move mismatch")
    if state1.skips != state2.skips:
        raise RuntimeError("Skips mismatch")
    if state1.table_card_needed != state2.table_card_needed:
        raise RuntimeError("Table card needed mismatch")
    if state1.deck_level != state2.deck_level:
        raise RuntimeError("Deck level mismatch")

    if len(state1.nobles) != len(state2.nobles):
        raise RuntimeError("Nobles length mismatch")
    for noble1, noble2 in zip(state1.nobles, state2.nobles):
        if noble1.points != noble2.points or noble1.price != noble2.price:
            raise RuntimeError("Noble mismatch")

    for level in range(CARD_LEVELS):
        if len(state1.decks[level]) != len(state2.decks[level]):
            raise RuntimeError(f"Deck level {level} length mismatch")
        for card1, card2 in zip(state1.decks[level], state2.decks[level]):
            if card1.gem != card2.gem or card1.points != card2.points or card1.price != card2.price:
                raise RuntimeError(f"Deck card mismatch at level {level}")

        if len(state1.cards[level]) != len(state2.cards[level]):
            raise RuntimeError(f"Table cards length mismatch at level {level}")
        for card1, card2 in zip(state1.cards[level], state2.cards[level]):
            if card1.gem != card2.gem or card1.points != card2.points or card1.price != card2.price:
                raise RuntimeError(f"Table card mismatch at level {level}")

    if state1.gems != state2.gems:
        raise RuntimeError("Table gems mismatch")

    if len(state1.players) != len(state2.players):
        raise RuntimeError("Players length mismatch")
    for n, (player1, player2) in enumerate(zip(state1.players, state2.players)):
        if player1.id != player2.id:
            raise RuntimeError(f"Player {n} ID mismatch")
        if player1.card_gems != player2.card_gems:
            raise RuntimeError(f"Player {n} card gems mismatch")
        if player1.gems != player2.gems:
            raise RuntimeError(f"Player {n} gems mismatch")
        if player1.points != player2.points:
            raise RuntimeError(f"Player {n} points mismatch")
        if len(player1.hand_cards) != len(player2.hand_cards):
            raise RuntimeError(f"Player {n} hand cards length mismatch")
        for card1, card2 in zip(player1.hand_cards, player2.hand_cards):
            if card1.gem != card2.gem or card1.points != card2.points or card1.price != card2.price:
                raise RuntimeError(f"Player {n} hand card mismatch")

    return True


def test_cpp_vs_python_splendor_implementation_equivalence():
    """Ensures that splendor game mechanics is implemented identically in c++ and python """

    print("Building cpp binary...")
    subprocess.run(["./build.sh"], check=True)

    print("Runs cpp binary to compute trajectories and dumps results...")
    subprocess.run([BINARY_PATH, CONFIG_PATH], check=True)

    assert os.path.exists(TRAJ_DUMP_PATH), f"Trajectories not found in {TRAJ_DUMP_PATH}"

    for n, traj in enumerate(traj_iter(TRAJ_DUMP_PATH)):
        print(f'Trajectory {n}')
        py_state: SplendorGameState = traj.initial_state.copy()
        for n, (action, cpp_state) in enumerate(zip(traj.actions, traj.states)):
            prev_state = py_state.copy()
            py_state.apply_action(Action.from_str(action))
            try:
                compare_states(py_state, cpp_state)
            except Exception as e:
                print('Previous state:')
                print(prev_state)
                print(f'Action {n}: {action}')
                print('Resulting python state:')
                print(py_state)
                print('Resulting c++ state:')
                print(cpp_state)
                raise e
            