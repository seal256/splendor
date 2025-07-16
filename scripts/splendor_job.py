from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class RandomAgentConfig:
    type: str = "RandomAgent"
    name: str = ""

@dataclass
class NNPolicyConfig:
    type: str = "NNPolicy"
    model_path: str = ""
    num_players: int = 2

@dataclass
class MCTSAgentConfig:
    type: str = "MCTSAgent"
    iterations: int = 500
    exploration: float = 1.4
    max_chance_children: int = 1
    train: bool = False
    name: str = ""

@dataclass
class PolicyMCTSAgentConfig(MCTSAgentConfig):
    type: str = "PolicyMCTSAgent"
    weighted_selection_moves: int = 20
    p_noise_level: float = 0.0
    alpha: float = 1.0  # creates a uniform distribution
    policy: NNPolicyConfig = field(default_factory=NNPolicyConfig)

@dataclass
class GameConfig:
    agents: List = field(default_factory=list)
    num_games: int = 10000
    num_workers: int = 9
    verbose: bool = False
    save_freqs: bool = True
    win_points: int = 15
    rotate_agents: bool = False
    dump_trajectories: str = ""

@dataclass
class PlayerStats:
    name: str
    total_score: float
    win_rate: float
    confidence_interval: float
    cards_mean: float
    cards_std: float

@dataclass
class GameStats:
    num_games: int
    players: List[PlayerStats]
    game_length_mean: float
    game_length_std: float

    @classmethod
    def from_json(cls, data: dict):
        players = [PlayerStats(**p) for p in data['players']]
        return cls(
            num_games=data['num_games'],
            players=players,
            game_length_mean=data['game_length_mean'],
            game_length_std=data['game_length_std']
        )

def avg_dev(values):
    """Calculate average and standard deviation of a list of values"""
    if not values:
        return 0.0, 0.0
    
    n = len(values)
    mean = sum(values) / n
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    return mean, std_dev

def extract_stats(trajectories) -> GameStats:
    if not trajectories:
        return GameStats(0, [], 0.0, 0.0)

    num_games = len(trajectories)
    num_players = len(trajectories[0].rewards)
    player_names = trajectories[0].agent_names

    total_scores = [0.0] * num_players
    card_counts = [[] for _ in range(num_players)]
    game_lengths = []

    for traj in trajectories:
        for player in range(num_players):
            idx = traj.agent_names.index(player_names[player])
            total_scores[player] += traj.rewards[idx]

        # Replay the game to get the final state
        state = traj.initial_state.copy()
        for action in traj.actions:
            state.apply_action(action)
        
        game_lengths.append(state.round)
        
        for player in range(num_players):
            idx = traj.agent_names.index(player_names[player])
            num_cards = sum(state.players[idx].card_gems._counts)
            card_counts[player].append(num_cards)

    sum_scores = sum(total_scores)
    game_length_mean, game_length_std = avg_dev(game_lengths)

    players_stats = []
    for player in range(num_players):
        cards_mean, cards_std = avg_dev(card_counts[player])
        win_rate = total_scores[player] / sum_scores
        conf_interval = 2.58 * math.sqrt(win_rate * (1.0 - win_rate) / sum_scores) # 99% conf interval
        
        players_stats.append(PlayerStats(player_names[player], total_scores[player], win_rate, conf_interval, cards_mean, cards_std))

    return GameStats(num_games, players_stats, game_length_mean, game_length_std)
