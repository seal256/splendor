from dataclasses import dataclass
import re
from typing import List

from dataclasses import dataclass, field
from typing import List

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
    win_points: int = 5
    rotate_agents: bool = False
    dump_trajectories: str = ""




@dataclass
class JobReport:
    step: int
    player_a_score: float
    player_a_conf: float
    player_a_cards_mean: float
    player_a_cards_std: float
    game_length_avg: float
    game_length_std: float

def parse_job_report(file_path: str) -> List[JobReport]:
    with open(file_path, 'r') as file:
        content = file.read()
    
    reports = re.split(r'(?=\nseed:)', content.strip())
    results = []
    
    for report in reports:
        if not report.strip():
            continue
            
        # Extract player a stats
        player_a_match = re.search(
            r'player a: total score: [\d.]+,\s*mean score: ([\d.]+),\s*score conf interval: ([\d.]+),\s*cards mean: ([\d.]+),\s*cards std dev: ([\d.]+)',
            report
        )
        # Extract game length stats
        game_length_match = re.search(
            r'game length avg: ([\d.]+),\s*std dev: ([\d.]+)',
            report
        )
        # Extract step number
        step_match = re.search(
            r'traj_trained_vs_mcts_step_(\d+)\.txt',
            report
        )
        
        if all([player_a_match, game_length_match, step_match]):
            results.append(JobReport(
                step=int(step_match.group(1)),
                player_a_score=float(player_a_match.group(1)),
                player_a_conf=float(player_a_match.group(2)),
                player_a_cards_mean=float(player_a_match.group(3)),
                player_a_cards_std=float(player_a_match.group(4)),
                game_length_avg=float(game_length_match.group(1)),
                game_length_std=float(game_length_match.group(2))
            ))
    
    return results
