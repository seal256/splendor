import os, json
from copy import deepcopy
from datetime import datetime
import subprocess
import argparse
from dataclasses import dataclass, asdict

from pysplendor.game import traj_loader
from train import train, create_random_model, TrainConfig
from prepare_features import prepare_features
from scripts.splendor_job import MCTSAgentConfig, PolicyMCTSAgentConfig, NNPolicyConfig, GameConfig

BINARY_PATH = "./splendor"

def agent_score(agent_name, traj_path):
    '''Returns win rate of the agent'''

    tloader = traj_loader(traj_path)
    first_player_score = 0
    total_score = 0
    for traj in tloader:
        agent_idx = traj.agent_names.index(agent_name)
        first_player_score += traj.rewards[agent_idx]
        total_score += sum(traj.rewards)
    return first_player_score / total_score


@dataclass
class SelfPlayConfig:
    """Self-play training configuration"""

    model: str = None                       # Path to initial model (if None a new random model will be used)
    start_step: int = 0                     # Restarts process from the middle if needed
    train_games: int = 5000                 # New train games generated per iteration
    val_games: int = 500                    # New validation games generated per iteration (used for model overfitting control)
    train_buffer_size: int = 10             # Number of past iterations kept in training buffer
    train_epochs: int = 1                   # Model training epochs per iteration
    new_model_eval_games: int = 1000        # Evaluation games against current best model
    min_win_rate: float = 0.54              # Min win rate to adopt a new model (0-1)
    max_iterations: int = 100               # Max total training iterations
    max_iters_without_improvement: int = 12 # Early stopping condition: Stop after this many non-improving iterations
    work_dir: str = "data"                  # Directory for output files
    train: TrainConfig = None               # Model train parameters

class SelfPlayTrainer:
    '''Runs self play iterations and manages all intermediary files in the working directory'''

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.best_model = config.model or f'{config.work_dir}/random.pt'
        self.train_dirs = [f'{config.work_dir}/train_step_{step}' for step in range(config.start_step)]
        self.val_dirs = [f'{config.work_dir}/val_step_{step}' for step in range(config.start_step)]
        self.iters_without_improvement = 0
        
        if not config.model:
            create_random_model(self.best_model)
        
        os.makedirs(config.work_dir, exist_ok=True)

    def train(self):
        """Runs self-play training loop."""

        print('Configuration:')
        print(asdict(self.config))

        for step in range(self.config.start_step, self.config.max_iterations):
            curr_time = datetime.now().strftime("%d.%m %H:%M:%S")
            print(f'\n\n### {curr_time} Iteration {step} ###\n', flush=True)

            train_traj, val_traj = self._collect_self_play_data(step)            
            self._prepare_training_data(step, train_traj, val_traj)        
            if len(self.train_dirs) < self.config.train_buffer_size:
                return False

            new_model = self._train_new_model(step)
            new_model_win_rate = self._evaluate_model(new_model, step)
            
            if new_model_win_rate > self.config.min_win_rate:
                self.best_model = new_model
                print(f'New best model is: {self.best_model}', flush=True)
                self.iters_without_improvement = 0
            else:            
                self.iters_without_improvement += 1
                print(f'Iterations without improvement: {self.iters_without_improvement}', flush=True)

                if self.iters_without_improvement >= self.config.max_iters_without_improvement:
                    print('Stopping', flush=True)
                    break

    def _run_games(self, name_suffix: str, step: int, model_a_path: str, model_b_path: str, num_games: int, train: bool, rotate_agents: bool) -> str:
        """Runs games between two models and returns the trajectory path."""

        print(f'Running {name_suffix} games step {step}', flush=True)
        traj_path = f'{self.config.work_dir}/traj_{name_suffix}_step_{step}.txt'
        
        agent_a = PolicyMCTSAgentConfig("a", train, NNPolicyConfig(model_a_path))
        agent_b = PolicyMCTSAgentConfig("b", train, NNPolicyConfig(model_b_path)) 
        config = GameConfig(agents=[agent_a, agent_b], num_games=num_games, rotate_agents=rotate_agents, dump_trajectories=traj_path)
        
        config_path = f'{self.config.work_dir}/{name_suffix}_step_{step}.json'
        json.dump(asdict(config), open(config_path, 'wt'))
        subprocess.run([BINARY_PATH, config_path], check=True)
        
        return traj_path

    def _collect_self_play_data(self, step: int) -> tuple[str, str]:
        """Runs self-play games and returns (train_traj_path, val_traj_path)."""

        val_traj = self._run_games('val', step, self.best_model, self.best_model, self.config.val_games, train=False, rotate_agents=False)
        train_traj = self._run_games('train', step, self.best_model, self.best_model, self.config.train_games, train=True, rotate_agents=False)
        return train_traj, val_traj

    def _prepare_training_data(self, step: int, train_traj: str, val_traj: str):
        """Processes trajectories into training/validation datasets."""

        train_dir = f'{self.config.work_dir}/train_step_{step}'
        val_dir = f'{self.config.work_dir}/val_step_{step}'
        
        prepare_features(train_traj, train_dir, only_winner_moves=True)
        prepare_features(val_traj, val_dir, only_winner_moves=True)
        
        self.train_dirs.append(train_dir)
        self.val_dirs.append(val_dir)

    def _train_new_model(self, step: int) -> str:
        """Trains a new model and returns its path."""

        new_model = f'{self.config.work_dir}/model_step_{step}.pt'
        print(f'Training new model {new_model}', flush=True)
        
        train_config = deepcopy(self.config.train)
        train_config.train_dir = self.train_dirs[-self.config.train_buffer_size:]
        train_config.val_dir = self.val_dirs[-self.config.train_buffer_size:]
        train_config.start_model_name = self.best_model
        train_config.result_model_name = new_model
        
        train(train_config)
        return new_model

    def _evaluate_model(self, new_model: str, step: int) -> float:
        """Evaluates new model against current best and returns win rate."""

        new_vs_best_traj = self._run_games('new_vs_best', step, new_model, self.best_model, self.config.new_model_eval_games, train=False, rotate_agents=True)
        new_model_win_rate = agent_score("a", new_vs_best_traj)
        print(f'New model win rate vs previous best model: {new_model_win_rate:.3f}', flush=True)
        return new_model_win_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-play configuration")
    parser.add_argument("-c", "--config-file", required=True, help="JSON configuration file")
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        data = json.load(f)
    config = SelfPlayConfig(**data)
    trainer = SelfPlayTrainer(config)

    trainer.train()
    




