import subprocess, json
import argparse, os
from dataclasses import asdict

from scripts.splendor_job import RandomAgentConfig, MCTSAgentConfig, PolicyMCTSAgentConfig, NNPolicyConfig, GameConfig, extract_stats
from pysplendor.game import traj_loader

BINARY_PATH = "./splendor"

def run_games(work_dir, name_suffix, agent_pair, num_games=1000, rotate_agents=True, overwrite=False):
    
    print(f'Running {name_suffix} games')

    traj_path = f'{work_dir}/traj_{name_suffix}.txt'
    if os.path.exists(traj_path):
        if overwrite:
            print(f'Overwriting trajectories: {traj_path}')
        else:
            print(f'Trajectories exist: {traj_path}')
            return traj_path
    
    config = GameConfig(agents=agent_pair, num_games=num_games, rotate_agents=rotate_agents, dump_trajectories=traj_path)
    config_path = f'{work_dir}/{name_suffix}.json'
    json.dump(asdict(config), open(config_path, 'wt'), indent=4)

    subprocess.run([BINARY_PATH, config_path], check=True)

    print(f'Created {traj_path}')
    return traj_path

def tournament(work_dir, agents, overwrite=False):
    '''Runs a tournament, all pairs of agents play a series of games'''
        
    for idx1 in range(len(agents)):
        for idx2 in range(idx1 + 1, len(agents)):
            agent_pair = [agents[idx1], agents[idx2]]
            name_suffix = f'{agent_pair[0].name}_vs_{agent_pair[1].name}'

            traj_file = run_games(work_dir, name_suffix, agent_pair, overwrite=overwrite)
            traj = list(traj_loader(traj_file))
            stats = extract_stats(traj)
            stat_file = f'{work_dir}/stats_{name_suffix}.json'
            json.dump(asdict(stats), open(stat_file, 'wt'), indent=4)
            print(f'Created {stat_file}')


if __name__ == '__main__':

    agents = []
    agents.append(RandomAgentConfig(name='random'))
    agents.append(MCTSAgentConfig(iterations=500, max_chance_children=100, name='MCTS_500_full_chance_nodes'))
    agents.append(MCTSAgentConfig(iterations=500, name='MCTS_500'))
    agents.append(MCTSAgentConfig(iterations=2000, name='MCTS_2000'))

    work_dir = './data_baselines'
    tournament(work_dir, agents, overwrite=True)
    


