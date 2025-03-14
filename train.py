import json

from pysplendor.game import Trajectory

def read_trajectories(file_name):
    trajectories = []
    with open(file_name, 'rt') as fin:
        for line in fin:
            data = json.loads(line)
            traj = Trajectory.from_json(data)
            trajectories.append(traj)

    return trajectories

if __name__ == '__main__':
    trajectories = read_trajectories('./data/traj_dump.txt')
    print(trajectories[0].actions[0])
