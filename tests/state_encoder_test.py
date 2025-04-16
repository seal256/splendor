import subprocess
import os, sys, json
import torch

# sys.path.append('/Users/seal/projects/splendor/')

from pysplendor.game import Trajectory, traj_loader
from pysplendor.splendor import SplendorGameState, CARD_LEVELS, Action
from play import load_mlp_model, SplendorGameStateEncoder, NNPolicy

BINARY_PATH = "./splendor"
CONFIG_PATH = "tests/state_encoder_test_config.json"

def test_cpp_vs_python_state_encoder_implementation_equivalence():
    """Ensures that splendor state encoder and model work identically in c++ and python """

    assert os.path.exists(CONFIG_PATH), f"Test config file not found in {CONFIG_PATH}"

    test_config = json.load(open(CONFIG_PATH, 'rt'))
    result_file = test_config['result_file']
    num_players = test_config['num_players']
    model_path = test_config['model_path']

    # 1. Run c++ model version

    print("Building cpp binary...")
    subprocess.run(["./build.sh"], check=True)

    print("Runs cpp binary to run model...")
    subprocess.run([BINARY_PATH, CONFIG_PATH], check=True)

    assert os.path.exists(result_file), f"Cpp run result not found in {result_file}"
    result = json.load(open(result_file))

    game_state = SplendorGameState.from_json(result['game_state'])
    state_vec_cpp = result['state_vec']
    prediction_cpp = result['prediction']

    # 2. Run python version

    # model = load_mlp_model(model_path)
    model = torch.jit.load(model_path)
    model.eval()
    state_encoder = SplendorGameStateEncoder(num_players)
    policy = NNPolicy(model, state_encoder)

    state_vec_python = state_encoder.state_to_vec(game_state)
    prediction_python = policy.predict(game_state)

    # 3. Compare results
      
    assert state_vec_cpp == state_vec_python, "Encoded state vectors are not equal" 
    assert prediction_cpp == prediction_python, "Model predictions are not equal" 

# if __name__ == '__main__':
#     test_cpp_vs_python_state_encoder_implementation_equivalence()
