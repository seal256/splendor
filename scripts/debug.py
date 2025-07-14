from pysplendor.game_state import GameState, CHANCE_PLAYER
from pysplendor.agents import RandomAgent, MCTSAgent, Agent, HumanPlayer
from pysplendor.splendor import SplendorGameState, ACTIONS_STR, SplendorGameRules, DEFAULT_RULES
from pysplendor.game import run_one_game, Trajectory, traj_loader
from pysplendor.mcts import MCTS, PVMCTS, Value, PolicyMCTS, Policy, MCTSParams
from prepare_data import SplendorGameStateEncoder
from train import STATE_LEN, NUM_ACTIONS, MLP

def print_trajectory_from_file(file_name, line_num):
    tloader = traj_loader(file_name)
    for _ in range(line_num):
        traj = next(tloader)
    print_game_record(traj)


def print_game_record(traj: Trajectory):
    '''Displays a recorded trajectory to console with some extra info. Debugging tool.'''

    # probs = [1/len(ACTION_ID)] * len(ACTION_ID)
    # mcts_params = MCTSParams()
    # const_policy = ConstantPolicy(probs)
    model = load_mlp_model('./data/models/mlp_iter1_best.pt')
    state_encoder = SplendorGameStateEncoder(2)
    nn_policy = NNPolicy(model, state_encoder)
    # value_fun = AccumValue()

    game_state = traj.initial_state.copy()
    for n, action in enumerate(traj.actions):
        print(game_state)
        print(f'active_player: {game_state.active_player()} action: {ACTIONS_STR[action]}')
        if traj.freqs:
            if game_state.active_player() != CHANCE_PLAYER:
                recorded_visits = {a: c for a, c in traj.freqs[n]}
                # print(f'recorded: {recorded_visits}')

                # mcts = PolicyMCTS(game_state, const_policy, mcts_params)
                # mcts.search()
                # mcts_vistis = {str(action): count for action, count in  mcts.root_visits()}

                # pv_mcts = PVMCTS(game_state, const_policy, value_fun, mcts_params)
                # pv_mcts.search()
                # pv_mcts_vistis = {str(action): count for action, count in  pv_mcts.root_visits()}

                probs = nn_policy.predict(game_state)
                visits_str = ''.join([f'{ACTIONS_STR[a]}: {c}, {probs[n]:.4f}\t' for n, (a, c) in enumerate(traj.freqs[n])])
                # visits_str = ''.join(sorted([f'{a}: {recorded_visits[a]}, {mcts_vistis[a]}, {pv_mcts_vistis[a]}, {probs[n]:.4f}\t' for n, a in enumerate(mcts_vistis.keys())]))
                print(f'recorded and probs: {visits_str}')
        print()
        game_state.apply_action(action)
    
    print(game_state)
    for id, r in enumerate(game_state.rewards()):
        print(f'player{id}: {r}')
 
def custom_model_evaluation():
    work_dir = './data_2404'

    device = 'mps'
    model_name = f'{work_dir}/model_reserve_masked_50k_best.pt'
    model = torch.jit.load(model_name, map_location=torch.device(device))
    model.eval()
    print_weigths(model)
        
    criterion = loss
    batch_size = 1024

    for move in [5, 10, 15, 20, 25, 30]:
        val_dir = f'{work_dir}/val_rm10k_move{move}'
        val_dataset = SplendorDataset(data_fname_prefix=val_dir)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_data_entropy = data_loss(val_loader, criterion)

        val_loss, val_classif_pred, val_classif_correct = validate(model, val_loader, criterion, device)
        val_accuracy = accuracy_score(val_classif_correct, val_classif_pred)

        print(f"move {move}: val loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
        print(f'data entropy: {val_data_entropy:.4f}')
        # print(classification_report(val_classif_correct, val_classif_pred, labels = list(range(len(PLAYER_ACTIONS))), target_names = PLAYER_ACTIONS, zero_division=0))
