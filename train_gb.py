import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from prepare_data import PLAYER_ACTIONS
import matplotlib.pyplot as plt

STATE_LEN = 1052
NUM_ACTIONS = 43

def load_train_data(data_fname_prefix, state_len=STATE_LEN):
    states = np.load(data_fname_prefix + "_states.npy", allow_pickle=True)
    actions = np.load(data_fname_prefix + "_actions.npy", allow_pickle=True)
    rewards = np.load(data_fname_prefix + "_rewards.npy", allow_pickle=True)
    states = np.unpackbits(states, axis=1, count=state_len)

    states = states.astype(np.float32)
    actions = actions.astype(np.int32)
    rewards = rewards.astype(np.float32)
    return states, actions, rewards

# https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters

# def train_policy(model_name, train_dir, val_dir):
#     X_train, y_train, _ = load_train_data(train_dir)
#     X_val, y_val, _ = load_train_data(val_dir)

#     train_data = lgb.Dataset(X_train, label=y_train)
#     val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

#     params = {
#         'objective': 'cross_entropy',
#         'boosting': 'gbdt',
#         'learning_rate': 0.05,
#         'verbose': 0
#     }

#     num_round = 100
#     bst = lgb.train(params, train_data, num_round, valid_sets=[train_data, val_data], callbacks=[lgb.log_evaluation(period=5)])

#     y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)

#     accuracy = accuracy_score(y_val, y_pred)
#     print(f"val accuracy: {accuracy:.4f}")
#     print(classification_report(y_val, y_pred, labels=list(range(len(PLAYER_ACTIONS))), target_names = PLAYER_ACTIONS))

#     bst.save_model(model_name)

def train_value(model_name, train_dir, val_dir):
    X_train, _, y_train = load_train_data(train_dir)
    X_val, _, y_val = load_train_data(val_dir)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'l2', 
        'boosting': 'gbdt',
        'learning_rate': 0.05,
        'verbose': 0
    }

    num_round = 100
    bst = lgb.train(params, train_data, num_round, valid_sets=[train_data, val_data], callbacks=[lgb.log_evaluation(period=5)])

    y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print(f'val mse: {mse:.3f} mae: {mae:.3f}')

    bst.save_model(model_name)

def inspect_model():
    # X, y, _ = load_train_data('./data/train/iter0_10k')

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # train_data = lgb.Dataset(X_train, label=y_train)
    # val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model_path = './data/models/lgb_10k.pth'
    bst = lgb.Booster(model_file=model_path)
    # y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
    # y_pred = np.argmax(y_pred, axis=1)     
    
    # print(classification_report(y_val, y_pred, target_names = PLAYER_ACTIONS))
    lgb.plot_importance(bst, max_num_features=20)
    plt.show()

if __name__ == '__main__':
    work_dir = './data_2404'
    name = 'reserve_masked'

    model_name = f'{work_dir}/lgb_model_{name}.txt'
    train_dir = f'{work_dir}/train_{name}'
    val_dir = f'{work_dir}/val_{name}'

    train_value(model_name, train_dir, val_dir)
    # train_policy(model_name, train_dir, val_dir)
    