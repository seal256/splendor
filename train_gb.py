import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from prepare_data import ALL_ACTIONS
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


def train():
    X, y, _ = load_train_data('./data/train/iter0_10k')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'multiclass',  
        'num_class': NUM_ACTIONS,  
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    num_round = 100
    bst = lgb.train(params, train_data, num_round, valid_sets=[val_data], callbacks=[lgb.log_evaluation(period=5)])

    y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
    y_pred = np.argmax(y_pred, axis=1) 

    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names = ALL_ACTIONS))

    model_path = './data/models/lgb_10k.pth'
    bst.save_model(model_path)

def inspect_model():
    # X, y, _ = load_train_data('./data/train/iter0_10k')

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # train_data = lgb.Dataset(X_train, label=y_train)
    # val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model_path = './data/models/lgb_10k.pth'
    bst = lgb.Booster(model_file=model_path)
    # y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
    # y_pred = np.argmax(y_pred, axis=1)     
    
    # print(classification_report(y_val, y_pred, target_names = ALL_ACTIONS))
    lgb.plot_importance(bst, max_num_features=20)
    plt.show()

if __name__ == '__main__':
    # inspect_model()
    train()