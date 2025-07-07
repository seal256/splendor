#include <cstdlib>
#include <stdexcept>
#include <numeric>

#include "splendor_agents.h"

int RandomAgent::pick_action(const std::vector<int>& legal_actions) const {
    if (legal_actions.empty()) {
        throw std::runtime_error("No legal actions available.");
    }
    return legal_actions[std::rand() % legal_actions.size()];
}

int RandomAgent::get_action(std::shared_ptr<const GameState> game_state) const {
    return pick_action(game_state->get_actions());
}

ActionInfo RandomAgent::get_action_info(std::shared_ptr<const GameState> game_state) const {
    ActionInfo action_info;
    const auto legal_actions = game_state->get_actions();
    action_info.action = pick_action(legal_actions);
    for (const auto& action : legal_actions) {
        action_info.freqs.emplace_back(action, 1);
    }
    return action_info;
}

MCTSAgent::MCTSAgent(const std::string& name, const mcts::MCTSParams& params) : Agent(name), mcts_params(params) {}

int MCTSAgent::get_action(std::shared_ptr<const GameState> game_state) const {
    mcts::MCTS mcts(game_state, mcts_params);
    return mcts.search();
}

ActionInfo MCTSAgent::get_action_info(std::shared_ptr<const GameState> game_state) const {
    ActionInfo action_info;
    mcts::MCTS mcts(game_state, mcts_params);
    action_info.action = mcts.search();
    action_info.freqs = mcts.root_visits();
    return action_info;
}

ConstantPolicy::ConstantPolicy(const std::vector<double>& probs, int num_actions)
    : probs(probs), num_actions(num_actions), max_round(probs.size() / num_actions - 1) {}

std::vector<double> ConstantPolicy::predict(std::shared_ptr<const GameState> game_state, const std::vector<int>& actions) const {
    std::vector<double> action_probs;
    action_probs.reserve(actions.size());
    double sum_probs = 0.0;
    int offset = num_actions * std::min(game_state->move_num(), max_round);
    for (const int action : actions) {
        double p = probs[offset + action];
        sum_probs += p;
        action_probs.push_back(p);
    }
    
    if (sum_probs <= 0.0) {
        sum_probs = 1.0;
    }
    for (double& p : action_probs) {
        p /= sum_probs;
    }
    return action_probs;
}

PolicyMCTSAgent::PolicyMCTSAgent(const std::string& name, 
                                std::shared_ptr<const mcts::Policy> policy, 
                                const mcts::MCTSParams& params)
    : Agent(name), policy(policy), mcts_params(params) {}

int PolicyMCTSAgent::get_action(std::shared_ptr<const GameState> game_state) const {
    mcts::PolicyMCTS mcts(game_state, policy, mcts_params);
    return mcts.search();
}

ActionInfo PolicyMCTSAgent::get_action_info(std::shared_ptr<const GameState> game_state) const {
    ActionInfo action_info;
    mcts::PolicyMCTS mcts(game_state, policy, mcts_params);
    action_info.action = mcts.search();
    action_info.freqs = mcts.root_visits();
    return action_info;
}

ValueMCTSAgent::ValueMCTSAgent(const std::string& name, std::shared_ptr<const mcts::Value> value, const mcts::MCTSParams& params)
    : Agent(name), value(value), mcts_params(params) {}

int ValueMCTSAgent::get_action(std::shared_ptr<const GameState> game_state) const {
    mcts::ValueMCTS mcts(game_state, value, mcts_params);
    return mcts.search();
}

ActionInfo ValueMCTSAgent::get_action_info(std::shared_ptr<const GameState> game_state) const {
    ActionInfo action_info;
    mcts::ValueMCTS mcts(game_state, value, mcts_params);
    action_info.action = mcts.search();
    action_info.freqs = mcts.root_visits();
    return action_info;
}

PVMCTSAgent::PVMCTSAgent(const std::string& name,
                        std::shared_ptr<const mcts::Policy> policy,
                        std::shared_ptr<const mcts::Value> value,
                        const mcts::MCTSParams& params)
    : Agent(name), policy(policy), value(value), mcts_params(params) {}

int PVMCTSAgent::get_action(std::shared_ptr<const GameState> game_state) const {
    mcts::PVMCTS mcts(game_state, policy, value, mcts_params);
    return mcts.search();
}

ActionInfo PVMCTSAgent::get_action_info(std::shared_ptr<const GameState> game_state) const {
    ActionInfo action_info;
    mcts::PVMCTS mcts(game_state, policy, value, mcts_params);
    action_info.action = mcts.search();
    action_info.freqs = mcts.root_visits();
    return action_info;
}