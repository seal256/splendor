#pragma once

#include <memory>
#include <stdexcept>

#include "game_state.h"
#include "mcts.h"

template<typename ActionT>
struct ActionInfo {
    ActionT action; // selected action
    std::vector<std::pair<ActionT, int>> freqs; // estimates of actions quality
};

template<typename ActionT>
class Agent {
public:
    virtual ~Agent() {};
    virtual ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const = 0;
    virtual ActionInfo<ActionT> get_action_info(const std::shared_ptr<GameState<ActionT>>& game_state) const = 0; // additionally returns frequencies of actions
};

template<typename ActionT>
class RandomAgent : public Agent<ActionT> {
public:
    RandomAgent() {};
    ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        std::vector<ActionT> legal_actions = game_state->get_actions();
        return pick_action(legal_actions);
    }

    ActionInfo<ActionT> get_action_info(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        ActionInfo<ActionT> action_info;
        const auto legal_actions = game_state->get_actions();
        action_info.action = pick_action(legal_actions);
        for (const auto& action: legal_actions) {
            action_info.freqs.emplace_back(action, 1);
        }
        return action_info;
    }

private:
    ActionT pick_action(const std::vector<ActionT>& legal_actions) const {
        if (legal_actions.empty()) {
            throw std::runtime_error("No legal actions available.");
        }
        int random_index = std::rand() % legal_actions.size();
        return legal_actions[random_index];
    }
};

template<typename ActionT>
class MCTSAgent : public Agent<ActionT> {
private:
    const mcts::MCTSParams mcts_params;

public:
    MCTSAgent(const mcts::MCTSParams& params = mcts::MCTSParams()) : mcts_params(params) {}

    ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        mcts::MCTS<ActionT> mcts(game_state, mcts_params);
        ActionT action = mcts.search();    
        return action;
    }

    ActionInfo<ActionT> get_action_info(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        ActionInfo<ActionT> action_info;
        mcts::MCTS<ActionT> mcts(game_state, mcts_params);
        action_info.action = mcts.search();    
        action_info.freqs = mcts.root_visits();
        return action_info;
    }
};

template<typename ActionT>
class ConstantPolicy : public mcts::Policy<ActionT> {
    // Uses constant probabiltiy distribution independent from the state
    const std::vector<double> probs;
    const std::unordered_map<std::string, size_t> action_ids;
public:
    ConstantPolicy(const std::vector<double>& probs, const std::unordered_map<std::string, size_t>& action_ids) : probs(probs), action_ids(action_ids) {}
   
    std::vector<double> predict(const std::shared_ptr<GameState<ActionT>> game_state) const override {
        const std::vector<ActionT> actions = game_state->get_actions();
        std::vector<double> action_probs;
        action_probs.reserve(actions.size());
        double sum_probs = 0.0;
        for (const ActionT& action : actions) {
            const std::string action_str = action.to_str();
            const auto action_id = action_ids.find(action_str);
            if (action_id == action_ids.end()) {
                throw std::invalid_argument("Unknown action string " + action_str);
            }
            double p = probs[action_id->second];
            sum_probs += p;
            action_probs.push_back(p);
        }
        if (sum_probs <= 0.0) {
            sum_probs = 1; 
        }
        for (double& p : action_probs) {
            p /= sum_probs;
        }
        return action_probs;
    }
    ~ConstantPolicy() override {};
};

template<typename ActionT>
class PolicyMCTSAgent : public Agent<ActionT> {
private:
    const std::shared_ptr<mcts::Policy<ActionT>> policy;
    const mcts::MCTSParams mcts_params;

public:
    PolicyMCTSAgent(const std::shared_ptr<mcts::Policy<ActionT>>& policy, const mcts::MCTSParams& params = mcts::MCTSParams()) : policy(policy), mcts_params(params) {}

    ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        mcts::PolicyMCTS<ActionT> mcts(game_state, policy, mcts_params);
        ActionT action = mcts.search();
        
        return action;
    }

    ActionInfo<ActionT> get_action_info(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        ActionInfo<ActionT> action_info;
        mcts::PolicyMCTS<ActionT> mcts(game_state, policy, mcts_params);
        action_info.action = mcts.search();    
        action_info.freqs = mcts.root_visits();
        return action_info;
    }
};


