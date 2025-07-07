#pragma once

#include <memory>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <string>

#include "game_state.h"
#include "mcts.h"

struct ActionInfo {
    int action; // selected action
    std::vector<std::pair<int, int>> freqs; // estimates of actions quality
};

class Agent {
public:
    const std::string name; // agent's name, usually the name of the underlying model
    explicit Agent(const std::string& name) : name(name) {}
    virtual ~Agent() = default;
    virtual int get_action(std::shared_ptr<const GameState> game_state) const = 0;
    virtual ActionInfo get_action_info(std::shared_ptr<const GameState> game_state) const = 0;
};

class RandomAgent : public Agent {
public:
    explicit RandomAgent(const std::string& name) : Agent(name) {}
    int get_action(std::shared_ptr<const GameState> game_state) const override;
    ActionInfo get_action_info(std::shared_ptr<const GameState> game_state) const override;

private:
    int pick_action(const std::vector<int>& legal_actions) const;
};

class MCTSAgent : public Agent {
private:
    const mcts::MCTSParams mcts_params;

public:
    explicit MCTSAgent(const std::string& name, const mcts::MCTSParams& params = mcts::MCTSParams());
    int get_action(std::shared_ptr<const GameState> game_state) const override;
    ActionInfo get_action_info(std::shared_ptr<const GameState> game_state) const override;
};

// Selects actions wiht probabilities proportional to probs vector.
// probs may be of size num_actions * max_round. This is interpreted as stacked move probability vectors wrt game round number.
class ConstantPolicy : public mcts::Policy {
    const std::vector<double> probs;
    const int num_actions;
    const int max_round;
    
public:
    ConstantPolicy(const std::vector<double>& probs, int num_actions);
    std::vector<double> predict(std::shared_ptr<const GameState> game_state, const std::vector<int>& actions) const override;
};

class PolicyMCTSAgent : public Agent {
private:
    std::shared_ptr<const mcts::Policy> policy;
    const mcts::MCTSParams mcts_params;

public:
    PolicyMCTSAgent(const std::string& name, std::shared_ptr<const mcts::Policy> policy, const mcts::MCTSParams& params = mcts::MCTSParams());
    int get_action(std::shared_ptr<const GameState> game_state) const override;
    ActionInfo get_action_info(std::shared_ptr<const GameState> game_state) const override;
};

class ValueMCTSAgent : public Agent {
private:
    std::shared_ptr<const mcts::Value> value;
    const mcts::MCTSParams mcts_params;

public:
    ValueMCTSAgent(const std::string& name, std::shared_ptr<const mcts::Value> value, const mcts::MCTSParams& params = mcts::MCTSParams());
    int get_action(std::shared_ptr<const GameState> game_state) const override;
    ActionInfo get_action_info(std::shared_ptr<const GameState> game_state) const override;
};


class PVMCTSAgent : public Agent {
private:
    std::shared_ptr<const mcts::Policy> policy;
    std::shared_ptr<const mcts::Value> value;
    const mcts::MCTSParams mcts_params;

public:
    PVMCTSAgent(const std::string& name,
                std::shared_ptr<const mcts::Policy> policy,
                std::shared_ptr<const mcts::Value> value,
                const mcts::MCTSParams& params = mcts::MCTSParams());
    int get_action(std::shared_ptr<const GameState> game_state) const override;
    ActionInfo get_action_info(std::shared_ptr<const GameState> game_state) const override;
};