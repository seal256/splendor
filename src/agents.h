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
    virtual int get_action(const std::shared_ptr<GameState>& game_state) const = 0;
    virtual ActionInfo get_action_info(const std::shared_ptr<GameState>& game_state) const = 0;
};

class RandomAgent : public Agent {
public:
    explicit RandomAgent(const std::string& name) : Agent(name) {}
    int get_action(const std::shared_ptr<GameState>& game_state) const override;
    ActionInfo get_action_info(const std::shared_ptr<GameState>& game_state) const override;

private:
    int pick_action(const std::vector<int>& legal_actions) const;
};

class MCTSAgent : public Agent {
private:
    const mcts::MCTSParams mcts_params;

public:
    explicit MCTSAgent(const std::string& name, const mcts::MCTSParams& params = mcts::MCTSParams());
    int get_action(const std::shared_ptr<GameState>& game_state) const override;
    ActionInfo get_action_info(const std::shared_ptr<GameState>& game_state) const override;
};

class ConstantPolicy : public mcts::Policy {
    const std::vector<double> probs;
    const int num_actions;
    const int max_round;
    
public:
    ConstantPolicy(const std::vector<double>& probs, int num_actions);
    std::vector<double> predict(const std::shared_ptr<GameState> game_state, const std::vector<int>& actions) const override;
};

class PolicyMCTSAgent : public Agent {
private:
    const std::shared_ptr<mcts::Policy> policy;
    const mcts::MCTSParams mcts_params;

public:
    PolicyMCTSAgent(const std::string& name, const std::shared_ptr<mcts::Policy>& policy, const mcts::MCTSParams& params = mcts::MCTSParams());
    int get_action(const std::shared_ptr<GameState>& game_state) const override;
    ActionInfo get_action_info(const std::shared_ptr<GameState>& game_state) const override;
};

class ValueMCTSAgent : public Agent {
private:
    const std::shared_ptr<mcts::Value> value;
    const mcts::MCTSParams mcts_params;

public:
    ValueMCTSAgent(const std::string& name, const std::shared_ptr<mcts::Value>& value, const mcts::MCTSParams& params = mcts::MCTSParams());
    int get_action(const std::shared_ptr<GameState>& game_state) const override;
    ActionInfo get_action_info(const std::shared_ptr<GameState>& game_state) const override;
};


class PVMCTSAgent : public Agent {
private:
    const std::shared_ptr<mcts::Policy> policy;
    const std::shared_ptr<mcts::Value> value;
    const mcts::MCTSParams mcts_params;

public:
    PVMCTSAgent(const std::string& name,
                const std::shared_ptr<mcts::Policy>& policy,
                const std::shared_ptr<mcts::Value>& value,
                const mcts::MCTSParams& params = mcts::MCTSParams());
    int get_action(const std::shared_ptr<GameState>& game_state) const override;
    ActionInfo get_action_info(const std::shared_ptr<GameState>& game_state) const override;
};