#pragma once

#include <memory>
#include <stdexcept>

#include "json.hpp"
using json = nlohmann::json;

#include "game_state.h"
#include "mcts.h"

template<typename ActionT>
class Agent {
public:
    virtual ~Agent() {};
    virtual ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const = 0;
};

template<typename ActionT>
class RandomAgent : public Agent<ActionT> {
public:
    RandomAgent() {};
    ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        std::vector<ActionT> legal_actions = game_state->get_actions();
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
    mcts::MCTSParams mcts_params;

public:
    MCTSAgent(const mcts::MCTSParams& params = mcts::MCTSParams()) : mcts_params(params) {}

    ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        mcts::MCTS<ActionT> mcts(game_state, mcts_params);
        // auto start = std::chrono::high_resolution_clock::now();
        
        ActionT action = mcts.search();
        
        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = static_cast<double>((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count()) / 1000.0;
        // std::cout << "Elapsed: " << duration << " sec, Iterations per second: " << mcts_params.iterations / duration << "\n";
        
        return action;
    }
};

template<typename ActionT>
class PolicyMCTSAgent : public Agent<ActionT> {
private:
    mcts::MCTSParams mcts_params;

public:
    PolicyMCTSAgent(const mcts::MCTSParams& params = mcts::MCTSParams()) : mcts_params(params) {}

    ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        mcts::MCTS<ActionT> mcts(game_state, mcts_params);
        ActionT action = mcts.search();
        
        return action;
    }
};


template<typename ActionT>
std::shared_ptr<Agent<ActionT>> construct_agent(const json& jsn) {
    if (!jsn.contains("type")) {
        throw std::runtime_error("JSON configuration must contain a 'type' field.");
    }
    std::string agent_type = jsn["type"];

    if (agent_type == "RandomAgent") {
        return std::make_shared<RandomAgent<ActionT>>();

    } else if (agent_type == "MCTSAgent") {
        mcts::MCTSParams params;
        if (jsn.contains("iterations")) {
            params.iterations = jsn["iterations"];
        }
        if (jsn.contains("exploration")) {
            params.exploration = jsn["exploration"];
        }
        return std::make_shared<MCTSAgent<ActionT>>(params);

    } else if (agent_type == "PolicyMCTSAgent") {
        mcts::MCTSParams params;
        if (jsn.contains("iterations")) {
            params.iterations = jsn["iterations"];
        }
        if (jsn.contains("exploration")) {
            params.exploration = jsn["exploration"];
        }
        return std::make_shared<PolicyMCTSAgent<ActionT>>(params);

    } else {
        throw std::runtime_error("Unknown agent type: " + agent_type);
    }
}

