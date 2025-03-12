#pragma once

#include<vector>
#include <iostream>
#include <memory>

template<typename ActionT> 
class GameState {
public:
    virtual ~GameState() {};

    virtual std::vector<ActionT> get_actions() const = 0;
    virtual int active_player() const = 0;
    virtual void apply_action(const ActionT& action) = 0;
    virtual bool is_terminal() const = 0;
    virtual std::vector<double> rewards() const = 0;
    virtual std::shared_ptr<GameState<ActionT>> clone() const = 0;
    virtual void print(std::ostream& os) const = 0;
};

template<typename ActionT> 
std::ostream& operator<<(std::ostream& os, const GameState<ActionT>& state) {
    state.print(os);
    return os;
}

const int CHANCE_PLAYER = -1; // Should be returned from GameState::active_player() for chance nodes

template<typename ActionT>
class Agent {
public:
    virtual ~Agent() {};
    virtual ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const = 0;
};

template<typename ActionT>
struct Trajectory {
    std::shared_ptr<GameState<ActionT>> initial_state;
    std::vector<ActionT> actions;
    std::vector<double> rewards; // obtained at the end of the game
};

template<typename ActionT>
Trajectory<ActionT> game_round(std::shared_ptr<GameState<ActionT>> game_state, const std::vector<std::shared_ptr<Agent<ActionT>>>& agents, bool verbose=false) {
    Trajectory<ActionT> trajectory;
    trajectory.initial_state = game_state->clone();

    int active_player = game_state->active_player();
    while (!game_state->is_terminal()) {
        ActionT action;
        if (active_player == CHANCE_PLAYER) { // chance game state
            auto legal_actions = game_state->get_actions();
            int idx = std::rand() % legal_actions.size();
            action = legal_actions[idx];

        } else {
            action = agents[active_player]->get_action(game_state);
        }

        if (verbose) {
            std::cout << "\n" << *game_state << "\n";
            std::cout << "selected action: " << action << "\n";
        }

        trajectory.actions.push_back(action);
        game_state->apply_action(action);
        active_player = game_state->active_player();
    }

    auto rewards = game_state->rewards();
    trajectory.rewards = rewards;

    if (verbose) {
        std::cout << *game_state << "\n";
        std::cout << "Final scores:" << "\n";
        for (size_t n = 0; n < rewards.size(); ++n) {
            std::cout << "player " << n << " score: " << rewards[n] << "\n";
        }
    }

    return trajectory;
}