#pragma once

#include "game.h"

template<typename ActionT>
class RandomAgent : public Agent<ActionT> {
public:
    RandomAgent() {};
    ~RandomAgent() {};
    ActionT get_action(const std::shared_ptr<GameState<ActionT>>& game_state) const override {
        std::vector<ActionT> legal_actions = game_state->get_actions();
        if (legal_actions.empty()) {
            throw std::runtime_error("No legal actions available.");
        }
        int random_index = std::rand() % legal_actions.size();
        return legal_actions[random_index];
    }
};


