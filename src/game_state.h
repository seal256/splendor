#pragma once

#include <vector>
#include <iostream>
#include <memory>

#include "json.hpp"

class GameState {
public:
    virtual ~GameState() {};

    virtual std::vector<int> get_actions() const = 0;
    virtual int active_player() const = 0;
    virtual void apply_action(const int action) = 0;
    virtual bool is_terminal() const = 0;
    virtual std::vector<double> rewards() const = 0;
    virtual std::shared_ptr<GameState> clone() const = 0;
    virtual int move_num() const = 0;

    virtual void print(std::ostream& os) const = 0;
    virtual void to_json(nlohmann::json& j) const = 0;
};

std::ostream& operator<<(std::ostream& os, const GameState& state);
void to_json(nlohmann::json& j, const GameState& state);
void to_json(nlohmann::json& j, const std::shared_ptr<GameState>& state);

constexpr int CHANCE_PLAYER = -1; // Should be returned from GameState::active_player() for chance nodes
