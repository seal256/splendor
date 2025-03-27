#pragma once

#include <vector>
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
    virtual int move_num() const = 0;
};

template<typename ActionT> 
std::ostream& operator<<(std::ostream& os, const GameState<ActionT>& state) {
    state.print(os);
    return os;
}

const int CHANCE_PLAYER = -1; // Should be returned from GameState::active_player() for chance nodes
