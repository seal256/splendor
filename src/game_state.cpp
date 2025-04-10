#include "game_state.h"

std::ostream& operator<<(std::ostream& os, const GameState& state) {
    state.print(os);
    return os;
}

void to_json(nlohmann::json& j, const GameState& state) {
    state.to_json(j); 
}

void to_json(nlohmann::json& j, const std::shared_ptr<GameState>& state) {
    state->to_json(j);
}
