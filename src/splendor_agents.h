#pragma once

#include "agents.h"
#include "splendor.h"
#include "mcts.h"

#include "json.hpp"


std::shared_ptr<Agent> construct_agent(const nlohmann::json& jsn);

namespace splendor {

class SplendorGameStateEncoder : public mcts::GameStateEncoder {
private:
    int num_players_;
    std::shared_ptr<const SplendorGameRules> rules_;
    const int max_card_points_ = 5; // The maximum number of win points on splendor cards
    const int max_cards_ = 6;       // The maximum number of cards of one color that the player can acquire
    
    size_t card_vec_len_;
    size_t noble_vec_len_;

    void calculate_vector_lengths();
public:
    SplendorGameStateEncoder(int num_players);
    std::vector<float> encode(std::shared_ptr<const GameState> game_state) const override;
    std::vector<int> gems_to_vec(const GemSet& gems, int max_gems = -1) const;
    std::vector<int> card_to_vec(const Card& card) const;
    std::vector<int> noble_to_vec(const Noble& noble) const;
    std::vector<int> player_to_vec(const SplendorPlayerState& player) const;
    std::vector<int> state_to_vec(const SplendorGameState& state) const;
    
};

// returns normalised accumulated win points for each player
class ColectedPointsValue : public mcts::Value {
private:
    const double score_norm_;
    
public:
    ColectedPointsValue(double score_norm = 15.0);
    std::vector<double> predict(std::shared_ptr<const GameState> game_state) const override;
};

// returns 1 for the player with most win points
class PointsWinnerValue : public mcts::Value {
public:
    PointsWinnerValue();
    std::vector<double> predict(std::shared_ptr<const GameState> game_state) const override;
};

};

