#include <vector>
#include <array>
#include <algorithm>

#include "splendor.h"
#include "splendor_agents.h"
#include "nn_policy.h"

using splendor::Action;

// class CheatMCTSAgent : public Agent<Action> {
// private:
//     mcts::MCTSParams mcts_params;

// public:
//     CheatMCTSAgent(const mcts::MCTSParams& params = mcts::MCTSParams()) : mcts_params(params) {}

//     Action get_action(const std::shared_ptr<GameState<Action>>& game_state) const override {
//         std::vector<double> probs(splendor::ACTION_ID.size(), 1);
//         if (game_state->active_player() != CHANCE_PLAYER) {
//             mcts::MCTS<Action> mcts(game_state, mcts_params);
//             mcts.search();

//             double count_sum = 0.0;
//             const auto root_visits = mcts.root_visits();
//             for (const auto& pr : root_visits) {
//                 probs[splendor::ACTION_ID.at(pr.first.to_str())] = pr.second;
//                 count_sum += pr.second;
//             }
//             count_sum += (splendor::ACTION_ID.size() - root_visits.size());

//             for (auto& p: probs) {
//                 p /= count_sum;
//             }            
//         }
//         std::shared_ptr<mcts::Policy<Action>> policy = std::make_shared<ConstantPolicy<Action>>(probs, splendor::ACTION_ID);
        
//         mcts::PolicyMCTS<Action> policy_mcts(game_state, policy, mcts_params);
//         Action action = policy_mcts.search();
        
//         return action;
//     }
// };



namespace splendor {

SplendorGameStateEncoder::SplendorGameStateEncoder(int num_players) 
    : num_players_(num_players), 
        rules_(&DEFAULT_RULES.at(num_players)) {
    calculate_vector_lengths();
}

void SplendorGameStateEncoder::calculate_vector_lengths() {
    // Calculate lengths based on sample cards/nobles
    Card sample_card(0, 0, GemSet());
    Noble sample_noble(0, GemSet());
    card_vec_len_ = card_to_vec(sample_card).size();
    noble_vec_len_ = noble_to_vec(sample_noble).size();
}

std::vector<float> SplendorGameStateEncoder::encode(const std::shared_ptr<GameState<Action>> game_state) const {
    auto splendor_state = std::dynamic_pointer_cast<SplendorGameState>(game_state);
    if (!splendor_state) {
        throw std::runtime_error("SplendorGameStateEncoder expects a SplendorGameState");
    }
    
    std::vector<int> int_vec = state_to_vec(*splendor_state);
    std::vector<float> float_vec(int_vec.begin(), int_vec.end());
    return float_vec;
}

std::vector<int> SplendorGameStateEncoder::gems_to_vec(const GemSet& gems, int max_gems) const {
    if (max_gems == -1) {
        max_gems = rules_->max_gems;
    }
    std::vector<int> vec(NUM_GEMS * (max_gems + 1), 0);
    for (int gem = 0; gem < NUM_GEMS; ++gem) {
        int count = std::min(max_gems, gems.gems[gem]); // this affects cards or gold counts exceeding the limit
        vec[gem * (max_gems + 1) + count] = 1;
    }
    return vec;
}

std::vector<int> SplendorGameStateEncoder::card_to_vec(const Card& card) const {
    std::vector<int> gem(NUM_GEMS, 0);
    gem[card.gem] = 1;
    
    std::vector<int> points(max_card_points_ + 1, 0);
    points[std::min(card.points, max_card_points_)] = 1;
    
    auto price = gems_to_vec(card.price);
    
    std::vector<int> result;
    result.reserve(gem.size() + points.size() + price.size());
    result.insert(result.end(), gem.begin(), gem.end());
    result.insert(result.end(), points.begin(), points.end());
    result.insert(result.end(), price.begin(), price.end());
    
    return result;
}

std::vector<int> SplendorGameStateEncoder::noble_to_vec(const Noble& noble) const {
    // A noble always gives you 3 points, no need to encode them
    return gems_to_vec(noble.price);
}

std::vector<int> SplendorGameStateEncoder::player_to_vec(const SplendorPlayerState& player) const {
    auto card_gems = gems_to_vec(player.card_gems, max_cards_);
    auto table_gems = gems_to_vec(player.gems);
    
    std::vector<int> hand_cards(rules_->max_hand_cards * card_vec_len_, 0);
    for (size_t n = 0; n < player.hand_cards.size(); ++n) {
        auto card = player.hand_cards[n];
        auto card_vec = card_to_vec(*card);
        std::copy(card_vec.begin(), card_vec.end(), hand_cards.begin() + n * card_vec_len_);
    }
    
    std::vector<int> points(rules_->win_points + 1, 0);
    points[std::min(player.points, rules_->win_points)] = 1;
    
    std::vector<int> result;
    result.reserve(card_gems.size() + table_gems.size() + hand_cards.size() + points.size());
    result.insert(result.end(), card_gems.begin(), card_gems.end());
    result.insert(result.end(), table_gems.begin(), table_gems.end());
    result.insert(result.end(), hand_cards.begin(), hand_cards.end());
    result.insert(result.end(), points.begin(), points.end());
    
    return result;
}

std::vector<int> SplendorGameStateEncoder::state_to_vec(const SplendorGameState& state) const {
    // Encodes only open game information (e.g., cards in decks are not encoded)
    std::vector<int> nobles(noble_vec_len_ * rules_->max_nobles, 0);
    for (size_t n = 0; n < state.nobles.size(); ++n) {
        auto noble_vec = noble_to_vec(*state.nobles[n]);
        std::copy(noble_vec.begin(), noble_vec.end(), nobles.begin() + n * noble_vec_len_);
    }

    std::vector<int> table_cards(card_vec_len_ * rules_->max_open_cards * CARD_LEVELS, 0);
    for (int level = 0; level < CARD_LEVELS; ++level) {
        for (size_t ncard = 0; ncard < state.cards[level].size(); ++ncard) {
            size_t pos = (level * rules_->max_open_cards + ncard) * card_vec_len_;
            auto card_vec = card_to_vec(*state.cards[level][ncard]);
            std::copy(card_vec.begin(), card_vec.end(), table_cards.begin() + pos);
        }
    }

    std::vector<int> players;
    int active_player = state.active_player();
    for (int n = 0; n < num_players_; ++n) {
        int player_idx = (n + active_player) % num_players_; // Always start feature vector from the active player
        auto player_vec = player_to_vec(state.players[player_idx]);
        players.insert(players.end(), player_vec.begin(), player_vec.end());
    }

    auto table_gems = gems_to_vec(state.gems);

    std::vector<int> result;
    result.reserve(nobles.size() + table_cards.size() + players.size() + table_gems.size());
    result.insert(result.end(), nobles.begin(), nobles.end());
    result.insert(result.end(), table_cards.begin(), table_cards.end());
    result.insert(result.end(), players.begin(), players.end());
    result.insert(result.end(), table_gems.begin(), table_gems.end());

    return result;
}

AccumValue::AccumValue(double score_norm) 
    : score_norm_(score_norm) {}

std::vector<double> AccumValue::predict(const std::shared_ptr<GameState<splendor::Action>> game_state) const {
    auto splendor_state = std::dynamic_pointer_cast<SplendorGameState>(game_state);
    if (!splendor_state) {
        throw std::runtime_error("AccumValue requires SplendorGameState");
    }

    std::vector<double> values;
    values.reserve(splendor_state->players.size());
    
    for (const auto& player : splendor_state->players) {
        values.push_back(player.points / score_norm_);
    }
    return values;
}

} // namespace splendor



mcts::MCTSParams parse_mcts_params(const json& jsn) {
    mcts::MCTSParams params;
    if (jsn.contains("iterations")) {
        params.iterations = jsn["iterations"];
    }
    if (jsn.contains("exploration")) {
        params.exploration = jsn["exploration"];
    }
    if (jsn.contains("weighted_selection_moves")) {
        params.weighted_selection_moves = jsn["weighted_selection_moves"];
    }
    if (jsn.contains("max_rollout_len")) {
        params.max_rollout_len = jsn["max_rollout_len"];
    }
    if (jsn.contains("max_choice_children")) {
        params.max_choice_children = jsn["max_choice_children"];
    }
    if (jsn.contains("use_rollout_policy")) {
        params.use_rollout_policy = jsn["use_rollout_policy"];
    }
    if (jsn.contains("value_weight")) {
        params.value_weight = jsn["value_weight"];
    }    
    return params;
}

std::shared_ptr<mcts::Policy<Action>> construct_policy(const json& jsn) {
    if (!jsn.contains("type")) {
        throw std::runtime_error("JSON polciy configuration must contain a 'type' field.");
    }
    std::string type = jsn["type"];

    if (type == "ConstantPolicy") {
        std::vector<double> probs = jsn["probs"];
        auto policy = std::make_shared<ConstantPolicy<Action>>(probs, splendor::ACTION_ID);
        return policy;

    } else if (type == "NNPolicy") {
        std::string model_path = jsn["model_path"];
        int num_players = jsn["num_players"];
        auto state_encoder = std::make_shared<splendor::SplendorGameStateEncoder>(num_players);
        auto policy = std::make_shared<mcts::NNPolicy<Action>>(model_path, state_encoder, splendor::ACTION_ID);
        return policy;
    
    } else {
        throw std::runtime_error("Unknown policy type: " + type);
    }
}

std::shared_ptr<Agent<Action>> construct_agent(const json& jsn) {
    if (!jsn.contains("type")) {
        throw std::runtime_error("JSON configuration must contain a 'type' field.");
    }
    std::string agent_type = jsn["type"];

    if (agent_type == "RandomAgent") {
        return std::make_shared<RandomAgent<Action>>();

    } else if (agent_type == "MCTSAgent") {
        mcts::MCTSParams params = parse_mcts_params(jsn);
        return std::make_shared<MCTSAgent<Action>>(params);

    } else if (agent_type == "PolicyMCTSAgent") {
        mcts::MCTSParams params = parse_mcts_params(jsn);
        if (!jsn.contains("policy")) {
            throw std::runtime_error("PolicyMCTSAgent must contain a 'policy' section.");
        }
        auto policy = construct_policy(jsn["policy"]);
        return std::make_shared<PolicyMCTSAgent<Action>>(policy, params);

    } else if (agent_type == "PVMCTSAgent") {
        mcts::MCTSParams params = parse_mcts_params(jsn);
        if (!jsn.contains("policy")) {
            throw std::runtime_error("PolicyMCTSAgent must contain a 'policy' section.");
        }
        auto policy = construct_policy(jsn["policy"]);
        auto value = std::make_shared<splendor::AccumValue>();
        return std::make_shared<PVMCTSAgent<Action>>(policy, value, params);

    // } else if (agent_type == "CheatMCTSAgent") {
    //     mcts::MCTSParams params = parse_mcts_params(jsn);
    //     return std::make_shared<CheatMCTSAgent>(params);

    } else {
        throw std::runtime_error("Unknown agent type: " + agent_type);
    }
}