#include <iostream>
#include <fstream>
#include <format>

#include "util.h"
#include "nn_policy.h"

using nlohmann::json;
using splendor::Action;
using splendor::SplendorGameState;

std::pair<double, double> avg_dev(const std::vector<double>& values) {
    if (values.empty()) {
        return {0.0, 0.0};
    }

    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();

    double variance = 0.0;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size();
    return {mean, std::sqrt(variance)};
}

void splendor_stats(const std::vector<Trajectory<Action>>& trajectories) {
    if (trajectories.empty()) {
        return;
    }

    int num_games = trajectories.size();
    std::cout << "Number of games: " << num_games << "\n";

    int num_players = trajectories[0].rewards.size();
    std::vector<double> total_scores(num_players, 0.0);
    std::vector<std::vector<double>> card_counts(num_players); // number of purchased cards for each player
    card_counts.reserve(2);
    std::vector<double> game_lengths;

    for (const auto& traj : trajectories) {
        for (int player = 0; player < num_players; ++player) {
            total_scores[player] += traj.rewards[player];
        }

        // replay the game to correctly count moves and inspect the final state
        auto state = traj.initial_state->clone();
        for (auto action : traj.actions) {
            state->apply_action(action);
        }
        auto final_state = std::dynamic_pointer_cast<SplendorGameState>(state);
        game_lengths.push_back(final_state->round);
        for (int player = 0; player < num_players; ++player) {
            int num_cards = final_state->players[player].card_gems.sum(); 
            card_counts[player].push_back(num_cards);
        }
    }

    // short report
    for (int player = 0; player < num_players; ++player) {
        double mean_score = total_scores[player] / num_games;
        double conf_interval = 2.58 * std::sqrt(mean_score * (1.0 - mean_score) / num_games); // 99% confidence
        std::cout << std::format("{:.3f} ({:.2f}), ",  mean_score, conf_interval);
    }
    std::cout << std::endl;

    for (int player = 0; player < num_players; ++player) {
        auto cards_avg_dev = avg_dev(card_counts[player]);
        double mean_score = total_scores[player] / num_games;
        double conf_interval = 2.58 * std::sqrt(mean_score * (1.0 - mean_score) / num_games); // 99% confidence
        std::cout << std::format(
            "player {}: total score: {:.1f}, mean score: {:.3f}, "
            "score conf interval: {:.3f}, cards mean: {:.1f}, "
            "cards std dev: {:.1f}\n",
            player,
            total_scores[player],
            mean_score,
            conf_interval,
            cards_avg_dev.first,
            cards_avg_dev.second
        );
    }

    auto len_mean_dev = avg_dev(game_lengths);
    std::cout << "game length avg: " << len_mean_dev.first << " std dev: " << len_mean_dev.second << "\n";
}

void to_json(json& j, const Trajectory<Action>& traj) {
    std::vector<std::string> actions;
    for (const auto action : traj.actions)
        actions.push_back(action.to_str());
    
    auto initial_state = std::dynamic_pointer_cast<SplendorGameState>(traj.initial_state);
    j = {
        {"initial_state", *initial_state},
        {"rewards", traj.rewards},
        {"actions", actions}
    };

    if (!traj.states.empty()) {
        std::vector<json> states;
        for (auto state : traj.states) {
            auto splendor_state = std::dynamic_pointer_cast<SplendorGameState>(state);
            states.push_back(*splendor_state);
        }
        j["states"] = states;
    }

    if (!traj.freqs.empty()) {
        std::vector<std::map<std::string, int>> freqs;
        freqs.resize(traj.freqs.size());
        for (size_t action_num = 0; action_num < freqs.size(); action_num++) {
            for (const auto& pr : traj.freqs[action_num]) {
                freqs[action_num][pr.first.to_str()] = pr.second;
            }
        }
        j["freqs"] = freqs;
    }
}

void dump_trajectories(const std::string& file_name, const std::vector<Trajectory<Action>>& trajectories) {
    std::ofstream out_file(file_name);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    for (const auto& traj : trajectories) {
        json jsn = traj;
        out_file << jsn.dump() << "\n";
    }

    out_file.close();    
    std::cout << trajectories.size() << " trajectories are dumped to " << file_name << "\n";
}

void run_model(const json& task_json) {
    const std::string model_path = task_json["model_path"];
    int num_players = task_json["num_players"];
    const std::string result_file = task_json["result_file"];

    auto game_state = std::make_shared<SplendorGameState>(num_players);
    int num_moves = std::rand() % 20;
    auto random_agent = RandomAgent<Action>();
    for (int n = 0; n < num_moves; n++) { // makes a few moves to disturb the state
        auto action = random_agent.get_action(game_state);
        game_state->apply_action(action);
        if (n == num_moves - 1 && game_state->active_player() == CHANCE_PLAYER) {
            num_moves += 1;
        }
    }

    auto state_encoder = std::make_shared<splendor::SplendorGameStateEncoder>(num_players);
    std::vector<int> state_vec = state_encoder->state_to_vec(*game_state);
    
    auto policy = std::make_shared<mcts::NNPolicy<Action>>(model_path, state_encoder, splendor::ACTION_ID);
    auto prediction = policy->predict(game_state);

    json result = {
        {"game_state", *game_state},
        {"state_vec", state_vec},
        {"prediction", prediction}
    };

    std::ofstream out_file(result_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open file: " + result_file);
    }

    out_file << result.dump() << "\n";
}

