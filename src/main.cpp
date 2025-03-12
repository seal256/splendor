#include <iostream>
#include <fstream>

#include "json.hpp"
using nlohmann::json;

#include "splendor.h"
#include "agents.h"
#include "game.h"

using splendor::Action;
using splendor::SplendorGameState;

const json DEFAULT_TASK = {
    {"agents", {
        {{"type", "RandomAgent"}},
        {{"type", "MCTSAgent"}, {"iterations", 1000}, {"exploration", 1.4}}
    }},
    {"num_games", 1},
    {"verbose", true}
};

void splendor_stats(const std::vector<Trajectory<Action>>& trajectories);

int main(int argc, char* argv[]) {
    json task_json;

    if (argc < 2) {
        task_json = DEFAULT_TASK;
        std::cout << "Usage: " << argv[0] << " <input_json_file>\n";
        std::cout << "Using default task configuration\n";
        
    } else {
        std::ifstream input_file(argv[1]);
        if (!input_file.is_open()) {
            std::cerr << "Failed to open input JSON file: " << argv[1] << "\n";
            return 1;
        }
        input_file >> task_json;
    }

    GameSeriesTask<Action> task(task_json);
    std::vector<Trajectory<Action>> trajectories = run_games<SplendorGameState, Action>(task);
    
    splendor_stats(trajectories);

    return 0;
}

//// utilities ////

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
        auto final_state = dynamic_cast<SplendorGameState *>(state.get());
        game_lengths.push_back(final_state->round);
        for (int player = 0; player < num_players; ++player) {
            int num_cards = final_state->players[player].card_gems.sum(); 
            card_counts[player].push_back(num_cards);
        }
    }

    for (int player = 0; player < num_players; ++player) {
        auto cards_avg_dev = avg_dev(card_counts[player]);
        std::cout << "player " << player << ":" 
            << " total score: " << total_scores[player]
            << " mean score: " << total_scores[player] / num_games
            << " cards mean: " << cards_avg_dev.first
            << " cards std dev: " << cards_avg_dev.second
            << "\n";
    }

    auto len_mean_dev = avg_dev(game_lengths);
    std::cout << "game length avg: " << len_mean_dev.first << " std dev: " << len_mean_dev.second << "\n";
}