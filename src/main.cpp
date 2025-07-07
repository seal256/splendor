#include <iostream>
#include <fstream>
#include <format>

#include "json.hpp"
using nlohmann::json;

#include "splendor.h"
#include "agents.h"
#include "game.h"
#include "nn_policy.h"
using splendor::SplendorGameState;

const json DEFAULT_TASK = {
    {"agents", {
        {{"type", "MCTSAgent"}, {"iterations", 1000}, {"exploration", 1.4}},
        {{"type", "MCTSAgent"}, {"iterations", 1000}, {"exploration", 1.4}}
    }},
    {"num_games", 5},
    {"verbose", true}
};

void splendor_stats(const std::vector<Trajectory>& trajectories);
void run_model(const nlohmann::json& task_json);

int main(int argc, char* argv[]) {
    json task_json;

    if (argc < 2) {
        task_json = DEFAULT_TASK;
        std::cout << "Usage: " << argv[0] << " <input_json_file>\n";
        std::cout << "Using default configuration\n";
        
    } else {
        std::ifstream input_file(argv[1]);
        if (!input_file.is_open()) {
            std::cerr << "Failed to open input JSON file: " << argv[1] << "\n";
            return 1;
        }
        input_file >> task_json;
    }
    unsigned int seed = task_json.value("random_seed", std::time(0));
    std::cout << "seed: " << seed << std::endl;
    std::srand(seed);

    const std::string task_name = task_json.value("task", "run_games");
    if (task_name == "run_games") {
        GameSeriesTask task(task_json);
        std::vector<Trajectory> trajectories = task.num_workers > 1 ? 
            run_games_parallel(task) : run_games(task);
        
        splendor_stats(trajectories);
        if (!task.dump_trajectories.empty()) {
            dump_trajectories(task.dump_trajectories, trajectories);
        }

    } else if (task_name == "run_model") {
        // for debug: runs the action model once
        run_model(task_json);
    }

    return 0;
}

// Utilities

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

size_t player_idx(const Trajectory& traj, const std::string& name) {
    // Finds index of a player with a given name
    return std::find(traj.agent_names.begin(), traj.agent_names.end(), name) - traj.agent_names.begin();
}

void splendor_stats(const std::vector<Trajectory>& trajectories) {
    if (trajectories.empty()) {
        return;
    }

    int num_games = trajectories.size();
    std::cout << "Number of games: " << num_games << "\n";

    int num_players = trajectories[0].rewards.size();
    const auto player_names = trajectories[0].agent_names;
    std::vector<double> total_scores(num_players, 0.0);
    std::vector<std::vector<double>> card_counts(num_players); // number of purchased cards for each player
    card_counts.reserve(2);
    std::vector<double> game_lengths;

    for (const auto& traj : trajectories) {
        for (int player = 0; player < num_players; ++player) {
            size_t idx = player_idx(traj, player_names[player]);
            total_scores[player] += traj.rewards[idx];
        }

        // replay the game to correctly count moves and inspect the final state
        auto state = traj.initial_state->clone();
        for (auto action : traj.actions) {
            state->apply_action(action);
        }
        auto final_state = std::dynamic_pointer_cast<const SplendorGameState>(state);
        game_lengths.push_back(final_state->round);
        for (int player = 0; player < num_players; ++player) {
            size_t idx = player_idx(traj, player_names[player]);
            int num_cards = final_state->players[idx].card_gems.sum(); 
            card_counts[player].push_back(num_cards);
        }
    }

    // short report
    double sum_scores = 0;
    for (int player = 0; player < num_players; ++player) {
        sum_scores += total_scores[player];
    }
    for (int player = 0; player < num_players; ++player) {
        double win_rate = total_scores[player] / sum_scores;
        double conf_interval = 2.58 * std::sqrt(win_rate * (1.0 - win_rate) / sum_scores); // 99% confidence
        std::cout << std::format("{:.3f} ({:.2f}), ",  win_rate, conf_interval);
    }
    std::cout << std::endl;

    for (int player = 0; player < num_players; ++player) {
        auto cards_avg_dev = avg_dev(card_counts[player]);
        double win_rate = total_scores[player] / sum_scores;
        double conf_interval = 2.58 * std::sqrt(win_rate * (1.0 - win_rate) / sum_scores); // 99% confidence
        std::cout << std::format(
            "player {}: total score: {:.1f}, mean score: {:.3f}, "
            "score conf interval: {:.3f}, cards mean: {:.1f}, "
            "cards std dev: {:.1f}\n",
            player_names[player],
            total_scores[player],
            win_rate,
            conf_interval,
            cards_avg_dev.first,
            cards_avg_dev.second
        );
    }

    auto len_mean_dev = avg_dev(game_lengths);
    std::cout << std::format("game length avg: {:.1f}, std dev: {:.1f}\n", len_mean_dev.first, len_mean_dev.second);
}

void run_model(const json& task_json) {
    const std::string model_path = task_json["model_path"];
    int num_players = task_json["num_players"];
    const std::string result_file = task_json["result_file"];

    auto game_state = std::make_shared<SplendorGameState>(num_players);
    int num_moves = std::rand() % 20;
    auto random_agent = RandomAgent("");
    for (int n = 0; n < num_moves; n++) { // makes a few moves to disturb the state
        auto action = random_agent.get_action(game_state);
        game_state->apply_action(action);
        if (n == num_moves - 1 && game_state->active_player() == CHANCE_PLAYER) {
            num_moves += 1;
        }
    }

    auto state_encoder = std::make_shared<splendor::SplendorGameStateEncoder>(num_players);
    std::vector<int> state_vec = state_encoder->state_to_vec(*game_state);
    
    auto policy = std::make_shared<mcts::NNPolicy>(model_path, state_encoder);
    auto actions = game_state->get_actions();
    auto prediction = policy->predict(game_state, actions);

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
