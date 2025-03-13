#pragma once

#include<vector>
#include <iostream>
#include <memory>
#include <cmath>
#include <numeric>

#include "agents.h"

template<typename ActionT>
struct Trajectory {
    std::shared_ptr<GameState<ActionT>> initial_state;
    std::vector<ActionT> actions;
    std::vector<double> rewards; // obtained at the end of the game
};

template<typename ActionT>
Trajectory<ActionT> run_one_game(std::shared_ptr<GameState<ActionT>> game_state, const std::vector<std::shared_ptr<Agent<ActionT>>>& agents, bool verbose=false) {
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

template<typename ActionT>
class GameSeriesTask {
public:
    std::vector<std::shared_ptr<Agent<ActionT>>> agents;
    int num_games;
    bool verbose;
    std::string dump_trajectories;

    GameSeriesTask(const json& jsn) {
        if (!jsn.contains("agents") || !jsn.contains("num_games")) {
            throw std::runtime_error("JSON must contain 'agents' and 'num_games' fields.");
        }

        num_games = jsn.at("num_games");
        dump_trajectories = jsn.contains("dump_trajectories") ? jsn.at("dump_trajectories") : "";
        verbose = jsn.value("verbose", false);
        for (const auto& player_config : jsn["agents"]) {
            agents.push_back(construct_agent<ActionT>(player_config));
        }
    }
};

template<typename GameStateT, typename ActionT>
std::vector<Trajectory<ActionT>> run_games(const GameSeriesTask<ActionT>& task) {
    std::vector<Trajectory<ActionT>> trajectories;

    for (int game_num = 0; game_num < task.num_games; ++game_num) {
        auto start = std::chrono::high_resolution_clock::now();

        auto game_state = std::make_shared<GameStateT>(task.agents.size());
        Trajectory<ActionT> trajectory = run_one_game<ActionT>(game_state, task.agents, task.verbose);
        trajectories.push_back(trajectory);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = static_cast<double>((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count()) / 1000.0;
        std::cout << "Game " << game_num << " of " << task.num_games 
            << " took: " << duration << " sec " 
            << " moves: " << trajectory.actions.size()
            << "\n";
    }

    return trajectories;
}
