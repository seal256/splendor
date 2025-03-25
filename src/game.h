#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cmath>
#include <numeric>
#include <future>
#include <chrono>

#include "splendor_agents.h"

template<typename ActionT>
struct Trajectory {
    std::shared_ptr<GameState<ActionT>> initial_state;
    std::vector<ActionT> actions;
    std::vector<double> rewards; // obtained at the end of the game
    std::vector<std::shared_ptr<GameState<ActionT>>> states; // optional
    std::vector<std::vector<std::pair<ActionT, int>>> freqs; // Optional, root node action counts for each state
};

template<typename ActionT>
Trajectory<ActionT> run_one_game(std::shared_ptr<GameState<ActionT>> game_state, const std::vector<std::shared_ptr<Agent<ActionT>>>& agents, const std::shared_ptr<Agent<ActionT>> random_agent, bool verbose=false, bool save_states=false, bool save_freqs=false) {
    Trajectory<ActionT> trajectory;
    trajectory.initial_state = game_state->clone();

    int active_player = game_state->active_player();
    while (!game_state->is_terminal()) {
        const auto& agent = active_player == CHANCE_PLAYER ? random_agent : agents[active_player];
        const auto action_info = agent->get_action_info(game_state);
        const ActionT& action = action_info.action;

        if (verbose) {
            std::cout << "\n" << *game_state << "\n";
            std::cout << "selected action: " << action << "\n";
        }

        trajectory.actions.push_back(action);
        game_state->apply_action(action);
        if (save_states) { // debug feature. Normally you should be able to reconstruct all states from initial_state and actions
            trajectory.states.push_back(game_state->clone());
        }
        if (save_freqs) {
            trajectory.freqs.push_back(action_info.freqs);
        }
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
    int num_workers;
    unsigned int random_seed;
    bool verbose;
    bool save_states;
    bool save_freqs;
    std::string dump_trajectories;

    GameSeriesTask(const json& jsn) {
        if (!jsn.contains("agents") || !jsn.contains("num_games")) {
            throw std::runtime_error("JSON must contain 'agents' and 'num_games' fields.");
        }

        num_games = jsn.at("num_games");
        num_workers = jsn.value("num_workers", 1);
        dump_trajectories = jsn.value("dump_trajectories", "");
        random_seed = jsn.value("random_seed", 11); 
        verbose = jsn.value("verbose", false);
        save_states = jsn.value("save_states", false); 
        save_freqs = jsn.value("save_freqs", false); 
        for (const auto& player_config : jsn.at("agents")) {
            agents.push_back(construct_agent(player_config));
        }
    }
};

template<typename GameStateT, typename ActionT>
std::vector<Trajectory<ActionT>> run_games(const GameSeriesTask<ActionT>& task) {
    std::vector<Trajectory<ActionT>> trajectories;
    std::shared_ptr<Agent<ActionT>> random_agent = std::make_shared<RandomAgent<ActionT>>();

    auto start = std::chrono::high_resolution_clock::now();
    for (int game_num = 0; game_num < task.num_games; ++game_num) {

        auto game_state = std::make_shared<GameStateT>(task.agents.size());
        Trajectory<ActionT> trajectory = run_one_game<ActionT>(game_state, task.agents, random_agent, task.verbose, task.save_states, task.save_freqs);
        trajectories.push_back(trajectory);

        std::cout << "Game " << game_num << " of " << task.num_games 
            << " actions: " << trajectory.actions.size()
            << "\n";
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = static_cast<double>((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count()) / 1000.0;
    std::cout << "Took: "<< duration << " sec, " << duration / task.num_games << " seconds per game\n";

    return trajectories;
}

template<typename GameStateT, typename ActionT>
std::vector<Trajectory<ActionT>> run_games_parallel(const GameSeriesTask<ActionT>& task) {
    // this implementation is suboptimal, but simple
    std::vector<Trajectory<ActionT>> all_trajectories;
    all_trajectories.reserve(task.num_games);
    std::vector<std::future<std::vector<Trajectory<ActionT>>>> futures;

    int num_workers = task.num_workers;
    GameSeriesTask<ActionT> subtask = task;
    subtask.num_games = task.num_games / num_workers;

    for (int i = 0; i < num_workers; ++i) {
        futures.push_back(std::async(std::launch::async, [subtask]() {
            return run_games<GameStateT, ActionT>(subtask);
        }));
    }
    for (auto& future : futures) {
        auto worker_trajectories = future.get();
        all_trajectories.insert(all_trajectories.end(), worker_trajectories.begin(), worker_trajectories.end());
    }

    return all_trajectories;
}