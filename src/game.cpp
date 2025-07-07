#include <stdexcept>
#include <chrono>
#include <iostream>
#include <fstream>

#include "game.h"

Trajectory run_one_game(std::shared_ptr<GameState> game_state, 
                       const std::vector<std::shared_ptr<Agent>>& agents, 
                       std::shared_ptr<const Agent> random_agent, 
                       bool verbose, 
                       bool save_states, 
                       bool save_freqs) {
    Trajectory trajectory;
    trajectory.initial_state = game_state->clone();
    trajectory.agent_names.reserve(agents.size());
    for (const auto& agent : agents) {
        trajectory.agent_names.push_back(agent->name); // that's unnecessary copying, but Trajectory is fat anyway
    }

    size_t active_player = game_state->active_player();
    size_t num_players = agents.size();
    while (!game_state->is_terminal()) {
        const auto& agent = active_player == CHANCE_PLAYER ? random_agent : agents[active_player];
        const auto action_info = agent->get_action_info(game_state);
        const int action = action_info.action;

        if (verbose) {
            std::cout << "\n" << *game_state << "\n";
            std::cout << "selected action: " << action << "\n";
        }

        trajectory.actions.push_back(action);
        game_state->apply_action(action);
        if (save_states) {
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

GameSeriesTask::GameSeriesTask(const json& jsn) {
    if (!jsn.contains("agents") || !jsn.contains("num_games")) {
        throw std::runtime_error("JSON must contain 'agents' and 'num_games' fields.");
    }

    num_games = jsn.at("num_games");
    num_workers = jsn.value("num_workers", 1);
    dump_trajectories = jsn.value("dump_trajectories", "");
    random_seed = jsn.value("random_seed", 0);
    verbose = jsn.value("verbose", false);
    rotate_agents = jsn.value("rotate_agents", false);
    save_states = jsn.value("save_states", false); 
    save_freqs = jsn.value("save_freqs", false); 
    for (const auto& player_config : jsn.at("agents")) {
        agents.push_back(construct_agent(player_config));
    }
    win_points = jsn.value("win_points", 0);
}

std::vector<std::vector<std::shared_ptr<Agent>>> precompute_rotations(const std::vector<std::shared_ptr<Agent>>& original) {
    std::vector<std::vector<std::shared_ptr<Agent>>> rotations;
    rotations.reserve(original.size());
    
    std::vector<std::shared_ptr<Agent>> current = original;
    for (size_t idx = 0; idx < original.size(); ++idx) {
        rotations.push_back(current);
        std::rotate(current.begin(), current.begin() + 1, current.end());
    }
    
    return rotations;
}

std::vector<Trajectory> run_games(const GameSeriesTask& task) {
    std::vector<Trajectory> trajectories;
    std::shared_ptr<Agent> random_agent = std::make_shared<RandomAgent>("");
    auto rules = std::make_shared<splendor::SplendorGameRules>(*splendor::DEFAULT_RULES.at(task.agents.size()));
    if (task.win_points > 0) {
        rules->win_points = task.win_points;
    }
    const auto& rotations = precompute_rotations(task.agents);

    auto start = std::chrono::high_resolution_clock::now();
    for (int game_num = 0; game_num < task.num_games; ++game_num) {
        auto game_state = std::make_shared<splendor::SplendorGameState>(task.agents.size(), rules);
        const auto& agents = task.rotate_agents ? rotations[game_num % task.agents.size()] : task.agents;
        Trajectory trajectory = run_one_game(game_state, agents, random_agent, task.verbose, task.save_states, task.save_freqs);
        trajectories.push_back(trajectory);

        // std::cout << "Game " << game_num << " of " << task.num_games 
        //           << " actions: " << trajectory.actions.size()
        //           << "\n";
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = static_cast<double>((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count()) / 1000.0;
    std::cout << "Took: "<< duration << " sec, " << duration / task.num_games << " seconds per game\n";

    return trajectories;
}

std::vector<Trajectory> run_games_parallel(const GameSeriesTask& task) {
    std::vector<Trajectory> all_trajectories;
    all_trajectories.reserve(task.num_games);
    std::vector<std::future<std::vector<Trajectory>>> futures;

    int num_workers = task.num_workers;
    GameSeriesTask subtask = task;
    subtask.num_games = task.num_games / num_workers;

    for (int i = 0; i < num_workers; ++i) {
        futures.push_back(std::async(std::launch::async, [subtask]() {
            return run_games(subtask);
        }));
    }
    for (auto& future : futures) {
        auto worker_trajectories = future.get();
        all_trajectories.insert(all_trajectories.end(), worker_trajectories.begin(), worker_trajectories.end());
    }

    return all_trajectories;
}

void to_json(json& j, const Trajectory& traj) {   
    j = {
        {"initial_state", traj.initial_state},
        {"agent_names", traj.agent_names},
        {"rewards", traj.rewards},
        {"actions", traj.actions}
    };

    if (!traj.states.empty()) {
        j["states"] = traj.states;
    }

    if (!traj.freqs.empty()) {
        j["freqs"] = traj.freqs;
    }
}

void dump_trajectories(const std::string& file_name, const std::vector<Trajectory>& trajectories) {
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