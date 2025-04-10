#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cmath>
#include <numeric>
#include <future>
#include <chrono>
#include <json.hpp>

#include "splendor_agents.h"

using json = nlohmann::json;

struct Trajectory {
    std::shared_ptr<GameState> initial_state;
    std::vector<int> actions;
    std::vector<double> rewards; // obtained at the end of the game
    std::vector<std::shared_ptr<GameState>> states; // optional
    std::vector<std::vector<std::pair<int, int>>> freqs; // Optional, root node action counts for each state
};

Trajectory run_one_game(std::shared_ptr<GameState> game_state, 
                       const std::vector<std::shared_ptr<Agent>>& agents, 
                       const std::shared_ptr<Agent> random_agent, 
                       bool verbose=false, 
                       bool save_states=false, 
                       bool save_freqs=false);

class GameSeriesTask {
public:
    std::vector<std::shared_ptr<Agent>> agents;
    int num_games;
    int num_workers;
    unsigned int random_seed;
    bool verbose;
    bool save_states;
    bool save_freqs;
    std::string dump_trajectories;

    GameSeriesTask(const json& jsn);
};

std::vector<Trajectory> run_games(const GameSeriesTask& task);
std::vector<Trajectory> run_games_parallel(const GameSeriesTask& task);
void dump_trajectories(const std::string& file_name, const std::vector<Trajectory>& trajectories);
