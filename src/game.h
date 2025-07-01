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
    size_t first_player = 0;        // index of the player to move first
    std::vector<int> actions;       // list of actions taken by agents. First action corresponds to agents[first_player]
    std::vector<double> rewards;    // obtained at the end of the game, given in the order of agents in the game_state
    std::vector<std::shared_ptr<GameState>> states;         // optional, full list of game states encountered during the game
    std::vector<std::vector<std::pair<int, int>>> freqs;    // optional, mcts root node action counts for each state
};

Trajectory run_one_game(std::shared_ptr<GameState> game_state, 
                       const std::vector<std::shared_ptr<Agent>>& agents, 
                       const std::shared_ptr<Agent> random_agent, 
                       bool verbose=false, 
                       bool save_states=false, 
                       bool save_freqs=false,
                       size_t first_player = 0);

class GameSeriesTask {
public:
    std::vector<std::shared_ptr<Agent>> agents;
    int num_games;
    int num_workers;
    unsigned int random_seed;
    bool verbose;
    bool rotate_agents;             // changes the order of players at each game
    bool save_states;
    bool save_freqs;
    int win_points;
    std::string dump_trajectories;

    GameSeriesTask(const json& jsn);
};

std::vector<Trajectory> run_games(const GameSeriesTask& task);
std::vector<Trajectory> run_games_parallel(const GameSeriesTask& task);
void dump_trajectories(const std::string& file_name, const std::vector<Trajectory>& trajectories);
