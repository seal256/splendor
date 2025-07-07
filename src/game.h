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
    std::vector<std::string> agent_names;                   // names of agents in order of their moves in the game (useful when the order is changed from game to game by rotate_agents option)
    std::vector<int> actions;                               // list of actions taken by agents. 
    std::vector<double> rewards;                            // rewards obtained at the end of the game
    std::vector<std::shared_ptr<GameState>> states;         // optional, full list of game states encountered during the game
    std::vector<std::vector<std::pair<int, int>>> freqs;    // optional, mcts root node action counts for each state
};

Trajectory run_one_game(std::shared_ptr<GameState> game_state, 
                       const std::vector<std::shared_ptr<Agent>>& agents, 
                       std::shared_ptr<const Agent> random_agent,           // selects cards in choice states
                       bool verbose=false, 
                       bool save_states=false,                              // saves all states (usually only initial state is saved)
                       bool save_freqs=false);                              // saves mcts visit frequencies for each move in each state 

class GameSeriesTask {
public:
    std::vector<std::shared_ptr<Agent>> agents;
    int num_games;
    int num_workers = 1;                // num cores for parallel execution
    unsigned int random_seed = 0;       // [warn] do not lead to reproducible games in parallel execution setup (num_workers > 1)
    bool verbose = false;
    bool rotate_agents = false;         // changes the order of players at each game
    bool save_states = false;           // saves all states (usually only initial state is saved)
    bool save_freqs = false;            // saves mcts visit frequencies for each move in each state 
    int win_points = 15;                // changes game rules
    std::string dump_trajectories;      // name of the file to dump resulting game trajectories

    explicit GameSeriesTask(const json& jsn);
};

std::vector<Trajectory> run_games(const GameSeriesTask& task);
std::vector<Trajectory> run_games_parallel(const GameSeriesTask& task);
void dump_trajectories(const std::string& file_name, const std::vector<Trajectory>& trajectories);
