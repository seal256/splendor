#include <iostream>
#include <fstream>

#include "json.hpp"
using nlohmann::json;

#include "splendor.h"
#include "agents.h"
#include "game.h"
#include "util.h"

using splendor::Action;
using splendor::SplendorGameState;

const json DEFAULT_TASK = {
    {"agents", {
        {{"type", "MCTSAgent"}, {"iterations", 1000}, {"exploration", 1.4}},
        {{"type", "MCTSAgent"}, {"iterations", 1000}, {"exploration", 1.4}}
    }},
    {"num_games", 5},
    {"verbose", true}
};

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
        GameSeriesTask<Action> task(task_json);
        std::vector<Trajectory<Action>> trajectories = task.num_workers > 1 ? 
            run_games_parallel<SplendorGameState, Action>(task) : run_games<SplendorGameState, Action>(task);
        
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
