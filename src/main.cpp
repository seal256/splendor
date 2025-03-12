#include <iostream>

#include "splendor.h"
#include "mcts.h"
#include "agents.h"

using splendor::Action;
using splendor::SplendorGameState;

int main() {
    auto mcts_params = mcts::MCTSParams();
    mcts_params.iterations = 100000;
    auto mcts_agent = std::make_shared<mcts::MCTSAgent<Action>>(mcts_params);
    auto random_agent = std::make_shared<RandomAgent<Action>>();
    std::vector<std::shared_ptr<Agent<Action>>> agents = {random_agent, mcts_agent};

    auto game_state = std::make_shared<SplendorGameState>(agents.size());
    Trajectory<Action> trajectory = game_round<Action>(game_state, agents);
    
    std::cout << "done!\n";
    return 0;
}
