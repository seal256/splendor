#include "splendor_agents.h"

using splendor::Action;

// class CheatMCTSAgent : public Agent<Action> {
// private:
//     mcts::MCTSParams mcts_params;

// public:
//     CheatMCTSAgent(const mcts::MCTSParams& params = mcts::MCTSParams()) : mcts_params(params) {}

//     Action get_action(const std::shared_ptr<GameState<Action>>& game_state) const override {
//         std::vector<double> probs(splendor::ACTION_ID.size(), 1);
//         if (game_state->active_player() != CHANCE_PLAYER) {
//             mcts::MCTS<Action> mcts(game_state, mcts_params);
//             mcts.search();

//             double count_sum = 0.0;
//             const auto root_visits = mcts.root_visits();
//             for (const auto& pr : root_visits) {
//                 probs[splendor::ACTION_ID.at(pr.first.to_str())] = pr.second;
//                 count_sum += pr.second;
//             }
//             count_sum += (splendor::ACTION_ID.size() - root_visits.size());

//             for (auto& p: probs) {
//                 p /= count_sum;
//             }            
//         }
//         std::shared_ptr<mcts::Policy<Action>> policy = std::make_shared<ConstantPolicy<Action>>(probs, splendor::ACTION_ID);
        
//         mcts::PolicyMCTS<Action> policy_mcts(game_state, policy, mcts_params);
//         Action action = policy_mcts.search();
        
//         return action;
//     }
// };


mcts::MCTSParams parse_mcts_params(const json& jsn) {
    mcts::MCTSParams params;
    if (jsn.contains("iterations")) {
        params.iterations = jsn["iterations"];
    }
    if (jsn.contains("exploration")) {
        params.exploration = jsn["exploration"];
    }
    return params;
}

std::shared_ptr<Agent<Action>> construct_agent(const json& jsn) {
    if (!jsn.contains("type")) {
        throw std::runtime_error("JSON configuration must contain a 'type' field.");
    }
    std::string agent_type = jsn["type"];

    if (agent_type == "RandomAgent") {
        return std::make_shared<RandomAgent<Action>>();

    } else if (agent_type == "MCTSAgent") {
        mcts::MCTSParams params = parse_mcts_params(jsn);
        return std::make_shared<MCTSAgent<Action>>(params);

    } else if (agent_type == "PolicyMCTSAgent") {
        mcts::MCTSParams params = parse_mcts_params(jsn);
        std::vector<double> probs = jsn["probs"];
        std::shared_ptr<mcts::Policy<Action>> policy = std::make_shared<ConstantPolicy<Action>>(probs, splendor::ACTION_ID);
        return std::make_shared<PolicyMCTSAgent<Action>>(policy, params);

    // } else if (agent_type == "CheatMCTSAgent") {
    //     mcts::MCTSParams params = parse_mcts_params(jsn);
    //     return std::make_shared<CheatMCTSAgent>(params);

    } else {
        throw std::runtime_error("Unknown agent type: " + agent_type);
    }
}