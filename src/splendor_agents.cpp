#include "splendor_agents.h"

using splendor::Action;

std::shared_ptr<Agent<Action>> construct_agent(const json& jsn) {
    if (!jsn.contains("type")) {
        throw std::runtime_error("JSON configuration must contain a 'type' field.");
    }
    std::string agent_type = jsn["type"];

    if (agent_type == "RandomAgent") {
        return std::make_shared<RandomAgent<Action>>();

    } else if (agent_type == "MCTSAgent") {
        mcts::MCTSParams params;
        if (jsn.contains("iterations")) {
            params.iterations = jsn["iterations"];
        }
        if (jsn.contains("exploration")) {
            params.exploration = jsn["exploration"];
        }
        return std::make_shared<MCTSAgent<Action>>(params);

    } else if (agent_type == "PolicyMCTSAgent") {
        mcts::MCTSParams params;
        if (jsn.contains("iterations")) {
            params.iterations = jsn["iterations"];
        }
        if (jsn.contains("exploration")) {
            params.exploration = jsn["exploration"];
        }
        std::vector<double> probs = jsn["probs"];
        std::shared_ptr<mcts::Policy<Action>> policy = std::make_shared<ConstantPolicy<Action>>(probs, splendor::ACTION_ID);
        return std::make_shared<PolicyMCTSAgent<Action>>(policy, params);

    } else {
        throw std::runtime_error("Unknown agent type: " + agent_type);
    }
}