#pragma once

#include <vector>

#include "game_state.h"

namespace mcts {

constexpr int UNKNOWN = -2;

struct Node {
    int action;         // the action that was applied to the parent and lead to this state
    Node* parent;
    int acting_player;  // active_player at the parent state (who made the action from parent node to this one)
    std::vector<std::shared_ptr<Node>> children;
    int visits;
    double wins;        // wins for the acting_player (the one who took the action)
    double p;           // prior action probability provided by an action policy

    Node(int action = 0, Node* parent = nullptr, int acting_player = UNKNOWN);
};

struct MCTSParams {
    int iterations = 1000;              // number of iterations of the search procedure
    double exploration = 1.4;           // exploration constant C
    int weighted_selection_moves = -1;  // before this move number in the game best action will be selected with weights, then greedy
    int max_chance_children = 100;      // max children of a chance node. The rest are discarded and not included in the search tree
    double p_noise_level = 0.25;        // fraction of noise in the probabilities estimated by the policy
    double alpha = 0.03;                // dirichlet distribution constant
    double value_weight = 0.5;          // weight for combining rollout and value estimates
    int max_rollout_len = 500;          // max game steps in the rollout phase, applies only to MCTS variations with value function
    bool use_selection_policy = true;   // applies policy for move selection
    bool use_rollout_policy = false;    // applies policy to rollouts (default rollout policy is random)
    bool train = false;                 // train mode enables train-specific features like noise and weighted move selection
};

class MCTS {
protected:
    std::shared_ptr<GameState> root_state;
    std::shared_ptr<Node> root;
    const MCTSParams params;

public:
    MCTS(std::shared_ptr<const GameState> state, const MCTSParams & params = MCTSParams());
    virtual ~MCTS() = default;

    // Runs search iterations and returns best action
    int search();

    // One expansion and rollout 
    virtual void search_iteration();

    // Best action index according to accumulated root node visit counts
    int best_action();

    // List of action to visit count pairs
    std::vector<std::pair<int, int>> root_visits() const;

    // Applies action to root state, keeping the corresponding search subtree intact
    void apply_action(int action);

protected:
    virtual std::shared_ptr<Node> select_child(std::shared_ptr<const GameState> state, std::shared_ptr<const Node> node);
    virtual void expand_node(std::shared_ptr<const GameState> state, std::shared_ptr<Node> node);
    virtual std::vector<double> rollout(std::shared_ptr<GameState> state);
    virtual std::vector<double> random_rollout(std::shared_ptr<GameState> state);
};

class Policy {
public:
    // Returns probabilities of available actions
    virtual std::vector<double> predict(std::shared_ptr<const GameState> game_state, const std::vector<int>& actions) const = 0;
    virtual ~Policy() = default;
};

// Encodes game state into a vector to be processed by nn
class GameStateEncoder {
public:
    virtual std::vector<float> encode(std::shared_ptr<const GameState> game_state) const = 0;
    virtual ~GameStateEncoder() = default;
};

// Augments basic MCTS with policy
class PolicyMCTS: public MCTS {
private:
    std::shared_ptr<const Policy> policy;
    
public:
    PolicyMCTS(std::shared_ptr<const GameState> state, std::shared_ptr<const Policy> policy, const MCTSParams& params = MCTSParams());

protected:
    std::shared_ptr<Node> select_child(std::shared_ptr<const GameState> state, std::shared_ptr<const Node> node) override;
    void expand_node(std::shared_ptr<const GameState> state, std::shared_ptr<Node> node) override;
    std::vector<double> rollout(std::shared_ptr<GameState> state) override;
    std::vector<double> policy_rollout(std::shared_ptr<GameState> state);
};

class Value {
public:
    // Estimates the value of the game_state for each player
    virtual std::vector<double> predict(std::shared_ptr<const GameState> game_state) const = 0;
    virtual ~Value() = default;
};

// Augments basic MCTS with state value estimation
class ValueMCTS : public MCTS {
private:
    std::shared_ptr<const Value> value;
    
public:
    ValueMCTS(std::shared_ptr<const GameState> state, 
            std::shared_ptr<const Value> value,
            const MCTSParams& params = MCTSParams());

protected:
    std::vector<double> rollout(std::shared_ptr<GameState> state) override;
};

// Combines policy and value with separate implementations
class PVMCTS : public MCTS {
private:
    std::shared_ptr<const Policy> policy;
    std::shared_ptr<const Value> value;
    
public:
    PVMCTS(std::shared_ptr<const GameState> state, 
           std::shared_ptr<const Policy> policy,
           std::shared_ptr<const Value> value,
           const MCTSParams& params = MCTSParams());

protected:
    std::shared_ptr<Node> select_child(std::shared_ptr<const GameState> state, std::shared_ptr<const Node> node) override;
    void expand_node(std::shared_ptr<const GameState> state, std::shared_ptr<Node> node) override;    
    std::vector<double> rollout(std::shared_ptr<GameState> state) override;
    std::vector<double> random_rollout(std::shared_ptr<GameState> state) override;
    std::vector<double> policy_rollout(std::shared_ptr<GameState> state);
};

} // namespace mcts