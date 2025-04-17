#pragma once

#include <vector>

#include "game_state.h"

namespace mcts {

constexpr int UNKNOWN = -2;

struct Node {
    int action; // the action that was applied to the parent and lead to this state
    Node* parent;
    int acting_player; // active_player at the parent state (who made the action from parent node to this one)
    std::vector<std::shared_ptr<Node>> children;
    int visits;
    double wins; // wins for the acting_player (who took the action)
    double p; // prior action probability provided by an action policy

    Node(int action = 0, Node* parent = nullptr, int acting_player = UNKNOWN);
};

struct MCTSParams {
    int iterations = 1000;              // number of iterations of the search procedure
    double exploration = 1.4;           // exploration constant C
    int weighted_selection_moves = -1;  // before this move number in the game best action will be selected with weights, then greedy
    int max_choice_children = 100;      // max children of a chance node. The rest are discarded and not included in the search tree
    double p_noise_level = 0.25;        // fraction of noise in the probabilities estimated by the policy
    double alpha = 0.03;                // dirichlet distribution constant
    double value_weight = 0.5;          // Weight for combining rollout and value estimates
    int max_rollout_len = 500;
    bool use_selection_policy = true;
    bool use_rollout_policy = false;
};

class MCTS {
protected:
    const std::shared_ptr<GameState> root_state;
    std::shared_ptr<Node> root;
    const MCTSParams params;

public:
    MCTS(const std::shared_ptr<GameState> & state, const MCTSParams & params = MCTSParams());
    virtual ~MCTS() = default;

    int search();
    virtual void search_iteration();
    int best_action();
    std::vector<std::pair<int, int>> root_visits() const;
    void apply_action(int action);

protected:
    virtual std::shared_ptr<Node> select_child(const std::shared_ptr<GameState> state, const std::shared_ptr<Node> node);
    virtual void expand_node(const std::shared_ptr<GameState> state, std::shared_ptr<Node> node);
    virtual std::vector<double> rollout(std::shared_ptr<GameState> state);
    virtual std::vector<double> random_rollout(std::shared_ptr<GameState> state);
};

class Policy {
public:
    // Returns probabilities of available actions
    virtual std::vector<double> predict(const std::shared_ptr<GameState> game_state, const std::vector<int>& actions) const = 0;
    virtual ~Policy() = default;
};

class GameStateEncoder {
public:
    virtual std::vector<float> encode(const std::shared_ptr<GameState> game_state) const = 0;
    virtual ~GameStateEncoder() = default;
};

class PolicyMCTS: public MCTS {
private:
    const std::shared_ptr<Policy> policy;
    
public:
    PolicyMCTS(const std::shared_ptr<GameState>& state, const std::shared_ptr<Policy>& policy, const MCTSParams& params = MCTSParams());

protected:
    std::shared_ptr<Node> select_child(const std::shared_ptr<GameState> state, const std::shared_ptr<Node> node) override;
    void expand_node(const std::shared_ptr<GameState> state, std::shared_ptr<Node> node) override;
    std::vector<double> rollout(std::shared_ptr<GameState> state) override;
    std::vector<double> ploicy_rollout(std::shared_ptr<GameState> state);
};

class Value {
public:
    // Estimates the value of the game_state for each player
    virtual std::vector<double> predict(const std::shared_ptr<GameState> game_state) const = 0;
    virtual ~Value() = default;
};

class PVMCTS : public MCTS {
private:
    const std::shared_ptr<Policy> policy;
    const std::shared_ptr<Value> value;
    
public:
    PVMCTS(const std::shared_ptr<GameState>& state, 
           const std::shared_ptr<Policy>& policy,
           const std::shared_ptr<Value>& value,
           const MCTSParams& params = MCTSParams());

protected:
    std::shared_ptr<Node> select_child(const std::shared_ptr<GameState> state, const std::shared_ptr<Node> node) override;
    void expand_node(const std::shared_ptr<GameState> state, std::shared_ptr<Node> node) override;    
    std::vector<double> rollout(std::shared_ptr<GameState> state) override;
    std::vector<double> random_rollout(std::shared_ptr<GameState> state) override;
    std::vector<double> ploicy_rollout(std::shared_ptr<GameState> state);
};

} // namespace mcts