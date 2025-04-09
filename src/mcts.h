#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <chrono>

#include "game_state.h"
#include "random_util.h"

namespace mcts {

constexpr int UNKNOWN = -2;

template<typename ActionT>
struct Node {
    ActionT action; // the action that was applied to the parent and lead to this state
    Node<ActionT>* parent;
    int acting_player; // active_player at the parent state (who made the action from parent node to this one)
    std::vector<std::shared_ptr<Node>> children;
    int visits;
    double wins; // wins for the acting_player (who took the action)
    double p; // prior action probability provided by an action policy

    Node(const ActionT& action = ActionT(), Node<ActionT>* parent = nullptr, int acting_player = UNKNOWN)
        : action(action), parent(parent), acting_player(acting_player), visits(0), wins(0), p(1.0) {}
};

struct MCTSParams {
    int iterations = 1000; // number of iterations
    double exploration = 1.4; // exploration constant C
    int weighted_selection_moves = -1; // before this move number in the game best action will be selected with weights, then greedy
    int max_choice_children = 100;
    double value_weight = 0.5; // Weight for combining rollout and value estimates
    int max_rollout_len = 500;
    bool use_selection_policy = true;
    bool use_rollout_policy = false;
};

template<typename ActionT>
class MCTS {
protected:
    const std::shared_ptr<GameState<ActionT>> root_state;
    std::shared_ptr<Node<ActionT>> root;
    const MCTSParams params;

public:
    MCTS(const std::shared_ptr<GameState<ActionT>> & state, const MCTSParams & params = MCTSParams())
        : root_state(state->clone()), root(std::make_shared<Node<ActionT>>()), params(params) {}

    ActionT search() {
        // Grows the search tree and returns the best expected action
        for (int iter = 0; iter < params.iterations; ++iter) {
            this->search_iteration();
        }
        return best_action();
    }

    virtual void search_iteration() {
        // This implementation is suited for memory intensive game setups: 
        // game states are big, next state computation is relatively cheap.  
        // In such conditions it is more efficient to store only action in each node, 
        // and recompute state evolution on the fly, starting from the root on each iteration.
        // This avoids excessive memory consumption and state copying cost.

        // Selection
        auto node = root;
        auto state = root_state->clone();
        while (!node->children.empty() && node->visits > 0) { // visits > 0 assures that we performed at least one rollout from each new node
            node = this->select_child(state, node);
            state->apply_action(node->action);
        }

        // Expansion
        if (!state->is_terminal() && node->children.empty() && node->visits > 0) {
            this->expand_node(state, node);
            if (!node->children.empty()) {
                node = this->select_child(state, node);
                state->apply_action(node->action);
            }
        }

        // Simulation
        std::vector<double> rewards = this->rollout(state);

        // Backpropagation
        Node<ActionT> * bp_node = node.get();
        while (bp_node) {
            bp_node->visits++;
            if (!bp_node->parent) { // skips root node
                break;
            }
            if (bp_node->acting_player == CHANCE_PLAYER) { // do not record wins, but downscale the rewards
                if (!bp_node->parent->children.empty()) {
                    for (auto& r : rewards) {
                        r /= bp_node->parent->children.size();
                    }
                }
            } else {
                bp_node->wins += rewards[bp_node->acting_player];
            }
            bp_node = bp_node->parent;
        }
    }

    ActionT best_action() {
        // Returns best action at the root state
        if (root->children.empty()) {
            throw std::runtime_error("No actions available");
        }
        if (root_state->move_num() > params.weighted_selection_moves) { // select by max visits
            auto best_child = std::max_element(root->children.begin(), root->children.end(),
                [](const std::shared_ptr<Node<ActionT>>& a, const std::shared_ptr<Node<ActionT>>& b) {
                    return a->visits < b->visits;
                });
            return (*best_child)->action;

        } else { // select with probabilites proportional to visits
            std::vector<int> visits;
            visits.reserve(root->children.size());
            for (const auto& child : root->children) {
                visits.emplace_back(child->visits);
            }
            size_t child_idx = weighted_random_choice(visits);
            return root->children[child_idx]->action;
        }
    }

    std::vector<std::pair<ActionT, int>> root_visits() const {
        // Helper function that returns children visits at the root state
        std::vector<std::pair<ActionT, int>> visits;
        visits.reserve(root->children.size());
        for (const auto& child : root->children) {
            visits.emplace_back(child->action, child->visits);
        }
        return visits;
    }

    void apply_action(const ActionT& action) {
        // Applies an actual game action and rebases the tree root.
        // Maintains a relevant part of the tree for next searches. 
        root_state->apply_action(action);
        
        auto next_node = std::find_if(root->children.begin(), root->children.end(),
            [&action](const std::shared_ptr<Node<ActionT>>& child) {
                return child->action == action;
            });

        if (next_node != root->children.end()) {
            root = *next_node;
            root->parent = nullptr;

        } else { // unexplored action
            root = std::make_shared<Node<ActionT>>(action, nullptr, root_state->active_player());
        }
    }

protected:
    virtual std::shared_ptr<Node<ActionT>> select_child(const std::shared_ptr<GameState<ActionT>> state, const std::shared_ptr<Node<ActionT>> node) {
        if (state->active_player() == CHANCE_PLAYER) {
            int random_idx = rand() % node->children.size();
            return node->children[random_idx];
        }

        // Use the first unexplored child, assuming the children array is shuffled
        for (const auto& child : node->children) {
            if (child->visits == 0) {
                return child; 
            }
        }

        // Find the max UCB child
        double max_ucb = -std::numeric_limits<double>::infinity(); // could use 0, but just in case of negative rewards
        std::shared_ptr<Node<ActionT>> best_child = nullptr;
        double log_parent_visits = std::log(node->visits);
        for (const auto& child : node->children) {
            double exploitation_term = child->wins / child->visits;
            double exploration_term = std::sqrt(log_parent_visits / child->visits);
            double ucb = exploitation_term + params.exploration * exploration_term;
            if (ucb > max_ucb) {
                max_ucb = ucb;
                best_child = child;
            }
        }
        if (best_child == nullptr) {
            throw std::runtime_error("Unable to find best child!");
        }
        return best_child;
    }

    virtual void expand_node(const std::shared_ptr<GameState<ActionT>> state, std::shared_ptr<Node<ActionT>> node) {
        int acting_player = state->active_player();
        for (const auto& action : state->get_actions()) {
            auto child_node = std::make_shared<Node<ActionT>>(action, node.get(), acting_player);
            node->children.push_back(child_node);
        }
        random_shuffle(node->children.begin(), node->children.end()); // Optional step, that simplifies selection phase
        if (acting_player == CHANCE_PLAYER && node->children.size() > params.max_choice_children) {
            node->children.resize(params.max_choice_children);
        }
    }

    virtual std::vector<double> rollout(std::shared_ptr<GameState<ActionT>> state) {
        return this->random_rollout(state);
    }

    virtual std::vector<double> random_rollout(std::shared_ptr<GameState<ActionT>> state) {
        // Simulates a random playout from the given state
        while (!state->is_terminal()) {
            auto actions = state->get_actions();
            if (actions.empty()) 
                break;
            int random_idx = rand() % actions.size();
            state->apply_action(actions[random_idx]);
        }
        return state->rewards();
    }

};

template<typename ActionT>
class Policy {
public:
    // Retunrs probabilities of available actions in the order provided by game_state->get_actions()
    virtual std::vector<double> predict(const std::shared_ptr<GameState<ActionT>> game_state) const = 0;
    virtual ~Policy() {};
};

template<typename ActionT>
class GameStateEncoder {
public:
    virtual std::vector<float> encode(const std::shared_ptr<GameState<ActionT>> game_state) const = 0;
    virtual ~GameStateEncoder() {};
};

template<typename ActionT>
class PolicyMCTS: public MCTS<ActionT> {
private:
    const std::shared_ptr<Policy<ActionT>> policy;
    
public:
    PolicyMCTS(const std::shared_ptr<GameState<ActionT>>& state, const std::shared_ptr<Policy<ActionT>>& policy, const MCTSParams& params = MCTSParams())
        : MCTS<ActionT>(state, params), policy(policy) {}

protected:
    std::shared_ptr<Node<ActionT>> select_child(const std::shared_ptr<GameState<ActionT>> state, const std::shared_ptr<Node<ActionT>> node) override {
        if (state->active_player() == CHANCE_PLAYER) {
            int random_idx = rand() % node->children.size();
            return node->children[random_idx];
        }
        
        // Find the max UCB child
        double max_ucb = -1;
        std::shared_ptr<Node<ActionT>> best_child = nullptr;
        double parent_visits_sqrts = std::sqrt(node->visits);
        for (const auto& child : node->children) {
            double exploitation_term = child->visits > 0 ? child->wins / child->visits : 0;
            double exploration_term = child->p * parent_visits_sqrts / (child->visits + 1);
            double ucb = exploitation_term + this->params.exploration * exploration_term;
            if (ucb > max_ucb) {
                max_ucb = ucb;
                best_child = child;
            }
        }
        if (best_child == nullptr) {
            throw std::runtime_error("Unable to find best child!");
        }
        return best_child;
    }

    virtual void expand_node(const std::shared_ptr<GameState<ActionT>> state, std::shared_ptr<Node<ActionT>> node) override {
        const std::vector<ActionT> actions = state->get_actions(); 
        int acting_player = state->active_player();
        const std::vector<double> probs = acting_player == CHANCE_PLAYER || !this->params.use_selection_policy ?
            std::vector<double>(actions.size(), 1.0) : policy->predict(state); 
        for (size_t id = 0; id < actions.size(); id++) { // assumes that get_actions orders actions consistently
            auto child_node = std::make_shared<Node<ActionT>>(actions[id], node.get(), acting_player);
            child_node->p = probs[id];
            node->children.push_back(child_node);
        }
        random_shuffle(node->children.begin(), node->children.end()); // Optional step, that simplifies selection phase
        if (acting_player == CHANCE_PLAYER && node->children.size() > this->params.max_choice_children) {
            node->children.resize(this->params.max_choice_children);
        }
    }

    std::vector<double> rollout(std::shared_ptr<GameState<ActionT>> state) override {
        return this->params.use_rollout_policy ? this->ploicy_rollout(state) : this->random_rollout(state);
    }

    std::vector<double> ploicy_rollout(std::shared_ptr<GameState<ActionT>> state) {
        // Simulates a policy directed playout from the given state
        while (!state->is_terminal()) {
            auto actions = state->get_actions();
            if (actions.empty()) 
                break;
            int random_idx;
            if (state->active_player() == CHANCE_PLAYER) {
                random_idx = rand() % actions.size();
            } else {
                const auto probs = this->policy->predict(state);
                random_idx = weighted_random_choice(probs);
            }
            state->apply_action(actions[random_idx]);
        }
        return state->rewards();
    }
};

template<typename ActionT>
class Value {
public:
    // Estimates the value of the game_state for each player
    virtual std::vector<double> predict(const std::shared_ptr<GameState<ActionT>> game_state) const = 0;
    virtual ~Value() {};
};

template<typename ActionT>
class PVMCTS : public MCTS<ActionT> {
private:
    const std::shared_ptr<Policy<ActionT>> policy;
    const std::shared_ptr<Value<ActionT>> value;
    
public:
    PVMCTS(const std::shared_ptr<GameState<ActionT>>& state, 
           const std::shared_ptr<Policy<ActionT>>& policy,
           const std::shared_ptr<Value<ActionT>>& value,
           const MCTSParams& params = MCTSParams())
        : MCTS<ActionT>(state, params), policy(policy), value(value) {}

protected:

    virtual std::shared_ptr<Node<ActionT>> select_child(const std::shared_ptr<GameState<ActionT>> state, const std::shared_ptr<Node<ActionT>> node) override {
        if (state->active_player() == CHANCE_PLAYER) {
            int random_idx = rand() % node->children.size();
            return node->children[random_idx];
        }
        
        // Find the max UCB child
        double max_ucb = -1;
        std::shared_ptr<Node<ActionT>> best_child = nullptr;
        double parent_visits_sqrts = std::sqrt(node->visits);
        for (const auto& child : node->children) {
            double exploitation_term = child->visits > 0 ? child->wins / child->visits : 0;
            double exploration_term = child->p * parent_visits_sqrts / (child->visits + 1);
            double ucb = exploitation_term + this->params.exploration * exploration_term;
            if (ucb > max_ucb) {
                max_ucb = ucb;
                best_child = child;
            }
        }
        if (best_child == nullptr) {
            throw std::runtime_error("Unable to find best child!");
        }
        return best_child;
    }

    virtual void expand_node(const std::shared_ptr<GameState<ActionT>> state, std::shared_ptr<Node<ActionT>> node) override {
        const std::vector<ActionT> actions = state->get_actions(); 
        int acting_player = state->active_player();
        const std::vector<double> probs = acting_player == CHANCE_PLAYER || !this->params.use_selection_policy ?
            std::vector<double>(actions.size(), 1.0) : policy->predict(state); 
        for (size_t id = 0; id < actions.size(); id++) { // assumes that get_actions orders actions consistently
            auto child_node = std::make_shared<Node<ActionT>>(actions[id], node.get(), acting_player);
            child_node->p = probs[id];
            node->children.push_back(child_node);
        }
        random_shuffle(node->children.begin(), node->children.end()); // Optional step, that simplifies selection phase
        if (acting_player == CHANCE_PLAYER && node->children.size() > this->params.max_choice_children) {
            node->children.resize(this->params.max_choice_children);
        }
    }    

    std::vector<double> rollout(std::shared_ptr<GameState<ActionT>> state) override {
        std::vector<double> predicted_values = value->predict(state);
        std::vector<double> rewards = this->params.use_rollout_policy ? this->ploicy_rollout(state) : this->random_rollout(state);
        
        // Combine rollout and value estimates
        if (state->is_terminal()) { // rollout reached terminal state
            double wv = this->params.value_weight;
            for (size_t player = 0; player < rewards.size(); ++player) {
                rewards[player] = (1.0 - wv) * rewards[player] + wv * predicted_values[player];
            }
        }

        return rewards;
    }
    
    std::vector<double> random_rollout(std::shared_ptr<GameState<ActionT>> state) override {
        // Simulates a random playout from the given state
        for (int t = 0; t < this->params.max_rollout_len && !state->is_terminal(); t++) {
            auto actions = state->get_actions();
            if (actions.empty()) 
                break;
            int random_idx = rand() % actions.size();
            state->apply_action(actions[random_idx]);
        }
        if (state->is_terminal()) {
            return state->rewards();
        } else {
            return value->predict(state);
        }
    }

    std::vector<double> ploicy_rollout(std::shared_ptr<GameState<ActionT>> state) {
        // Simulates a policy directed playout from the given state
        for (int t = 0; t < this->params.max_rollout_len && !state->is_terminal(); t++) {
            auto actions = state->get_actions();
            if (actions.empty()) 
                break;
            int random_idx;
            if (state->active_player() == CHANCE_PLAYER) {
                random_idx = rand() % actions.size();
            } else {
                const auto probs = this->policy->predict(state);
                random_idx = weighted_random_choice(probs);
            }
            state->apply_action(actions[random_idx]);
        }
        if (state->is_terminal()) {
            return state->rewards();
        } else {
            return value->predict(state);
        }
    }
};


}; // namespace mcts