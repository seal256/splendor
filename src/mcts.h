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
    int active_player; // who's turn it is now, at the current state
    std::vector<std::shared_ptr<Node>> children;
    int visits;
    double wins; // wins for the parent's active_player (who took the action)
    double p; // prior action probability

    Node(const ActionT& action = ActionT(), Node<ActionT>* parent = nullptr, int active_player = UNKNOWN)
        : action(action), parent(parent), active_player(active_player), visits(0), wins(0), p(1.0) {}
};

struct MCTSParams {
    int iterations = 1000; // number of iterations
    double exploration = 1.4; // exploration constant C
    int weighted_selection_moves = -1; // before this move number in the game best action will be selected with weights, then greedy
};

template<typename ActionT>
class MCTS {
private:
    const std::shared_ptr<GameState<ActionT>> root_state;
    std::shared_ptr<Node<ActionT>> root;
    const MCTSParams params;

public:
    MCTS(const std::shared_ptr<GameState<ActionT>> & state, const MCTSParams & params = MCTSParams())
        : root_state(state->clone()), params(params) {
        root = std::make_shared<Node<ActionT>>();
        root->active_player = root_state->active_player();
        root->visits = 1;
        _expand_node(root_state, root);
    }

    ActionT search() {
        // Grows the search tree and returns the best expected action
        for (int iter = 0; iter < params.iterations; ++iter) {
            _search_iteration();
        }
        return best_action();
    }

    void _search_iteration() {
        // This implementation is suited for memory intensive game setups: 
        // game states are big, next state computation is relatively cheap.  
        // In such conditions it is more efficient to store only action in each node, 
        // and recompute state evolution on the fly, starting from the root on each iteration.
        // This avoids excessive memory consumption and state copying cost.

        // Selection
        auto node = root;
        auto state = root_state->clone();
        while (!node->children.empty() && node->visits > 0) { // visits > 0 assures that we performed at least one rollout from each new node
            node = _select_child(node);
            state->apply_action(node->action);
        }
        if (node->visits == 0) { // first visit of the node
            node->active_player = state->active_player();
        }

        // Expansion
        if (!state->is_terminal() && node->children.empty() && node->visits > 0) {
            _expand_node(state, node);
            if (!node->children.empty()) {
                node = _select_child(node);
                state->apply_action(node->action);
                node->active_player = state->active_player();
            }
        }

        // Simulation
        std::vector<double> rewards = _rollout(state);

        // Backpropagation
        Node<ActionT> * bp_node = node.get();
        while (bp_node->parent) {
            bp_node->visits++;
            if (bp_node->active_player == CHANCE_PLAYER) { // downscale the rewards that came from a chance node
                if (!bp_node->children.empty()) {
                    for (auto& r : rewards) {
                        r /= bp_node->children.size();
                    }
                }
            }

            int acting_player = bp_node->parent->active_player; // who made the move that lead to this state
            if (acting_player != CHANCE_PLAYER) { // do not record wins for chance actions
                bp_node->wins += rewards[acting_player];
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

private:
    std::shared_ptr<Node<ActionT>> _select_child(const std::shared_ptr<Node<ActionT>>& node) {
        if (node->active_player == CHANCE_PLAYER) {
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

    void _expand_node(const std::shared_ptr<GameState<ActionT>> state, std::shared_ptr<Node<ActionT>> node) {
        // Adds children to the node using the corresponding game state
        for (const auto& action : state->get_actions()) {
            auto child_node = std::make_shared<Node<ActionT>>(action, node.get());
            node->children.push_back(child_node);
        }
        random_shuffle(node->children.begin(), node->children.end()); // Optional step, that simplifies selection phase
    }

    std::vector<double> _rollout(std::shared_ptr<GameState<ActionT>> state) {
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
class PolicyMCTS {
private:
    const std::shared_ptr<GameState<ActionT>> root_state;
    std::shared_ptr<Node<ActionT>> root;
    const std::shared_ptr<Policy<ActionT>> policy;
    MCTSParams params;
    
public:
    PolicyMCTS(const std::shared_ptr<GameState<ActionT>> & state, const std::shared_ptr<Policy<ActionT>>& policy, const MCTSParams & params = MCTSParams())
        : root_state(state->clone()), policy(policy), params(params) {
        root = std::make_shared<Node<ActionT>>();
        root->active_player = root_state->active_player();        
    }

    ActionT search() {
        // Grows the search tree and returns the best expected action
        for (int iter = 0; iter < params.iterations; ++iter) {
            _search_iteration();
        }
        return best_action();
    }

    void _search_iteration() {
        // Selection
        auto node = root;
        auto state = root_state->clone();
        while (!node->children.empty() && node->visits > 0) { // visits > 0 assures that we performed at least one rollout from each new node
            node = _select_child(node);
            state->apply_action(node->action);
        }

        // Expansion
        if (!state->is_terminal() && node->children.empty() && node->visits > 0) {
            const std::vector<ActionT> actions = state->get_actions(); 
            const std::vector<double> probs = state->active_player() == CHANCE_PLAYER ?
                std::vector<double>(actions.size(), 1.0) : policy->predict(state); 
            for (size_t id = 0; id < actions.size(); id++) { // assumes that get_actions orders actions consistently
                auto child_node = std::make_shared<Node<ActionT>>(actions[id], node.get(), state->active_player());
                child_node->p = probs[id];
                node->children.push_back(child_node);
            }
            if (state->active_player() == CHANCE_PLAYER) {
                random_shuffle(node->children.begin(), node->children.end()); // Optional step
            }
            if (!node->children.empty()) {
                node = _select_child(node);
                state->apply_action(node->action);
            }
        }

        // Simulation
        std::vector<double> rewards = _rollout(state);

        // Backpropagation
        Node<ActionT> * bp_node = node.get();
        while (bp_node) {
            bp_node->visits++;
            // active_player may be none if we are at a chance node
            if (bp_node->active_player != CHANCE_PLAYER) {
                bp_node->wins += rewards[bp_node->active_player];
            } else {
                if (!bp_node->children.empty()) {
                    for (auto& r : rewards) {
                        r /= bp_node->children.size();
                    }
                }
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

private:
    std::shared_ptr<Node<ActionT>> _select_child(const std::shared_ptr<Node<ActionT>>& node) {
        if (node->active_player == CHANCE_PLAYER) {
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

    std::vector<double> _rollout(std::shared_ptr<GameState<ActionT>> state) {
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

}; // namespace mcts