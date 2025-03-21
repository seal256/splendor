#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <chrono>

#include "game_state.h"

namespace mcts {

template<typename ActionT>
struct Node {
    ActionT action; // the action that was applied to the parent and lead to this state
    Node<ActionT>* parent;
    int active_player;
    std::vector<std::shared_ptr<Node>> children;
    int visits;
    double wins; // wins for the active_player (who's turn it is at the current step)


    Node(const ActionT& action = ActionT(), Node<ActionT>* parent = nullptr, int active_player = 0)
        : action(action), parent(parent), active_player(active_player), visits(0), wins(0) {}
};

struct MCTSParams {
    int iterations = 1000; // number of iterations
    double exploration = 1.4; // exploration constant C
};

template<typename ActionT>
class MCTS {
private:
    const std::shared_ptr<GameState<ActionT>> root_state;
    std::shared_ptr<Node<ActionT>> root;
    MCTSParams params;

public:
    MCTS(const std::shared_ptr<GameState<ActionT>> & state, const MCTSParams & params = MCTSParams())
        : root_state(state->clone()), params(params) {
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

        // Expansion
        if (!state->is_terminal() && node->children.empty() && node->visits > 0) {
            for (const auto& action : state->get_actions()) {
                auto child_node = std::make_shared<Node<ActionT>>(action, node.get(), state->active_player());
                node->children.emplace_back(child_node);
            }
            std::random_shuffle(node->children.begin(), node->children.end()); // Optional step
            if (!node->children.empty()) {
                node = node->children[0]; // Assumes that the children are shuffled. Otherwize pick random index
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
            }
            bp_node = bp_node->parent;
        }
    }

    ActionT best_action() {
        // Returns best action at the root state
        if (root->children.empty()) {
            throw std::runtime_error("No actions available");
        }
        auto best_child = std::max_element(root->children.begin(), root->children.end(),
            [](const std::shared_ptr<Node<ActionT>>& a, const std::shared_ptr<Node<ActionT>>& b) {
                return a->visits < b->visits;
            });
        return (*best_child)->action;
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