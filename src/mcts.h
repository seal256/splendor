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
    double p; // prior action probability

    Node(const ActionT& action = ActionT(), Node<ActionT>* parent = nullptr, int active_player = 0)
        : action(action), parent(parent), active_player(active_player), visits(0), wins(0), p(1.0) {}
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


// template<typename GameStateT>
// class Policy {
// public:
//     virtual std::vector<double> predict(const GameStateT& game_state) = 0;
//     virtual ~Policy() {};
// };


template<typename ActionT>
class PolicyMCTS {
private:
    const std::shared_ptr<GameState<ActionT>> root_state;
    std::shared_ptr<Node<ActionT>> root;
    MCTSParams params;
    const std::vector<double> probs = {0.0034,0.0064,0.0039,0.0033,0.0044,0.0042,0.0293,0.0329,0.0352,0.0330,0.0327,0.0370,0.0239,0.0235,0.0285,0.0286,0.0215,0.0189,0.0214,0.0220,0.0136,0.0134,0.0150,0.0168,0.0101,0.0102,0.0108,0.0143,0.0512,0.0572,0.0647,0.0846,0.0238,0.0238,0.0250,0.0285,0.0047,0.0057,0.0059,0.0068,0.0404,0.0355,0.0240};
    const std::vector<std::string> actions = {"s","tr2","tg2","tb2","tw2","tk2","tr1g1b1","tr1g1w1","tr1g1k1","tr1b1w1","tr1b1k1","tr1w1k1","tg1b1w1","tg1b1k1","tg1w1k1","tb1w1k1",
               "r0n0","r0n1","r0n2","r0n3","r1n0","r1n1","r1n2","r1n3","r2n0","r2n1","r2n2","r2n3",
               "p0n0","p0n1","p0n2","p0n3","p1n0","p1n1","p1n2","p1n3","p2n0","p2n1","p2n2","p2n3",
               "h0","h1","h2"};
    std::unordered_map<std::string, size_t> action_id;

public:
    PolicyMCTS(const std::shared_ptr<GameState<ActionT>> & state, const MCTSParams & params = MCTSParams())
        : root_state(state->clone()), params(params) {
        root = std::make_shared<Node<ActionT>>();
        root->active_player = root_state->active_player();
        for (size_t id = 0; id < actions.size(); id++) {
            action_id[actions[id]] = id;
        }
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
            for (const auto& action : state->get_actions()) {
                auto child_node = std::make_shared<Node<ActionT>>(action, node.get(), state->active_player());
                child_node->p = probs[action_id[action.to_str()]];
                node->children.emplace_back(child_node);
            }
            std::random_shuffle(node->children.begin(), node->children.end()); // Optional step
            if (!node->children.empty()) {
                node = _select_child(node); // Assumes that the children are shuffled. Otherwize pick random index
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
        // // Use the first unexplored child, assuming the children array is shuffled
        // for (const auto& child : node->children) {
        //     if (child->visits == 0) {
        //         return child; 
        //     }
        // }

        // Find the max UCB child
        double max_ucb = -std::numeric_limits<double>::infinity(); // could use 0, but just in case of negative rewards
        std::shared_ptr<Node<ActionT>> best_child = nullptr;
        double log_parent_visits = std::log(node->visits + 1);
        for (const auto& child : node->children) {
            double exploitation_term = child->wins / (child->visits + 1);
            double exploration_term = child.p * std::sqrt(log_parent_visits / (child->visits + 1));
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