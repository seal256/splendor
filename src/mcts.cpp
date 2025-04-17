#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <chrono>

#include "mcts.h"
#include "random_util.h"

namespace mcts {

Node::Node(int action, Node* parent, int acting_player) 
    : action(action), parent(parent), acting_player(acting_player), visits(0), wins(0), p(1.0) {}

MCTS::MCTS(const std::shared_ptr<GameState> & state, const MCTSParams & params)
    : root_state(state->clone()), root(std::make_shared<Node>()), params(params) {}

int MCTS::search() {
    for (int iter = 0; iter < params.iterations; ++iter) {
        this->search_iteration();
    }
    return best_action();
}

void MCTS::search_iteration() {
    auto node = root;
    auto state = root_state->clone();
    while (!node->children.empty() && node->visits > 0) {
        node = this->select_child(state, node);
        state->apply_action(node->action);
    }

    if (!state->is_terminal() && node->children.empty() && node->visits > 0) {
        this->expand_node(state, node);
        if (!node->children.empty()) {
            node = this->select_child(state, node);
            state->apply_action(node->action);
        }
    }

    std::vector<double> rewards = this->rollout(state);

    Node * bp_node = node.get();
    while (bp_node) {
        bp_node->visits++;
        if (!bp_node->parent) {
            break;
        }
        if (bp_node->acting_player == CHANCE_PLAYER) {
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

int MCTS::best_action() {
    if (root->children.empty()) {
        throw std::runtime_error("No actions available");
    }
    if (root_state->move_num() > params.weighted_selection_moves) {
        auto best_child = std::max_element(root->children.begin(), root->children.end(),
            [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
                return a->visits < b->visits;
            });
        return (*best_child)->action;
    } else {
        std::vector<int> visits;
        visits.reserve(root->children.size());
        for (const auto& child : root->children) {
            visits.emplace_back(child->visits);
        }
        size_t child_idx = weighted_random_choice(visits);
        return root->children[child_idx]->action;
    }
}

std::vector<std::pair<int, int>> MCTS::root_visits() const {
    std::vector<std::pair<int, int>> visits;
    visits.reserve(root->children.size());
    for (const auto& child : root->children) {
        visits.emplace_back(child->action, child->visits);
    }
    return visits;
}

void MCTS::apply_action(int action) {
    root_state->apply_action(action);
    
    auto next_node = std::find_if(root->children.begin(), root->children.end(),
        [&action](const std::shared_ptr<Node>& child) {
            return child->action == action;
        });

    if (next_node != root->children.end()) {
        root = *next_node;
        root->parent = nullptr;
    } else {
        root = std::make_shared<Node>(action, nullptr, root_state->active_player());
    }
}

std::shared_ptr<Node> MCTS::select_child(const std::shared_ptr<GameState> state, const std::shared_ptr<Node> node) {
    if (state->active_player() == CHANCE_PLAYER) {
        int random_idx = rand() % node->children.size();
        return node->children[random_idx];
    }

    for (const auto& child : node->children) {
        if (child->visits == 0) {
            return child; 
        }
    }

    double max_ucb = -std::numeric_limits<double>::infinity();
    std::shared_ptr<Node> best_child = nullptr;
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

void MCTS::expand_node(const std::shared_ptr<GameState> state, std::shared_ptr<Node> node) {
    int acting_player = state->active_player();
    for (const auto& action : state->get_actions()) {
        auto child_node = std::make_shared<Node>(action, node.get(), acting_player);
        node->children.push_back(child_node);
    }
    random_shuffle(node->children.begin(), node->children.end());
    if (acting_player == CHANCE_PLAYER && node->children.size() > params.max_choice_children) {
        node->children.resize(params.max_choice_children);
    }
}

std::vector<double> MCTS::rollout(std::shared_ptr<GameState> state) {
    return this->random_rollout(state);
}

std::vector<double> MCTS::random_rollout(std::shared_ptr<GameState> state) {
    while (!state->is_terminal()) {
        auto actions = state->get_actions();
        if (actions.empty()) 
            break;
        int random_idx = rand() % actions.size();
        state->apply_action(actions[random_idx]);
    }
    return state->rewards();
}

PolicyMCTS::PolicyMCTS(const std::shared_ptr<GameState>& state, const std::shared_ptr<Policy>& policy, const MCTSParams& params)
    : MCTS(state, params), policy(policy) {}

std::shared_ptr<Node> PolicyMCTS::select_child(const std::shared_ptr<GameState> state, const std::shared_ptr<Node> node) {
    if (state->active_player() == CHANCE_PLAYER) {
        int random_idx = rand() % node->children.size();
        return node->children[random_idx];
    }
    
    double max_ucb = -1;
    std::shared_ptr<Node> best_child = nullptr;
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

void PolicyMCTS::expand_node(const std::shared_ptr<GameState> state, std::shared_ptr<Node> node) {
    const std::vector<int> actions = state->get_actions(); 
    int acting_player = state->active_player();
    bool use_policy = acting_player != CHANCE_PLAYER && this->params.use_selection_policy;
    std::vector<double> probs;
    if (use_policy) {
        probs = policy->predict(state, actions); 
        double eps = this->params.p_noise_level;
        if (eps > 0) {
            const auto noise = sample_dirichlet(this->params.alpha, probs.size());
            for (size_t n = 0; n < probs.size(); n++) {
                probs[n] = (1.0 - eps) * probs[n] + eps * noise[n];
            }
        }
    }
    for (size_t id = 0; id < actions.size(); id++) {
        auto child_node = std::make_shared<Node>(actions[id], node.get(), acting_player);
        child_node->p = use_policy ? probs[id] : 1.0;
        node->children.push_back(child_node);
    }
    random_shuffle(node->children.begin(), node->children.end());
    if (acting_player == CHANCE_PLAYER && node->children.size() > this->params.max_choice_children) {
        node->children.resize(this->params.max_choice_children);
    }
}

std::vector<double> PolicyMCTS::rollout(std::shared_ptr<GameState> state) {
    return this->params.use_rollout_policy ? this->ploicy_rollout(state) : this->random_rollout(state);
}

std::vector<double> PolicyMCTS::ploicy_rollout(std::shared_ptr<GameState> state) {
    while (!state->is_terminal()) {
        auto actions = state->get_actions();
        if (actions.empty()) 
            break;
        int random_idx;
        if (state->active_player() == CHANCE_PLAYER) {
            random_idx = rand() % actions.size();
        } else {
            const auto probs = this->policy->predict(state, actions);
            random_idx = weighted_random_choice(probs);
        }
        state->apply_action(actions[random_idx]);
    }
    return state->rewards();
}

PVMCTS::PVMCTS(const std::shared_ptr<GameState>& state, 
       const std::shared_ptr<Policy>& policy,
       const std::shared_ptr<Value>& value,
       const MCTSParams& params)
    : MCTS(state, params), policy(policy), value(value) {}

std::shared_ptr<Node> PVMCTS::select_child(const std::shared_ptr<GameState> state, const std::shared_ptr<Node> node) {
    if (state->active_player() == CHANCE_PLAYER) {
        int random_idx = rand() % node->children.size();
        return node->children[random_idx];
    }
    
    double max_ucb = -1;
    std::shared_ptr<Node> best_child = nullptr;
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

void PVMCTS::expand_node(const std::shared_ptr<GameState> state, std::shared_ptr<Node> node) {
    const std::vector<int> actions = state->get_actions(); 
    int acting_player = state->active_player();
    bool use_policy = acting_player != CHANCE_PLAYER && this->params.use_selection_policy;
    std::vector<double> probs;
    if (use_policy) {
        probs = policy->predict(state, actions); 
    }
    for (size_t id = 0; id < actions.size(); id++) {
        auto child_node = std::make_shared<Node>(actions[id], node.get(), acting_player);
        child_node->p = use_policy ? probs[id] : 1.0;
        node->children.push_back(child_node);
    }
    random_shuffle(node->children.begin(), node->children.end());
    if (acting_player == CHANCE_PLAYER && node->children.size() > this->params.max_choice_children) {
        node->children.resize(this->params.max_choice_children);
    }
}    

std::vector<double> PVMCTS::rollout(std::shared_ptr<GameState> state) {
    std::vector<double> predicted_values = value->predict(state);
    std::vector<double> rewards = this->params.use_rollout_policy ? this->ploicy_rollout(state) : this->random_rollout(state);
    
    if (state->is_terminal()) {
        double wv = this->params.value_weight;
        for (size_t player = 0; player < rewards.size(); ++player) {
            rewards[player] = (1.0 - wv) * rewards[player] + wv * predicted_values[player];
        }
    }

    return rewards;
}

std::vector<double> PVMCTS::random_rollout(std::shared_ptr<GameState> state) {
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

std::vector<double> PVMCTS::ploicy_rollout(std::shared_ptr<GameState> state) {
    for (int t = 0; t < this->params.max_rollout_len && !state->is_terminal(); t++) {
        auto actions = state->get_actions();
        if (actions.empty()) 
            break;
        int random_idx;
        if (state->active_player() == CHANCE_PLAYER) {
            random_idx = rand() % actions.size();
        } else {
            const auto probs = this->policy->predict(state, actions);
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

} // namespace mcts