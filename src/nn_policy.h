#include <memory>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

#include "mcts.h"

// https://pytorch.org/tutorials/advanced/cpp_export.html

namespace mcts {

template<typename ActionT>
class NNPolicy : public mcts::Policy<ActionT> {
private:
    mutable torch::jit::script::Module model_; // since the model_.eval() is called, it's actually const
    const std::shared_ptr<mcts::GameStateEncoder<ActionT>> state_encoder_;
    const std::unordered_map<std::string, size_t>& action_ids_;
    const torch::Device device_;
    
public:
    NNPolicy(const std::string& model_path, 
        const std::shared_ptr<mcts::GameStateEncoder<ActionT>>& state_encoder, 
        const std::unordered_map<std::string, size_t>& action_ids, 
        torch::Device device = torch::kCPU)
        : state_encoder_(state_encoder), action_ids_(action_ids), device_(device) {
        try {
            model_ = torch::jit::load(model_path);
            model_.to(device_);
            model_.eval();
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading model: " + std::string(e.what()));
        }
    }

    std::vector<double> predict(const std::shared_ptr<GameState<ActionT>> game_state) const override {       
        std::vector<float> state_vec = state_encoder_->encode(game_state); // Converts the game state to a feature vector
        
        torch::Tensor input_tensor = torch::from_blob(state_vec.data(), {static_cast<int32_t>(state_vec.size())}, torch::kFloat32).to(device_);
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        torch::Tensor output_tensor;
        try {
            output_tensor = model_.forward(inputs).toTensor();
        } catch (const c10::Error& e) {
            throw std::runtime_error("Model forward pass failed: " + std::string(e.what()));
        }
        
        output_tensor = output_tensor.to(torch::kCPU).contiguous();
        float* output_data = output_tensor.data_ptr<float>();
                
        // Maps probabilities to legal actions (the model returns a full list for all possible actions)
        auto legal_actions = game_state->get_actions();
        std::vector<double> action_probs;
        action_probs.reserve(legal_actions.size());
        for (const auto& action : legal_actions) {
            auto action_id = action_ids_.at(action.to_str());
            action_probs.push_back(static_cast<double>(output_data[action_id]));
        }
    
        // Renormalization
        // double sum = std::accumulate(action_probs.begin(), action_probs.end(), 0.0);
        // if (sum > 0.0) {
        //     for (auto& prob : action_probs) {
        //         prob /= sum;
        //     }
        // } else {
        //     std::fill(action_probs.begin(), action_probs.end(), 1.0/legal_actions.size());
        // }

        return action_probs;
    }


    
    ~NNPolicy() override = default;
};

} // namespace mcts