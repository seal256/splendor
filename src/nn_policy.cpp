#include "nn_policy.h"
#include <stdexcept>
#include <numeric>

namespace mcts {

NNPolicy::NNPolicy(const std::string& model_path, 
                   const std::shared_ptr<mcts::GameStateEncoder>& state_encoder,
                   torch::Device device)
    : state_encoder(state_encoder), device(device) {
    try {
        model = torch::jit::load(model_path);
        model.to(device);
        model.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}

std::vector<double> NNPolicy::predict(const std::shared_ptr<GameState> game_state, const std::vector<int>& actions) const {       
    std::vector<float> state_vec = state_encoder->encode(game_state);
    
    torch::Tensor input_tensor = torch::from_blob(
        state_vec.data(), 
        {static_cast<int32_t>(state_vec.size())}, 
        torch::kFloat32
    ).to(device);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    
    torch::Tensor output_tensor;
    try {
        output_tensor = model.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Model forward pass failed: " + std::string(e.what()));
    }
    
    output_tensor = output_tensor.to(torch::kCPU).contiguous();
    float* output_data = output_tensor.data_ptr<float>();
            
    std::vector<double> action_probs;
    action_probs.reserve(actions.size());
    
    double sum_probs = 0.0;
    for (const auto& action : actions) {
        double p = static_cast<double>(output_data[action]);
        sum_probs += p;
        action_probs.push_back(p);
    }

    if (sum_probs <= 0.0) {
        sum_probs = 1.0;
    }
    for (auto p : action_probs) {
        p /= sum_probs;
    }

    return action_probs;
}

} // namespace mcts