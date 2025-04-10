#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <torch/torch.h>
#include <torch/script.h>

#include "mcts.h"

namespace mcts {

class NNPolicy : public mcts::Policy {
private:
    mutable torch::jit::script::Module model;
    const std::shared_ptr<mcts::GameStateEncoder> state_encoder;
    const torch::Device device;
    
public:
    NNPolicy(const std::string& model_path, 
             const std::shared_ptr<mcts::GameStateEncoder>& state_encoder,
             torch::Device device = torch::kCPU);
    
    std::vector<double> predict(const std::shared_ptr<GameState> game_state) const override;
    
    ~NNPolicy() override = default;
};

} // namespace mcts