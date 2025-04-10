#pragma once

#include "splendor.h"
#include "game.h"
#include "json.hpp"

void splendor_stats(const std::vector<Trajectory>& trajectories);
void run_model(const nlohmann::json& task_json);
