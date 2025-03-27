#pragma once

#include "splendor.h"
#include "game.h"
#include "json.hpp"

void splendor_stats(const std::vector<Trajectory<splendor::Action>>& trajectories);
void dump_trajectories(const std::string& file_name, const std::vector<Trajectory<splendor::Action>>& trajectories);
void run_model(const nlohmann::json& task_json);
