#include "json.hpp"
using nlohmann::json;

#include "splendor.h"
#include "game.h"

void splendor_stats(const std::vector<Trajectory<splendor::Action>>& trajectories);
void dump_trajectories(const std::string& file_name, const std::vector<Trajectory<splendor::Action>>& trajectories);

