#pragma once

#include "agents.h"
#include "splendor.h"

#include "json.hpp"
using json = nlohmann::json;


std::shared_ptr<Agent<splendor::Action>> construct_agent(const json& jsn);


