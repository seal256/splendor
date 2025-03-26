#pragma once

#include "agents.h"
#include "splendor.h"

#include "json.hpp"


std::shared_ptr<Agent<splendor::Action>> construct_agent(const nlohmann::json& jsn);


