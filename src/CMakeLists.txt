set(SOURCE_FILES
    agents.cpp
    game_state.cpp
    game.cpp
    main.cpp
    mcts.cpp
    nn_policy.cpp
    random_util.cpp
    splendor.cpp
    splendor_agents.cpp
    util.cpp
)

add_executable(splendor ${SOURCE_FILES})

target_link_libraries(splendor "${TORCH_LIBRARIES}")

target_include_directories(splendor PRIVATE .)
