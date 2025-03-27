set(SOURCE_FILES
    main.cpp
    splendor.cpp
    util.cpp
    splendor_agents.cpp
    random_util.cpp
)

add_executable(splendor ${SOURCE_FILES})

target_link_libraries(splendor "${TORCH_LIBRARIES}")

target_include_directories(splendor PRIVATE .)
