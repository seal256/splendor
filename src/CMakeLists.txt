set(SOURCE_FILES
    main.cpp
    splendor.cpp
)

add_executable(splendor ${SOURCE_FILES})

target_include_directories(splendor PRIVATE .)
