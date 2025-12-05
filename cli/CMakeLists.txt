# CLI11 for command-line parsing
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cli11)

add_executable(spicelab
    src/main.cpp
)

target_link_libraries(spicelab
    PRIVATE
        spicelab::core
        CLI11::CLI11
)

# Install target
install(TARGETS spicelab
    RUNTIME DESTINATION bin
)
