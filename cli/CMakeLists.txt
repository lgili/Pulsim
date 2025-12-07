# CLI11 for command-line parsing
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cli11)

# Optional HDF5 support
option(PULSIM_WITH_HDF5 "Enable HDF5 output support" OFF)
if(PULSIM_WITH_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS CXX)
endif()

# Optional Parquet/Arrow support
option(PULSIM_WITH_PARQUET "Enable Parquet output support" OFF)
if(PULSIM_WITH_PARQUET)
    find_package(Arrow REQUIRED)
    find_package(Parquet REQUIRED)
endif()

add_executable(pulsim
    src/main.cpp
)

target_link_libraries(pulsim
    PRIVATE
        pulsim::core
        CLI11::CLI11
        nlohmann_json::nlohmann_json
)

# Add optional gRPC support for serve command
if(PULSIM_BUILD_GRPC AND TARGET pulsim::grpc_proto)
    find_package(gRPC REQUIRED)
    find_package(Protobuf REQUIRED)
    target_compile_definitions(pulsim PRIVATE PULSIM_WITH_GRPC)
    target_link_libraries(pulsim
        PRIVATE
            pulsim::grpc_proto
            gRPC::grpc++
            gRPC::grpc++_reflection
            protobuf::libprotobuf
            httplib::httplib
    )
    target_include_directories(pulsim
        PRIVATE
            ${CMAKE_SOURCE_DIR}/api-grpc/include
            ${CMAKE_BINARY_DIR}/api-grpc
    )
    # Link the server implementation
    target_sources(pulsim PRIVATE
        ${CMAKE_SOURCE_DIR}/api-grpc/src/server.cpp
        ${CMAKE_SOURCE_DIR}/api-grpc/src/session_manager.cpp
        ${CMAKE_SOURCE_DIR}/api-grpc/src/server_config.cpp
    )
endif()

# HDF5 support
if(PULSIM_WITH_HDF5)
    target_compile_definitions(pulsim PRIVATE PULSIM_WITH_HDF5)
    target_link_libraries(pulsim PRIVATE ${HDF5_CXX_LIBRARIES})
    target_include_directories(pulsim PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

# Parquet support
if(PULSIM_WITH_PARQUET)
    target_compile_definitions(pulsim PRIVATE PULSIM_WITH_PARQUET)
    target_link_libraries(pulsim PRIVATE Arrow::arrow_shared Parquet::parquet_shared)
endif()

# Install target
install(TARGETS pulsim
    RUNTIME DESTINATION bin
)
