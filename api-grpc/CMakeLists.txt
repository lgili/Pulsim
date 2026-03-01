set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Try to use system-installed Protobuf/gRPC first. Only fetch gRPC sources
# if the packages were not found on the system. This avoids pulling in
# large dependency trees (abseil, etc.) which can create conflicting test
# targets when FetchContent reconfigures nested projects.
find_package(Protobuf CONFIG QUIET)
find_package(gRPC CONFIG QUIET)

if(NOT Protobuf_FOUND OR NOT gRPC_FOUND)
    include(FetchContent)

    # gRPC and Protobuf (fetched only when not available on the system)
    set(ABSL_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
    set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_CODEGEN ON CACHE BOOL "" FORCE)
    set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        grpc
        GIT_REPOSITORY https://github.com/grpc/grpc.git
        GIT_TAG v1.59.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(grpc)

    find_package(Protobuf CONFIG REQUIRED)
    find_package(gRPC CONFIG REQUIRED)
endif()

# cpp-httplib for metrics endpoint
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.14.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(httplib)

set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_FILES ${PROTO_DIR}/simulator.proto)

set(GENERATED_PROTO_SRC)
set(GENERATED_PROTO_HDR)
set(GENERATED_GRPC_SRC)
set(GENERATED_GRPC_HDR)

foreach(proto ${PROTO_FILES})
    get_filename_component(proto_name ${proto} NAME_WE)
    # Place generated sources under a predictable include path that matches
    # the project's headers (so includes like
    # "pulsim/api/grpc/simulator.grpc.pb.h" resolve correctly).
    set(GEN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pulsim/api/grpc)
    set(proto_src "${GEN_INCLUDE_DIR}/${proto_name}.pb.cc")
    set(proto_hdr "${GEN_INCLUDE_DIR}/${proto_name}.pb.h")
    set(grpc_src "${GEN_INCLUDE_DIR}/${proto_name}.grpc.pb.cc")
    set(grpc_hdr "${GEN_INCLUDE_DIR}/${proto_name}.grpc.pb.h")

    add_custom_command(
        OUTPUT ${proto_src} ${proto_hdr} ${grpc_src} ${grpc_hdr}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_INCLUDE_DIR}
        COMMAND $<TARGET_FILE:protobuf::protoc>
        ARGS --grpc_out ${GEN_INCLUDE_DIR}
             --cpp_out ${GEN_INCLUDE_DIR}
             --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
             -I ${PROTO_DIR}
             -I ${grpc_SOURCE_DIR}/third_party/protobuf/src
             ${proto}
        DEPENDS ${proto}
        COMMENT "Generating gRPC sources for ${proto_name}.proto"
    )

    list(APPEND GENERATED_PROTO_SRC ${proto_src})
    list(APPEND GENERATED_PROTO_HDR ${proto_hdr})
    list(APPEND GENERATED_GRPC_SRC ${grpc_src})
    list(APPEND GENERATED_GRPC_HDR ${grpc_hdr})
endforeach()

add_library(pulsim_grpc_proto STATIC
    ${GENERATED_PROTO_SRC}
    ${GENERATED_GRPC_SRC}
)

add_library(pulsim::grpc_proto ALIAS pulsim_grpc_proto)

target_include_directories(pulsim_grpc_proto
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(pulsim_grpc_proto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
)

set(PULSIM_GRPC_SOURCES
    src/server.cpp
    src/main.cpp
    src/session_manager.cpp
    src/server_config.cpp
    src/job_queue.cpp
    src/metrics.cpp
)

add_executable(pulsim_grpc_server
    ${PULSIM_GRPC_SOURCES}
)

target_include_directories(pulsim_grpc_server
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(pulsim_grpc_server
    PRIVATE
        pulsim::core
        pulsim::grpc_proto
        gRPC::grpc++
        gRPC::grpc++_reflection
        protobuf::libprotobuf
        httplib::httplib
)

add_dependencies(pulsim_grpc_server pulsim_grpc_proto)

install(TARGETS pulsim_grpc_server
    RUNTIME DESTINATION bin
)

# Integration tests for gRPC API
if(PULSIM_BUILD_TESTS)
    add_executable(pulsim_grpc_tests
        tests/test_grpc_api.cpp
        src/server.cpp
        src/session_manager.cpp
        src/server_config.cpp
        src/job_queue.cpp
        src/metrics.cpp
    )

    target_include_directories(pulsim_grpc_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}
    )

    target_link_libraries(pulsim_grpc_tests
        PRIVATE
            pulsim::core
            pulsim::grpc_proto
            gRPC::grpc++
            gRPC::grpc++_reflection
            protobuf::libprotobuf
            httplib::httplib
            Catch2::Catch2WithMain
    )

    add_dependencies(pulsim_grpc_tests pulsim_grpc_proto)

    add_test(NAME grpc_api_tests COMMAND pulsim_grpc_tests)
endif()
