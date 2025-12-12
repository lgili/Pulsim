# PulsimCore v2 - Header-only library
# The v2 API is implemented entirely in headers using C++23 features

# Create an interface (header-only) library
add_library(pulsim_core INTERFACE)

target_include_directories(pulsim_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(pulsim_core INTERFACE cxx_std_23)

target_link_libraries(pulsim_core
    INTERFACE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)

# Alias for cleaner usage
add_library(pulsim::core ALIAS pulsim_core)

# Tests - v2 concepts test (other tests need updating to v2)
if(PULSIM_BUILD_TESTS)
    add_executable(pulsim_tests
        tests/test_main.cpp
        tests/test_v2_concepts.cpp
    )

    target_link_libraries(pulsim_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )

    catch_discover_tests(pulsim_tests)
endif()
