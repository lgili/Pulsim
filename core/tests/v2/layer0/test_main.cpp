// pulsim_v2_layer0_tests — main entry point.
//
// Catch2's `Catch2WithMain` link target provides main() for us; this
// file exists to anchor the test binary's translation unit list in
// CMake. Per-feature tests live in sibling files.

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
