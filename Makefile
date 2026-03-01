# Top-level Makefile for pulsim-core
# Usage:
#  make            # configure + build (default target)
#  make configure  # run cmake configure in ./build
#  make build      # build all targets
#  make lib        # build library target only
#  make cli        # build CLI executable
#  make grpc       # build gRPC server
#  make tests      # build and run C++ tests
#  make pytest     # build Python module and run pytest
#  make test-all   # run both C++ and Python tests
#  make run RUN_BIN=build/cli/pulsim  # run a built binary
#  make run-grpc   # run the gRPC server
#  make clean      # remove build directory
#  make distclean  # clean + git clean (use with care)

BUILD_DIR ?= build
BUILD_TYPE ?= Debug
BUILD_TESTS ?= ON
JOBS ?= $(shell sysctl -n hw.ncpu 2>/dev/null || echo 2)
CMAKE ?= cmake
CTEST ?= ctest

# Default binaries (can be overridden on command line)
RUN_BIN ?= $(BUILD_DIR)/cli/pulsim
GRPC_BIN ?= $(BUILD_DIR)/api-grpc/pulsim_grpc_server

.PHONY: all help configure build lib cli grpc python tests pytest test-all run run-grpc clean distclean

all: build

help:
	@printf "Available targets:\n"
	@printf "  make configure        - run CMake configure (into $(BUILD_DIR))\n"
	@printf "  make build            - configure (if needed) and build all targets\n"
	@printf "  make lib              - build core library only\n"
	@printf "  make cli              - build CLI executable\n"
	@printf "  make grpc             - build gRPC server\n"
	@printf "  make python           - build Python module\n"
	@printf "  make tests            - build and run C++ tests\n"
	@printf "  make pytest           - build Python module and run pytest\n"
	@printf "  make test-all         - run both C++ and Python tests\n"
	@printf "  make run RUN_BIN=...  - run a built binary (default $(RUN_BIN))\n"
	@printf "  make run-grpc         - run gRPC server (default $(GRPC_BIN))\n"
	@printf "  make clean            - remove $(BUILD_DIR)\n"
	@printf "  make distclean        - remove build and untracked files (git clean -fdx)\n"

configure:
	@mkdir -p $(BUILD_DIR)
	@echo "Configuring (build dir: $(BUILD_DIR))"
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DPULSIM_BUILD_TESTS=$(BUILD_TESTS) ..

build: configure
	@echo "Building (jobs=$(JOBS))"
	@$(CMAKE) --build $(BUILD_DIR) -- -j$(JOBS)

lib: configure
	@echo "Building library target"
	@$(CMAKE) --build $(BUILD_DIR) --target pulsim_core -- -j$(JOBS)

cli: configure
	@echo "Building CLI target"
	@$(CMAKE) --build $(BUILD_DIR) --target pulsim -- -j$(JOBS)

grpc: configure
	@echo "Building gRPC server target"
	@$(CMAKE) --build $(BUILD_DIR) --target pulsim_grpc_server -- -j$(JOBS)

python:
	@echo "Building Python module"
	@mkdir -p $(BUILD_DIR)
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DPULSIM_BUILD_PYTHON=ON >/dev/null
	@$(CMAKE) --build $(BUILD_DIR) --target _pulsim -- -j$(JOBS)
	@cp -r python/tests $(BUILD_DIR)/python/ 2>/dev/null || true

tests:
	@echo "Running tests (convenience mode: forcing PULSIM_BUILD_TESTS=ON and Debug build)"
	@mkdir -p $(BUILD_DIR)
	@echo "Configuring $(BUILD_DIR) with tests enabled..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DPULSIM_BUILD_TESTS=ON >/dev/null || true
	@echo "Building project test target"
	@$(CMAKE) --build $(BUILD_DIR) --target pulsim_tests -- -j$(JOBS)
	@# Test executable may be placed under a subdirectory (e.g. core/). Try common locations.
	@TEST_BIN="$(BUILD_DIR)/pulsim_tests"; \
	if [ ! -x $$TEST_BIN ]; then TEST_BIN="$(BUILD_DIR)/core/pulsim_tests"; fi; \
	if [ -x $$TEST_BIN ]; then \
		echo "Running test executable: $$TEST_BIN"; \
		$$TEST_BIN; \
	else \
		echo "Test binary not found: $$TEST_BIN"; \
		echo "You can also run 'ctest --test-dir $(BUILD_DIR)' to run all registered tests."; \
		exit 2; \
	fi

pytest: python
	@echo "Running Python tests with pytest"
	@pytest $(BUILD_DIR)/python/tests/ -v

test-all: tests pytest
	@echo "All tests completed"

run:
	@echo "Running: $(RUN_BIN)"
	@if [ -x "$(RUN_BIN)" ]; then $(RUN_BIN) $(ARGS); else echo "Binary not found or not executable: $(RUN_BIN)"; exit 2; fi

run-grpc:
	@echo "Running gRPC server: $(GRPC_BIN)"
	@if [ -x "$(GRPC_BIN)" ]; then $(GRPC_BIN) $(ARGS); else echo "Binary not found or not executable: $(GRPC_BIN)"; exit 2; fi

clean:
	@echo "Removing $(BUILD_DIR)"
	@rm -rf $(BUILD_DIR)

distclean: clean
	@echo "Running git clean -fdx (untracked files will be removed)"
	@git clean -fdx
