# =============================================================================
# PulsimCore Developer Makefile
# =============================================================================
# Usage:
#   make                  # Configure and build (default)
#   make help             # Show all available targets
#   make configure        # Run CMake configuration
#   make build            # Build all targets
#   make rebuild          # Clean and rebuild
#   make test             # Run C++ tests
#   make pytest           # Run Python tests
#   make test-all         # Run all tests
#   make clean            # Remove build directory
#   make distclean        # Remove all generated files
#   make install          # Install to prefix
#   make format           # Format source code
#   make lint             # Run linters
#
# Variables (override on command line):
#   BUILD_TYPE=Debug|Release|RelWithDebInfo (default: Debug)
#   BUILD_DIR=build (default)
#   JOBS=auto (default: auto-detected CPU count)
#   PREFIX=/usr/local (default)
#   TOOLCHAIN=clang (use Clang toolchain file)
# =============================================================================

# Detect OS
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
ifeq ($(UNAME_S),Darwin)
    OS := macos
    NPROC := $(shell sysctl -n hw.ncpu 2>/dev/null || echo 4)
    LLVM_PREFIX := $(shell brew --prefix llvm 2>/dev/null || echo "/opt/homebrew/opt/llvm")
else ifeq ($(UNAME_S),Linux)
    OS := linux
    NPROC := $(shell nproc 2>/dev/null || echo 4)
    LLVM_PREFIX := $(shell llvm-config --prefix 2>/dev/null || echo "/usr")
else
    OS := windows
    NPROC := $(NUMBER_OF_PROCESSORS)
    LLVM_PREFIX := C:/Program Files/LLVM
endif

# =============================================================================
# Configuration Variables
# =============================================================================
BUILD_DIR ?= build
BUILD_TYPE ?= Debug
JOBS ?= $(NPROC)
PREFIX ?= /usr/local
CMAKE ?= cmake
CTEST ?= ctest
TOOLCHAIN ?=

# CMake base options
CMAKE_OPTIONS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
CMAKE_OPTIONS += -DPULSIM_BUILD_TESTS=ON
CMAKE_OPTIONS += -DPULSIM_BUILD_CLI=ON

# Toolchain selection
ifeq ($(TOOLCHAIN),clang)
    CMAKE_OPTIONS += -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-clang.cmake
endif

# Install prefix
ifdef PREFIX
    CMAKE_OPTIONS += -DCMAKE_INSTALL_PREFIX=$(PREFIX)
endif

# Generator (prefer Ninja if available)
NINJA := $(shell command -v ninja 2>/dev/null)
ifdef NINJA
    CMAKE_GENERATOR := -G Ninja
else
    CMAKE_GENERATOR :=
endif

# Binary paths
CLI_BIN := $(BUILD_DIR)/cli/pulsim
GRPC_BIN := $(BUILD_DIR)/api-grpc/pulsim_grpc_server
TEST_BIN := $(BUILD_DIR)/core/pulsim_tests

# Clang tools
CLANG_FORMAT := $(shell command -v clang-format 2>/dev/null || echo "$(LLVM_PREFIX)/bin/clang-format")
CLANG_TIDY := $(shell command -v clang-tidy 2>/dev/null || echo "$(LLVM_PREFIX)/bin/clang-tidy")

# Python tools
PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)
PYTEST := $(PYTHON) -m pytest
RUFF := $(shell command -v ruff 2>/dev/null || echo "$(PYTHON) -m ruff")

# =============================================================================
# PHONY Targets
# =============================================================================
.PHONY: all help configure reconfigure build rebuild \
        lib cli grpc python \
        test test-quick test-converters test-verbose test-list \
        pytest test-all test-coverage \
        clean distclean \
        install uninstall \
        format format-cpp format-python format-check \
        lint lint-cpp lint-python \
        run run-grpc \
        setup setup-macos setup-windows \
        info check-deps

# =============================================================================
# Default Target
# =============================================================================
all: build

# =============================================================================
# Help
# =============================================================================
help:
	@echo ""
	@echo "PulsimCore Build System"
	@echo "======================="
	@echo ""
	@echo "Build targets:"
	@echo "  make configure      Configure CMake (build dir: $(BUILD_DIR))"
	@echo "  make build          Configure (if needed) and build all targets"
	@echo "  make rebuild        Clean and rebuild everything"
	@echo "  make lib            Build core library only"
	@echo "  make cli            Build CLI executable"
	@echo "  make grpc           Build gRPC server"
	@echo "  make python         Build Python bindings"
	@echo ""
	@echo "Test targets:"
	@echo "  make test           Run all C++ simulation tests"
	@echo "  make test-quick     Run quick converter tests only"
	@echo "  make test-converters Run converter topology tests"
	@echo "  make test-verbose   Run tests with detailed output"
	@echo "  make test-list      List all available tests"
	@echo "  make pytest         Run Python tests"
	@echo "  make test-all       Run all tests (C++ and Python)"
	@echo "  make test-coverage  Run tests with coverage"
	@echo ""
	@echo "Install targets:"
	@echo "  make install        Install to $(PREFIX)"
	@echo "  make uninstall      Uninstall from $(PREFIX)"
	@echo ""
	@echo "Clean targets:"
	@echo "  make clean          Remove $(BUILD_DIR)"
	@echo "  make distclean      Remove all generated files"
	@echo ""
	@echo "Code quality:"
	@echo "  make format         Format C++ and Python code"
	@echo "  make format-cpp     Format C++ code only"
	@echo "  make format-python  Format Python code only"
	@echo "  make format-check   Check formatting without changes"
	@echo "  make lint           Run all linters"
	@echo "  make lint-cpp       Run clang-tidy on C++ code"
	@echo "  make lint-python    Run ruff on Python code"
	@echo ""
	@echo "Run targets:"
	@echo "  make run ARGS=...   Run CLI with arguments"
	@echo "  make run-grpc       Run gRPC server"
	@echo ""
	@echo "Setup targets:"
	@echo "  make setup          Install dependencies for current OS"
	@echo "  make setup-macos    Install macOS dependencies"
	@echo "  make setup-windows  Install Windows dependencies"
	@echo ""
	@echo "Other targets:"
	@echo "  make info           Show build configuration"
	@echo "  make check-deps     Check if dependencies are installed"
	@echo ""
	@echo "Variables (override with VAR=value):"
	@echo "  BUILD_TYPE   Build type: Debug, Release, RelWithDebInfo (current: $(BUILD_TYPE))"
	@echo "  BUILD_DIR    Build directory (current: $(BUILD_DIR))"
	@echo "  JOBS         Parallel jobs (current: $(JOBS))"
	@echo "  PREFIX       Install prefix (current: $(PREFIX))"
	@echo "  TOOLCHAIN    Toolchain: clang (current: $(or $(TOOLCHAIN),default))"
	@echo ""
	@echo "Examples:"
	@echo "  make BUILD_TYPE=Release                    # Release build"
	@echo "  make TOOLCHAIN=clang                       # Use Clang toolchain"
	@echo "  make test JOBS=4                           # Run tests with 4 parallel jobs"
	@echo "  make run ARGS='run examples/rc_circuit.json'"
	@echo ""

# =============================================================================
# Configuration
# =============================================================================
$(BUILD_DIR)/CMakeCache.txt:
	@echo "Configuring CMake ($(BUILD_TYPE), generator: $(or $(CMAKE_GENERATOR),default))..."
	@mkdir -p $(BUILD_DIR)
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_GENERATOR) $(CMAKE_OPTIONS)

configure: $(BUILD_DIR)/CMakeCache.txt
	@echo "Configuration complete."

reconfigure:
	@echo "Reconfiguring CMake..."
	@rm -f $(BUILD_DIR)/CMakeCache.txt
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_GENERATOR) $(CMAKE_OPTIONS)

# =============================================================================
# Build Targets
# =============================================================================
build: $(BUILD_DIR)/CMakeCache.txt
	@echo "Building all targets ($(JOBS) parallel jobs)..."
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

rebuild: clean build

lib: $(BUILD_DIR)/CMakeCache.txt
	@echo "Building core library..."
	$(CMAKE) --build $(BUILD_DIR) --target pulsim_core -j $(JOBS)

cli: $(BUILD_DIR)/CMakeCache.txt
	@echo "Building CLI executable..."
	$(CMAKE) --build $(BUILD_DIR) --target pulsim -j $(JOBS)

grpc: $(BUILD_DIR)/CMakeCache.txt
	@echo "Building gRPC server..."
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_OPTIONS) -DPULSIM_BUILD_GRPC=ON
	$(CMAKE) --build $(BUILD_DIR) --target pulsim_grpc_server -j $(JOBS)

python: $(BUILD_DIR)/CMakeCache.txt
	@echo "Building Python bindings..."
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_OPTIONS) -DPULSIM_BUILD_PYTHON=ON
	$(CMAKE) --build $(BUILD_DIR) --target _pulsim -j $(JOBS)
	@mkdir -p $(BUILD_DIR)/python
	@cp -r python/tests $(BUILD_DIR)/python/ 2>/dev/null || true

# =============================================================================
# Test Targets
# =============================================================================
test: $(BUILD_DIR)/CMakeCache.txt
	@echo "Building and running C++ tests..."
	$(CMAKE) --build $(BUILD_DIR) --target pulsim_simulation_tests -j $(JOBS)
	@$(BUILD_DIR)/core/pulsim_simulation_tests

test-quick: $(BUILD_DIR)/CMakeCache.txt
	@echo "Running quick tests..."
	$(CMAKE) --build $(BUILD_DIR) --target pulsim_simulation_tests -j $(JOBS)
	@$(BUILD_DIR)/core/pulsim_simulation_tests "[quick]"

test-converters: $(BUILD_DIR)/CMakeCache.txt
	@echo "Running converter tests..."
	$(CMAKE) --build $(BUILD_DIR) --target pulsim_simulation_tests -j $(JOBS)
	@$(BUILD_DIR)/core/pulsim_simulation_tests "[converter]"

test-verbose: $(BUILD_DIR)/CMakeCache.txt
	@echo "Running tests with verbose output..."
	$(CMAKE) --build $(BUILD_DIR) --target pulsim_simulation_tests -j $(JOBS)
	@$(BUILD_DIR)/core/pulsim_simulation_tests -s -d

test-list: $(BUILD_DIR)/CMakeCache.txt
	@echo "Available tests:"
	$(CMAKE) --build $(BUILD_DIR) --target pulsim_simulation_tests -j $(JOBS)
	@$(BUILD_DIR)/core/pulsim_simulation_tests --list-tests

pytest: python
	@echo "Running Python tests..."
	$(PYTEST) python/tests/ -v

test-all: test pytest
	@echo ""
	@echo "All tests completed."

test-coverage:
	@echo "Building with coverage..."
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_GENERATOR) $(CMAKE_OPTIONS) \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_FLAGS="--coverage" \
		-DCMAKE_EXE_LINKER_FLAGS="--coverage"
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)
	@echo "Running tests..."
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure
	@echo "Coverage data generated. Use gcov/lcov to generate reports."

# =============================================================================
# Install/Uninstall
# =============================================================================
install: build
	@echo "Installing to $(PREFIX)..."
	$(CMAKE) --install $(BUILD_DIR) --prefix $(PREFIX)

uninstall:
	@echo "Uninstalling from $(PREFIX)..."
	@if [ -f "$(BUILD_DIR)/install_manifest.txt" ]; then \
		xargs rm -vf < $(BUILD_DIR)/install_manifest.txt; \
		echo "Uninstall complete."; \
	else \
		echo "No install manifest found. Nothing to uninstall."; \
	fi

# =============================================================================
# Clean Targets
# =============================================================================
clean:
	@echo "Removing $(BUILD_DIR)..."
	@rm -rf $(BUILD_DIR)

distclean: clean
	@echo "Removing all generated files..."
	@rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__
	@rm -rf *.egg-info dist
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

# =============================================================================
# Code Formatting
# =============================================================================
format: format-cpp format-python
	@echo "Formatting complete."

format-cpp:
	@echo "Formatting C++ code..."
	@if [ -x "$(CLANG_FORMAT)" ] || command -v clang-format >/dev/null 2>&1; then \
		find core cli api-grpc -name "*.cpp" -o -name "*.hpp" -o -name "*.h" 2>/dev/null | \
			xargs $(CLANG_FORMAT) -i -style=file 2>/dev/null || \
			echo "Note: Some files could not be formatted"; \
	else \
		echo "clang-format not found. Install LLVM or run: make setup"; \
	fi

format-python:
	@echo "Formatting Python code..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format python/; \
	elif $(PYTHON) -m ruff --version >/dev/null 2>&1; then \
		$(PYTHON) -m ruff format python/; \
	else \
		echo "ruff not found. Install with: pip install ruff"; \
	fi

format-check:
	@echo "Checking C++ formatting..."
	@if [ -x "$(CLANG_FORMAT)" ] || command -v clang-format >/dev/null 2>&1; then \
		find core cli api-grpc -name "*.cpp" -o -name "*.hpp" -o -name "*.h" 2>/dev/null | \
			xargs $(CLANG_FORMAT) --dry-run --Werror -style=file 2>/dev/null && \
			echo "C++ formatting OK" || echo "C++ formatting issues found"; \
	fi
	@echo "Checking Python formatting..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format --check python/ && echo "Python formatting OK"; \
	fi

# =============================================================================
# Linting
# =============================================================================
lint: lint-cpp lint-python
	@echo "Linting complete."

lint-cpp: $(BUILD_DIR)/CMakeCache.txt
	@echo "Running clang-tidy..."
	@if [ -x "$(CLANG_TIDY)" ] || command -v clang-tidy >/dev/null 2>&1; then \
		find core/src cli/src -name "*.cpp" 2>/dev/null | head -20 | \
			xargs $(CLANG_TIDY) -p $(BUILD_DIR) 2>/dev/null || \
			echo "Note: Some lint issues found"; \
	else \
		echo "clang-tidy not found. Install LLVM or run: make setup"; \
	fi

lint-python:
	@echo "Running ruff linter..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check python/; \
	elif $(PYTHON) -m ruff --version >/dev/null 2>&1; then \
		$(PYTHON) -m ruff check python/; \
	else \
		echo "ruff not found. Install with: pip install ruff"; \
	fi

# =============================================================================
# Run Targets
# =============================================================================
run: cli
	@echo "Running: $(CLI_BIN) $(ARGS)"
	@if [ -x "$(CLI_BIN)" ]; then \
		$(CLI_BIN) $(ARGS); \
	else \
		echo "CLI binary not found at $(CLI_BIN)"; \
		exit 1; \
	fi

run-grpc: grpc
	@echo "Running gRPC server: $(GRPC_BIN)"
	@if [ -x "$(GRPC_BIN)" ]; then \
		$(GRPC_BIN) $(ARGS); \
	else \
		echo "gRPC binary not found at $(GRPC_BIN)"; \
		exit 1; \
	fi

# =============================================================================
# Setup Targets
# =============================================================================
setup:
ifeq ($(OS),macos)
	@$(MAKE) setup-macos
else ifeq ($(OS),linux)
	@echo "Linux setup: Please install dependencies manually or use your package manager"
	@echo "  Ubuntu/Debian: sudo apt install cmake ninja-build clang-17 python3"
	@echo "  Fedora: sudo dnf install cmake ninja-build clang python3"
	@echo "  Arch: sudo pacman -S cmake ninja clang python"
else
	@$(MAKE) setup-windows
endif

setup-macos:
	@echo "Setting up macOS dependencies..."
	@chmod +x scripts/setup-macos.sh
	@./scripts/setup-macos.sh $(if $(FULL),--full)

setup-windows:
	@echo "Setting up Windows dependencies..."
	@powershell -ExecutionPolicy Bypass -File scripts/setup-windows.ps1 $(if $(FULL),-Full)

# =============================================================================
# Info and Diagnostics
# =============================================================================
info:
	@echo ""
	@echo "PulsimCore Build Configuration"
	@echo "=============================="
	@echo "OS:             $(OS)"
	@echo "Build Dir:      $(BUILD_DIR)"
	@echo "Build Type:     $(BUILD_TYPE)"
	@echo "Parallel Jobs:  $(JOBS)"
	@echo "Install Prefix: $(PREFIX)"
	@echo "Toolchain:      $(or $(TOOLCHAIN),default)"
	@echo ""
	@echo "Tools:"
	@echo "  CMake:        $(CMAKE) ($(shell $(CMAKE) --version 2>/dev/null | head -1 || echo 'not found'))"
	@echo "  Generator:    $(or $(if $(NINJA),Ninja,),Make)"
	@echo "  LLVM Prefix:  $(LLVM_PREFIX)"
	@echo "  Clang Format: $(CLANG_FORMAT)"
	@echo "  Python:       $(PYTHON) ($(shell $(PYTHON) --version 2>&1 || echo 'not found'))"
	@echo ""
	@echo "CMake Options:"
	@echo "  $(CMAKE_OPTIONS)"
	@echo ""

check-deps:
	@echo "Checking dependencies..."
	@echo ""
	@echo "Required:"
	@command -v cmake >/dev/null 2>&1 && echo "  [OK] cmake: $(shell cmake --version | head -1)" || echo "  [MISSING] cmake"
	@command -v ninja >/dev/null 2>&1 && echo "  [OK] ninja: $(shell ninja --version)" || echo "  [OPTIONAL] ninja (will use make)"
	@$(PYTHON) --version >/dev/null 2>&1 && echo "  [OK] python: $(shell $(PYTHON) --version 2>&1)" || echo "  [MISSING] python"
	@echo ""
	@echo "C++ Compiler:"
	@if [ -x "$(LLVM_PREFIX)/bin/clang++" ]; then \
		echo "  [OK] clang++: $(shell $(LLVM_PREFIX)/bin/clang++ --version 2>/dev/null | head -1)"; \
	elif command -v clang++ >/dev/null 2>&1; then \
		echo "  [OK] clang++: $(shell clang++ --version | head -1)"; \
	elif command -v g++ >/dev/null 2>&1; then \
		echo "  [OK] g++: $(shell g++ --version | head -1)"; \
	else \
		echo "  [MISSING] No C++ compiler found"; \
	fi
	@echo ""
	@echo "Optional (code quality):"
	@command -v clang-format >/dev/null 2>&1 && echo "  [OK] clang-format" || echo "  [MISSING] clang-format"
	@command -v clang-tidy >/dev/null 2>&1 && echo "  [OK] clang-tidy" || echo "  [MISSING] clang-tidy"
	@command -v ruff >/dev/null 2>&1 && echo "  [OK] ruff" || echo "  [MISSING] ruff (pip install ruff)"
	@echo ""
