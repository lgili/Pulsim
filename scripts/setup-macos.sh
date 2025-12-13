#!/usr/bin/env bash
# =============================================================================
# PulsimCore - macOS Dependency Installer
# =============================================================================
# Usage:
#   ./scripts/setup-macos.sh          # Install required dependencies
#   ./scripts/setup-macos.sh --full   # Install all dependencies (including optional)
#   ./scripts/setup-macos.sh --help   # Show help
#
# This script installs:
#   Required: Homebrew, LLVM/Clang 17+, CMake, Ninja, Python 3, SuiteSparse (KLU)
#   Optional (--full): SUNDIALS, gRPC, protobuf
# =============================================================================

set -euo pipefail

# Configuration
LLVM_MIN_VERSION=17
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

show_help() {
    cat << EOF
PulsimCore macOS Dependency Installer

Usage: $0 [OPTIONS]

Options:
    --full      Install optional dependencies (SUNDIALS, gRPC)
    --help, -h  Show this help message

Required dependencies (always installed):
    - Homebrew (if not present)
    - LLVM/Clang 17+ (for C++23 support)
    - CMake 3.20+
    - Ninja build system
    - Python 3.10+
    - SuiteSparse/KLU (sparse matrix solver for circuits)

Optional dependencies (with --full):
    - SUNDIALS (advanced ODE/DAE solvers)
    - gRPC and protobuf (remote API)

After installation, add these to your shell profile (~/.zshrc or ~/.bashrc):
    export LLVM_PATH="\$(brew --prefix llvm)"
    export PATH="\${LLVM_PATH}/bin:\$PATH"
    export CC="\${LLVM_PATH}/bin/clang"
    export CXX="\${LLVM_PATH}/bin/clang++"

EOF
}

# =============================================================================
# Dependency Check Functions
# =============================================================================

is_installed() {
    local pkg="$1"
    brew list "$pkg" &>/dev/null
}

get_clang_version() {
    local clang_path="$1"
    "$clang_path" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 | cut -d. -f1
}

# =============================================================================
# Installation Functions
# =============================================================================

check_macos() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        print_error "This script is for macOS only."
        exit 1
    fi
    print_success "Running on macOS $(sw_vers -productVersion)"
}

install_homebrew() {
    print_header "Checking Homebrew"

    if command -v brew &>/dev/null; then
        print_success "Homebrew is installed: $(brew --version | head -1)"
        print_info "Updating Homebrew..."
        brew update
    else
        print_warning "Homebrew not found. Installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add to PATH for Apple Silicon
        if [[ -f "/opt/homebrew/bin/brew" ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi

        print_success "Homebrew installed successfully"
    fi
}

install_llvm() {
    print_header "Installing LLVM/Clang"

    if is_installed llvm; then
        print_info "LLVM already installed, checking version..."
    else
        print_info "Installing LLVM..."
        brew install llvm
    fi

    local llvm_path
    llvm_path="$(brew --prefix llvm)"
    local clang_path="${llvm_path}/bin/clang"

    if [[ ! -x "$clang_path" ]]; then
        print_error "Clang not found at ${clang_path}"
        exit 1
    fi

    local clang_version
    clang_version=$(get_clang_version "$clang_path")

    if [[ -z "$clang_version" ]]; then
        print_error "Could not determine Clang version"
        exit 1
    fi

    if [[ "$clang_version" -lt "$LLVM_MIN_VERSION" ]]; then
        print_error "Clang version ${clang_version} is below minimum required (${LLVM_MIN_VERSION})"
        print_info "Try: brew upgrade llvm"
        exit 1
    fi

    print_success "LLVM/Clang ${clang_version} installed at ${llvm_path}"
}

install_build_tools() {
    print_header "Installing Build Tools"

    # CMake
    if is_installed cmake; then
        print_success "CMake already installed: $(cmake --version | head -1)"
    else
        print_info "Installing CMake..."
        brew install cmake
        print_success "CMake installed: $(cmake --version | head -1)"
    fi

    # Ninja
    if is_installed ninja; then
        print_success "Ninja already installed: $(ninja --version)"
    else
        print_info "Installing Ninja..."
        brew install ninja
        print_success "Ninja installed: $(ninja --version)"
    fi
}

install_python() {
    print_header "Checking Python"

    if command -v python3 &>/dev/null; then
        local py_version
        py_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        local py_major
        py_major=$(echo "$py_version" | cut -d. -f1)
        local py_minor
        py_minor=$(echo "$py_version" | cut -d. -f2)

        if [[ "$py_major" -ge 3 && "$py_minor" -ge 10 ]]; then
            print_success "Python ${py_version} is installed"
        else
            print_warning "Python ${py_version} found, but 3.10+ recommended"
            print_info "Installing Python 3.12..."
            brew install python@3.12
        fi
    else
        print_info "Installing Python 3.12..."
        brew install python@3.12
        print_success "Python installed"
    fi
}

install_suitesparse() {
    print_header "Installing SuiteSparse/KLU"

    # SuiteSparse (required for KLU sparse linear solver)
    if is_installed suite-sparse; then
        print_success "SuiteSparse already installed"
    else
        print_info "Installing SuiteSparse (required for KLU linear solver)..."
        brew install suite-sparse
        print_success "SuiteSparse installed"
    fi
}

install_optional_deps() {
    print_header "Installing Optional Dependencies"

    # SUNDIALS
    if is_installed sundials; then
        print_success "SUNDIALS already installed"
    else
        print_info "Installing SUNDIALS..."
        brew install sundials
        print_success "SUNDIALS installed"
    fi

    # gRPC and protobuf
    if is_installed grpc; then
        print_success "gRPC already installed"
    else
        print_info "Installing gRPC..."
        brew install grpc
        print_success "gRPC installed"
    fi

    if is_installed protobuf; then
        print_success "protobuf already installed"
    else
        print_info "Installing protobuf..."
        brew install protobuf
        print_success "protobuf installed"
    fi
}

show_environment_setup() {
    print_header "Environment Configuration"

    local llvm_path
    llvm_path="$(brew --prefix llvm)"

    echo -e "${BOLD}Add the following to your shell profile (~/.zshrc or ~/.bashrc):${NC}\n"

    cat << EOF
# PulsimCore - LLVM/Clang configuration
export LLVM_PATH="${llvm_path}"
export PATH="\${LLVM_PATH}/bin:\$PATH"
export CC="\${LLVM_PATH}/bin/clang"
export CXX="\${LLVM_PATH}/bin/clang++"
export LDFLAGS="-L\${LLVM_PATH}/lib/c++ -Wl,-rpath,\${LLVM_PATH}/lib/c++"
export CPPFLAGS="-I\${LLVM_PATH}/include"
EOF

    echo ""
    print_info "Then reload your shell: source ~/.zshrc (or ~/.bashrc)"
    echo ""
}

show_build_instructions() {
    print_header "Build Instructions"

    cat << EOF
To build PulsimCore with the Clang toolchain:

    # Using Make (recommended)
    make configure TOOLCHAIN=clang
    make build

    # Or directly with CMake
    cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-clang.cmake
    cmake --build build

To run tests:
    make test          # C++ tests
    make pytest        # Python tests
    make test-all      # All tests

For more options:
    make help
EOF
}

verify_installation() {
    print_header "Verifying Installation"

    local llvm_path
    llvm_path="$(brew --prefix llvm)"
    local clang_version
    clang_version=$(get_clang_version "${llvm_path}/bin/clang")

    echo "Clang:       ${clang_version} (${llvm_path}/bin/clang)"
    echo "CMake:       $(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
    echo "Ninja:       $(ninja --version)"
    echo "Python:      $(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
    is_installed suite-sparse && echo "SuiteSparse: installed (KLU enabled)" || echo "SuiteSparse: not installed"

    if [[ "${FULL_INSTALL:-false}" == "true" ]]; then
        echo ""
        echo "Optional dependencies:"
        is_installed sundials && echo "  SUNDIALS:    installed" || echo "  SUNDIALS:    not installed"
        is_installed grpc && echo "  gRPC:        installed" || echo "  gRPC:        not installed"
        is_installed protobuf && echo "  protobuf:    installed" || echo "  protobuf:    not installed"
    fi

    echo ""
    print_success "All required dependencies are installed!"
}

# =============================================================================
# Main
# =============================================================================

main() {
    local full_install=false

    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --full)
                full_install=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
    done

    export FULL_INSTALL="$full_install"

    echo -e "${BOLD}"
    echo "=============================================="
    echo "  PulsimCore - macOS Dependency Installer"
    echo "=============================================="
    echo -e "${NC}"

    check_macos
    install_homebrew
    install_llvm
    install_build_tools
    install_python
    install_suitesparse

    if [[ "$full_install" == "true" ]]; then
        install_optional_deps
    fi

    verify_installation
    show_environment_setup
    show_build_instructions

    echo ""
    print_success "Setup complete!"
}

main "$@"
