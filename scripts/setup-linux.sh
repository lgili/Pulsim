#!/usr/bin/env bash
# =============================================================================
# PulsimCore - Linux Dependency Installer
# =============================================================================
# Usage:
#   ./scripts/setup-linux.sh          # Install required dependencies
#   ./scripts/setup-linux.sh --full   # Install all dependencies (including optional)
#   ./scripts/setup-linux.sh --help   # Show help
#
# This script installs:
#   Required: Clang 17+, CMake, Ninja, Python 3, SuiteSparse (KLU)
#   Optional (--full): SUNDIALS, gRPC, protobuf
#
# Supports: Ubuntu/Debian, Fedora/RHEL, Arch Linux
# =============================================================================

set -euo pipefail

# Configuration
CLANG_MIN_VERSION=17
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
PulsimCore Linux Dependency Installer

Usage: $0 [OPTIONS]

Options:
    --full      Install optional dependencies (SUNDIALS, gRPC)
    --help, -h  Show this help message

Required dependencies (always installed):
    - Clang 17+ (for C++23 support)
    - CMake 3.20+
    - Ninja build system
    - Python 3.10+
    - SuiteSparse/KLU (sparse matrix solver for circuits)

Optional dependencies (with --full):
    - SUNDIALS (advanced ODE/DAE solvers)
    - gRPC and protobuf (remote API)

Supported distributions:
    - Ubuntu/Debian (apt)
    - Fedora/RHEL (dnf)
    - Arch Linux (pacman)

EOF
}

# =============================================================================
# Distribution Detection
# =============================================================================

detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        echo "$ID"
    elif command -v lsb_release &>/dev/null; then
        lsb_release -is | tr '[:upper:]' '[:lower:]'
    else
        echo "unknown"
    fi
}

get_package_manager() {
    local distro="$1"
    case "$distro" in
        ubuntu|debian|pop|linuxmint)
            echo "apt"
            ;;
        fedora|rhel|centos|rocky|almalinux)
            echo "dnf"
            ;;
        arch|manjaro|endeavouros)
            echo "pacman"
            ;;
        opensuse*)
            echo "zypper"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# =============================================================================
# Installation Functions
# =============================================================================

check_linux() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        print_error "This script is for Linux only."
        exit 1
    fi

    local distro
    distro=$(detect_distro)
    print_success "Running on Linux (distribution: $distro)"
}

install_apt_deps() {
    print_header "Installing Dependencies (apt)"

    sudo apt-get update

    # Clang 17+
    print_info "Installing LLVM/Clang..."
    sudo apt-get install -y wget gnupg software-properties-common

    # Add LLVM repository for newer Clang
    if ! command -v clang-17 &>/dev/null && ! command -v clang-18 &>/dev/null; then
        wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
        local codename
        codename=$(lsb_release -cs)
        echo "deb http://apt.llvm.org/${codename}/ llvm-toolchain-${codename}-17 main" | sudo tee /etc/apt/sources.list.d/llvm.list
        sudo apt-get update
        sudo apt-get install -y clang-17 lldb-17 lld-17 libc++-17-dev libc++abi-17-dev
    fi

    # Build tools
    print_info "Installing build tools..."
    sudo apt-get install -y cmake ninja-build python3 python3-pip python3-venv

    # SuiteSparse (required for KLU)
    print_info "Installing SuiteSparse..."
    sudo apt-get install -y libsuitesparse-dev

    print_success "Base dependencies installed"
}

install_dnf_deps() {
    print_header "Installing Dependencies (dnf)"

    # Clang
    print_info "Installing LLVM/Clang..."
    sudo dnf install -y clang clang-tools-extra llvm llvm-devel libcxx libcxx-devel

    # Build tools
    print_info "Installing build tools..."
    sudo dnf install -y cmake ninja-build python3 python3-pip

    # SuiteSparse (required for KLU)
    print_info "Installing SuiteSparse..."
    sudo dnf install -y suitesparse-devel

    print_success "Base dependencies installed"
}

install_pacman_deps() {
    print_header "Installing Dependencies (pacman)"

    # Clang
    print_info "Installing LLVM/Clang..."
    sudo pacman -Sy --noconfirm clang llvm libc++

    # Build tools
    print_info "Installing build tools..."
    sudo pacman -S --noconfirm cmake ninja python python-pip

    # SuiteSparse (required for KLU)
    print_info "Installing SuiteSparse..."
    sudo pacman -S --noconfirm suitesparse

    print_success "Base dependencies installed"
}

install_zypper_deps() {
    print_header "Installing Dependencies (zypper)"

    # Clang
    print_info "Installing LLVM/Clang..."
    sudo zypper install -y clang llvm libc++-devel

    # Build tools
    print_info "Installing build tools..."
    sudo zypper install -y cmake ninja python3 python3-pip

    # SuiteSparse (required for KLU)
    print_info "Installing SuiteSparse..."
    sudo zypper install -y suitesparse-devel

    print_success "Base dependencies installed"
}

install_base_deps() {
    local pkg_manager="$1"

    case "$pkg_manager" in
        apt)
            install_apt_deps
            ;;
        dnf)
            install_dnf_deps
            ;;
        pacman)
            install_pacman_deps
            ;;
        zypper)
            install_zypper_deps
            ;;
        *)
            print_error "Unsupported package manager: $pkg_manager"
            print_info "Please install manually: clang-17+, cmake, ninja, python3, libsuitesparse-dev"
            exit 1
            ;;
    esac
}

install_optional_apt() {
    print_header "Installing Optional Dependencies (apt)"

    # SUNDIALS
    print_info "Installing SUNDIALS..."
    sudo apt-get install -y libsundials-dev || print_warning "SUNDIALS not available in repos"

    # gRPC
    print_info "Installing gRPC..."
    sudo apt-get install -y libgrpc++-dev protobuf-compiler-grpc || print_warning "gRPC not available in repos"

    print_success "Optional dependencies installed"
}

install_optional_dnf() {
    print_header "Installing Optional Dependencies (dnf)"

    # SUNDIALS
    print_info "Installing SUNDIALS..."
    sudo dnf install -y sundials-devel || print_warning "SUNDIALS not available in repos"

    # gRPC
    print_info "Installing gRPC..."
    sudo dnf install -y grpc-devel grpc-plugins || print_warning "gRPC not available in repos"

    print_success "Optional dependencies installed"
}

install_optional_pacman() {
    print_header "Installing Optional Dependencies (pacman)"

    # SUNDIALS
    print_info "Installing SUNDIALS..."
    sudo pacman -S --noconfirm sundials || print_warning "SUNDIALS not available in repos"

    # gRPC
    print_info "Installing gRPC..."
    sudo pacman -S --noconfirm grpc || print_warning "gRPC not available in repos"

    print_success "Optional dependencies installed"
}

install_optional_deps() {
    local pkg_manager="$1"

    case "$pkg_manager" in
        apt)
            install_optional_apt
            ;;
        dnf)
            install_optional_dnf
            ;;
        pacman)
            install_optional_pacman
            ;;
        *)
            print_warning "Optional dependencies installation not supported for $pkg_manager"
            ;;
    esac
}

show_environment_setup() {
    print_header "Environment Configuration"

    echo -e "${BOLD}For apt-based systems with Clang 17, add to your shell profile:${NC}\n"

    cat << 'EOF'
# PulsimCore - Clang configuration (if using clang-17 from LLVM repos)
export CC=clang-17
export CXX=clang++-17
EOF

    echo ""
    print_info "Then reload your shell: source ~/.bashrc (or ~/.zshrc)"
    echo ""
}

show_build_instructions() {
    print_header "Build Instructions"

    cat << EOF
To build PulsimCore:

    # Configure and build
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build

    # With specific Clang version (if needed)
    CC=clang-17 CXX=clang++-17 cmake -B build -G Ninja
    cmake --build build

To run tests:
    cmake --build build --target pulsim_tests
    ./build/core/pulsim_tests

For Python development:
    pip install -e .
EOF
}

verify_installation() {
    print_header "Verifying Installation"

    # Find clang
    local clang_cmd
    if command -v clang-17 &>/dev/null; then
        clang_cmd="clang-17"
    elif command -v clang-18 &>/dev/null; then
        clang_cmd="clang-18"
    elif command -v clang &>/dev/null; then
        clang_cmd="clang"
    else
        clang_cmd="not found"
    fi

    if [[ "$clang_cmd" != "not found" ]]; then
        local version
        version=$($clang_cmd --version | head -1)
        echo "Clang:       $version"
    else
        echo "Clang:       not found"
    fi

    command -v cmake &>/dev/null && echo "CMake:       $(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')" || echo "CMake:       not found"
    command -v ninja &>/dev/null && echo "Ninja:       $(ninja --version)" || echo "Ninja:       not found"
    command -v python3 &>/dev/null && echo "Python:      $(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')" || echo "Python:      not found"

    # Check SuiteSparse
    if ldconfig -p 2>/dev/null | grep -q libklu || [[ -f /usr/lib/libklu.so ]] || [[ -f /usr/lib/x86_64-linux-gnu/libklu.so ]]; then
        echo "SuiteSparse: installed (KLU enabled)"
    else
        echo "SuiteSparse: not found"
    fi

    if [[ "${FULL_INSTALL:-false}" == "true" ]]; then
        echo ""
        echo "Optional dependencies:"
        ldconfig -p 2>/dev/null | grep -q libsundials && echo "  SUNDIALS:    installed" || echo "  SUNDIALS:    not installed"
        ldconfig -p 2>/dev/null | grep -q libgrpc && echo "  gRPC:        installed" || echo "  gRPC:        not installed"
    fi

    echo ""
    print_success "Installation verification complete!"
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
    echo "  PulsimCore - Linux Dependency Installer"
    echo "=============================================="
    echo -e "${NC}"

    check_linux

    local distro
    distro=$(detect_distro)
    local pkg_manager
    pkg_manager=$(get_package_manager "$distro")

    print_info "Detected package manager: $pkg_manager"

    install_base_deps "$pkg_manager"

    if [[ "$full_install" == "true" ]]; then
        install_optional_deps "$pkg_manager"
    fi

    verify_installation
    show_environment_setup
    show_build_instructions

    echo ""
    print_success "Setup complete!"
}

main "$@"
