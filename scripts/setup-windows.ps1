# =============================================================================
# PulsimCore - Windows Dependency Installer
# =============================================================================
# Usage:
#   .\scripts\setup-windows.ps1          # Install required dependencies
#   .\scripts\setup-windows.ps1 -Full    # Install all dependencies (including optional)
#   .\scripts\setup-windows.ps1 -Help    # Show help
#
# This script installs:
#   Required: LLVM/Clang 17+, CMake, Ninja, Python 3
#   Optional (-Full): vcpkg, SuiteSparse, SUNDIALS, gRPC
#
# Run as Administrator for best results.
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Full,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$LLVM_MIN_VERSION = 17
$LLVM_DEFAULT_PATH = "C:\Program Files\LLVM"
$VCPKG_PATH = "C:\vcpkg"

# =============================================================================
# Helper Functions
# =============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Blue
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Cyan -NoNewline
    Write-Host $Message
}

function Show-Help {
    @"
PulsimCore Windows Dependency Installer

Usage: .\setup-windows.ps1 [OPTIONS]

Options:
    -Full       Install optional dependencies (vcpkg, SuiteSparse, SUNDIALS, gRPC)
    -Help       Show this help message

Required dependencies (always installed):
    - LLVM/Clang 17+ (for C++23 support)
    - CMake 3.20+
    - Ninja build system
    - Python 3.10+

Optional dependencies (with -Full):
    - vcpkg (C++ package manager)
    - SuiteSparse (sparse matrix operations)
    - SUNDIALS (advanced ODE/DAE solvers)
    - gRPC and protobuf (remote API)

After installation, restart your terminal or run:
    refreshenv

"@
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-PackageManager {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        return "winget"
    }
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        return "chocolatey"
    }
    return $null
}

function Get-ClangVersion {
    param([string]$ClangPath)

    if (-not (Test-Path $ClangPath)) {
        return $null
    }

    try {
        $versionOutput = & $ClangPath --version 2>&1
        if ($versionOutput -match "(\d+)\.\d+\.\d+") {
            return [int]$Matches[1]
        }
    }
    catch {
        return $null
    }
    return $null
}

# =============================================================================
# Installation Functions
# =============================================================================

function Install-PackageManager {
    Write-Header "Checking Package Manager"

    $pkgMgr = Get-PackageManager

    if ($pkgMgr -eq "winget") {
        Write-Success "winget is available"
        return "winget"
    }
    elseif ($pkgMgr -eq "chocolatey") {
        Write-Success "Chocolatey is available"
        return "chocolatey"
    }
    else {
        Write-Warning "No package manager found"
        Write-Info "Attempting to install Chocolatey..."

        if (-not (Test-Administrator)) {
            Write-Error "Administrator privileges required to install Chocolatey"
            Write-Info "Please run this script as Administrator or install winget manually"
            Write-Info "winget is included with Windows 11 and Windows 10 (via App Installer from Microsoft Store)"
            exit 1
        }

        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

        # Refresh environment
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

        Write-Success "Chocolatey installed"
        return "chocolatey"
    }
}

function Install-LLVM {
    param([string]$PackageManager)

    Write-Header "Installing LLVM/Clang"

    $clangPath = "$LLVM_DEFAULT_PATH\bin\clang.exe"

    # Check if already installed
    if (Test-Path $clangPath) {
        $version = Get-ClangVersion $clangPath
        if ($version -ge $LLVM_MIN_VERSION) {
            Write-Success "LLVM/Clang $version already installed"
            return
        }
        Write-Warning "LLVM/Clang $version found, but version $LLVM_MIN_VERSION+ required"
    }

    Write-Info "Installing LLVM..."

    if ($PackageManager -eq "winget") {
        winget install LLVM.LLVM --accept-package-agreements --accept-source-agreements
    }
    else {
        choco install llvm -y
    }

    # Verify installation
    if (-not (Test-Path $clangPath)) {
        Write-Error "LLVM installation failed - clang.exe not found at $clangPath"
        exit 1
    }

    $version = Get-ClangVersion $clangPath
    if ($version -lt $LLVM_MIN_VERSION) {
        Write-Error "Installed Clang version $version is below minimum required ($LLVM_MIN_VERSION)"
        exit 1
    }

    Write-Success "LLVM/Clang $version installed at $LLVM_DEFAULT_PATH"
}

function Install-BuildTools {
    param([string]$PackageManager)

    Write-Header "Installing Build Tools"

    # CMake
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        $cmakeVersion = (cmake --version | Select-Object -First 1) -replace "cmake version ", ""
        Write-Success "CMake $cmakeVersion already installed"
    }
    else {
        Write-Info "Installing CMake..."
        if ($PackageManager -eq "winget") {
            winget install Kitware.CMake --accept-package-agreements --accept-source-agreements
        }
        else {
            choco install cmake -y
        }

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        Write-Success "CMake installed"
    }

    # Ninja
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $ninjaVersion = ninja --version
        Write-Success "Ninja $ninjaVersion already installed"
    }
    else {
        Write-Info "Installing Ninja..."
        if ($PackageManager -eq "winget") {
            winget install Ninja-build.Ninja --accept-package-agreements --accept-source-agreements
        }
        else {
            choco install ninja -y
        }
        Write-Success "Ninja installed"
    }
}

function Install-Python {
    param([string]$PackageManager)

    Write-Header "Checking Python"

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pyVersion = (python --version 2>&1) -replace "Python ", ""
        $pyMajor = [int]($pyVersion.Split('.')[0])
        $pyMinor = [int]($pyVersion.Split('.')[1])

        if ($pyMajor -ge 3 -and $pyMinor -ge 10) {
            Write-Success "Python $pyVersion is installed"
            return
        }
        Write-Warning "Python $pyVersion found, but 3.10+ recommended"
    }

    Write-Info "Installing Python 3.12..."
    if ($PackageManager -eq "winget") {
        winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    }
    else {
        choco install python312 -y
    }
    Write-Success "Python installed"
}

function Set-EnvironmentVariables {
    Write-Header "Configuring Environment Variables"

    $llvmBinPath = "$LLVM_DEFAULT_PATH\bin"

    # Check if LLVM is in PATH
    $currentPath = [System.Environment]::GetEnvironmentVariable("Path", "User")

    if ($currentPath -notlike "*$llvmBinPath*") {
        Write-Info "Adding LLVM to user PATH..."
        [System.Environment]::SetEnvironmentVariable("Path", "$llvmBinPath;$currentPath", "User")
        Write-Success "LLVM added to PATH"
    }
    else {
        Write-Success "LLVM already in PATH"
    }

    # Set CC and CXX
    [System.Environment]::SetEnvironmentVariable("CC", "$llvmBinPath\clang.exe", "User")
    [System.Environment]::SetEnvironmentVariable("CXX", "$llvmBinPath\clang++.exe", "User")
    Write-Success "CC and CXX environment variables set"

    # Update current session
    $env:Path = "$llvmBinPath;$env:Path"
    $env:CC = "$llvmBinPath\clang.exe"
    $env:CXX = "$llvmBinPath\clang++.exe"
}

function Install-Vcpkg {
    Write-Header "Installing vcpkg"

    if (Test-Path "$VCPKG_PATH\vcpkg.exe") {
        Write-Success "vcpkg already installed at $VCPKG_PATH"
        return
    }

    Write-Info "Cloning vcpkg..."
    git clone https://github.com/Microsoft/vcpkg.git $VCPKG_PATH

    Write-Info "Bootstrapping vcpkg..."
    & "$VCPKG_PATH\bootstrap-vcpkg.bat" -disableMetrics

    # Add to PATH
    $currentPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$VCPKG_PATH*") {
        [System.Environment]::SetEnvironmentVariable("Path", "$VCPKG_PATH;$currentPath", "User")
    }

    Write-Success "vcpkg installed at $VCPKG_PATH"
}

function Install-OptionalDeps {
    Write-Header "Installing Optional Dependencies"

    if (-not (Test-Path "$VCPKG_PATH\vcpkg.exe")) {
        Install-Vcpkg
    }

    $vcpkg = "$VCPKG_PATH\vcpkg.exe"

    # SuiteSparse
    Write-Info "Installing SuiteSparse (this may take a while)..."
    & $vcpkg install suitesparse:x64-windows
    Write-Success "SuiteSparse installed"

    # SUNDIALS
    Write-Info "Installing SUNDIALS..."
    & $vcpkg install sundials:x64-windows
    Write-Success "SUNDIALS installed"

    # gRPC
    Write-Info "Installing gRPC (this may take a while)..."
    & $vcpkg install grpc:x64-windows
    Write-Success "gRPC installed"

    # Integrate vcpkg with CMake
    Write-Info "Integrating vcpkg with CMake..."
    & $vcpkg integrate install
    Write-Success "vcpkg integrated"
}

function Show-EnvironmentSetup {
    Write-Header "Environment Configuration"

    Write-Host "The following environment variables have been set:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  CC  = $LLVM_DEFAULT_PATH\bin\clang.exe"
    Write-Host "  CXX = $LLVM_DEFAULT_PATH\bin\clang++.exe"
    Write-Host "  PATH includes $LLVM_DEFAULT_PATH\bin"
    Write-Host ""
    Write-Host "To apply changes to your current terminal, run:" -ForegroundColor Yellow
    Write-Host "  refreshenv  (if using Chocolatey)"
    Write-Host "  or restart your terminal"
    Write-Host ""
}

function Show-BuildInstructions {
    Write-Header "Build Instructions"

    @"
To build PulsimCore with the Clang toolchain:

    # Using Make (if available, e.g., via Git Bash or MSYS2)
    make configure TOOLCHAIN=clang
    make build

    # Or directly with CMake
    cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-clang.cmake
    cmake --build build

    # With vcpkg integration (if -Full was used)
    cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-clang.cmake -DCMAKE_PREFIX_PATH=C:\vcpkg\installed\x64-windows
    cmake --build build

To run tests:
    cmake --build build --target pulsim_tests
    .\build\core\pulsim_tests.exe

For Python development:
    pip install -e .
"@
}

function Test-Installation {
    Write-Header "Verifying Installation"

    $clangPath = "$LLVM_DEFAULT_PATH\bin\clang.exe"
    $clangVersion = Get-ClangVersion $clangPath

    Write-Host "Clang:    $clangVersion ($clangPath)"

    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        $cmakeVersion = (cmake --version | Select-Object -First 1) -replace "cmake version ", ""
        Write-Host "CMake:    $cmakeVersion"
    }

    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        Write-Host "Ninja:    $(ninja --version)"
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pyVersion = (python --version 2>&1) -replace "Python ", ""
        Write-Host "Python:   $pyVersion"
    }

    if ($Full) {
        Write-Host ""
        Write-Host "Optional dependencies:"
        if (Test-Path "$VCPKG_PATH\vcpkg.exe") {
            Write-Host "  vcpkg:       installed at $VCPKG_PATH"
        }
        if (Test-Path "$VCPKG_PATH\installed\x64-windows\lib\suitesparse*") {
            Write-Host "  SuiteSparse: installed"
        }
        if (Test-Path "$VCPKG_PATH\installed\x64-windows\lib\sundials*") {
            Write-Host "  SUNDIALS:    installed"
        }
        if (Test-Path "$VCPKG_PATH\installed\x64-windows\lib\grpc*") {
            Write-Host "  gRPC:        installed"
        }
    }

    Write-Host ""
    Write-Success "All required dependencies are installed!"
}

# =============================================================================
# Main
# =============================================================================

function Main {
    if ($Help) {
        Show-Help
        return
    }

    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Blue
    Write-Host "  PulsimCore - Windows Dependency Installer" -ForegroundColor Blue
    Write-Host "==============================================" -ForegroundColor Blue
    Write-Host ""

    if (-not (Test-Administrator)) {
        Write-Warning "Running without Administrator privileges"
        Write-Info "Some operations may require Administrator access"
        Write-Host ""
    }

    $pkgMgr = Install-PackageManager
    Install-LLVM -PackageManager $pkgMgr
    Install-BuildTools -PackageManager $pkgMgr
    Install-Python -PackageManager $pkgMgr
    Set-EnvironmentVariables

    if ($Full) {
        Install-OptionalDeps
    }

    Test-Installation
    Show-EnvironmentSetup
    Show-BuildInstructions

    Write-Host ""
    Write-Success "Setup complete!"
}

Main
