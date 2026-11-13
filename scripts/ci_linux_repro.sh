#!/usr/bin/env bash
# Reproduce the Linux `python-tests` / `reference-smoke` CI jobs in Docker
# (ubuntu:24.04, Eigen 3.4.0, GCC 13, pulsim installed as a wheel in
# site-packages) from this checkout.  Mirrors .github/workflows/ci.yml.
#
#   scripts/ci_linux_repro.sh                 # python-tests: pytest -m "not reference"
#   scripts/ci_linux_repro.sh reference       # reference-smoke
#   scripts/ci_linux_repro.sh -- <pytest args>  # any pytest selection
#
# The container (default name pulsim-ci-linux, override with
# PULSIM_CI_CONTAINER) is kept between runs so the apt/pip setup and the
# CMake build directory are reused; remove it with `docker rm -f <name>`.
# The checkout is mounted read-only and copied (without build trees, .git
# or host .so files) to /work inside the container.  Docker on an
# Apple-silicon Mac is aarch64-linux; CI runners are x86-64: set
# PULSIM_CI_PLATFORM=linux/amd64 (and another PULSIM_CI_CONTAINER) to
# reproduce last-bit differences that depend on the CPU (slow, emulated).
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${PULSIM_CI_CONTAINER:-pulsim-ci-linux}"
JOBS="${PULSIM_CI_JOBS:-2}"   # CI: CMAKE_BUILD_PARALLEL_LEVEL=2 (cc1plus needs ~2 GB each)
PLATFORM="${PULSIM_CI_PLATFORM:-}"   # e.g. linux/amd64 (CI's CPU; emulated on Apple silicon)

case "${1:-}" in
  reference) shift; PYTEST=(python/tests/test_benchmark_harness.py -v -m reference) ;;
  --) shift; PYTEST=("$@") ;;
  "") PYTEST=(python/tests -v -m "not reference") ;;
  *) echo "usage: $0 [reference | -- <pytest args>]" >&2; exit 2 ;;
esac

if ! docker inspect "$NAME" >/dev/null 2>&1; then
  docker run -d --name "$NAME" ${PLATFORM:+--platform "$PLATFORM"} -v "$SRC":/src:ro ubuntu:24.04 sleep infinity >/dev/null
fi
docker start "$NAME" >/dev/null

docker exec "$NAME" bash -euc '
if [ ! -x /venv/bin/python ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq cmake ninja-build libeigen3-dev gcc-13 g++-13 \
    python3 python3-venv python3-dev rsync git ca-certificates >/dev/null
  python3 -m venv /venv
  /venv/bin/python -m pip install -q --upgrade pip
  /venv/bin/pip install -q build pytest numpy scipy
fi
rsync -a --delete --exclude=.git --exclude=.scratch --exclude=/build \
  --exclude=/build-tests --exclude="*.so" --exclude=__pycache__ /src/ /work/
cd /work
CC=gcc-13 CXX=g++-13 CMAKE_BUILD_PARALLEL_LEVEL='"$JOBS"' \
  /venv/bin/pip install -q ".[dev]" -C build-dir=/build/{wheel_tag}
'
docker exec -w /work "$NAME" /venv/bin/python -m pytest "${PYTEST[@]}"
