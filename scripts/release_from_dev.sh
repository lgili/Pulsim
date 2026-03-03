#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/release_from_dev.sh <version> [options]

Description:
  Creates a release snapshot from dev into main:
    1) checkout main
    2) restore full tree from dev
    3) commit "release: v<version>" (allow-empty)
    4) create tag "v<version>"
    5) optionally push main and tag

Options:
  --dev-branch <name>    Source branch (default: dev)
  --main-branch <name>   Target branch (default: main)
  --remote <name>        Remote name for push/sync (default: origin)
  --sync                 Fast-forward sync dev/main from remote before release
  --push                 Push main and tag after local creation
  --dry-run              Print commands without changing repository
  -h, --help             Show this help

Example:
  scripts/release_from_dev.sh 0.6.2 --sync --push
EOF
}

if [[ $# -eq 1 ]] && [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

VERSION_RAW="$1"
shift

VERSION="${VERSION_RAW#v}"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z]+)*$ ]]; then
  echo "Error: invalid version '$VERSION_RAW'. Expected X.Y.Z (optionally with suffix)." >&2
  exit 1
fi

TAG="v${VERSION}"
DEV_BRANCH="dev"
MAIN_BRANCH="main"
REMOTE="origin"
DO_SYNC=false
DO_PUSH=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev-branch)
      DEV_BRANCH="${2:-}"
      shift 2
      ;;
    --main-branch)
      MAIN_BRANCH="${2:-}"
      shift 2
      ;;
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    --sync)
      DO_SYNC=true
      shift
      ;;
    --push)
      DO_PUSH=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" == false ]]; then
    "$@"
  fi
}

ensure_clean_worktree() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: working tree has tracked changes. Commit/stash first." >&2
    exit 1
  fi
  if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "Error: working tree has untracked files. Commit/stash/clean first." >&2
    exit 1
  fi
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository." >&2
  exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/${DEV_BRANCH}"; then
  echo "Error: local branch '${DEV_BRANCH}' does not exist." >&2
  exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/${MAIN_BRANCH}"; then
  echo "Error: local branch '${MAIN_BRANCH}' does not exist." >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "Error: tag '${TAG}' already exists locally." >&2
  exit 1
fi

if [[ "$DO_PUSH" == true ]] && [[ "$DRY_RUN" == false ]]; then
  if git ls-remote --exit-code --tags "$REMOTE" "refs/tags/${TAG}" >/dev/null 2>&1; then
    echo "Error: tag '${TAG}' already exists on remote '${REMOTE}'." >&2
    exit 1
  fi
fi

ensure_clean_worktree

START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$START_BRANCH" == "HEAD" ]]; then
  echo "Error: detached HEAD. Checkout a branch first." >&2
  exit 1
fi

if [[ "$DO_SYNC" == true ]]; then
  run git fetch "$REMOTE" "$DEV_BRANCH" "$MAIN_BRANCH" --tags
  run git checkout "$DEV_BRANCH"
  run git pull --ff-only "$REMOTE" "$DEV_BRANCH"
  run git checkout "$MAIN_BRANCH"
  run git pull --ff-only "$REMOTE" "$MAIN_BRANCH"
else
  run git checkout "$MAIN_BRANCH"
fi

run git restore --source "$DEV_BRANCH" --staged --worktree :/
run git commit --allow-empty -m "release: ${TAG}"
run git tag "${TAG}"

if [[ "$DO_PUSH" == true ]]; then
  run git push "$REMOTE" "$MAIN_BRANCH"
  run git push "$REMOTE" "${TAG}"
fi

if [[ "$START_BRANCH" != "$MAIN_BRANCH" ]]; then
  run git checkout "$START_BRANCH"
fi

echo "Release snapshot created: ${TAG}"
