#!/usr/bin/env bash
# CI verification script for chomp3rs.
# Single source of truth for all CI checks. Called by GitHub Actions
# with individual subcommands, and usable locally to run everything.
#
# Usage:
#   ./scripts/ci.sh              # run all checks sequentially
#   ./scripts/ci.sh --quick      # skip slow checks (Miri, deny, spellcheck)
#   ./scripts/ci.sh <command>    # run a specific check group
#
# Commands: fmt, clippy, test, doc, miri, deny, spellcheck

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
RESET='\033[0m'

step=0
run_step() {
    step=$((step + 1))
    echo -e "\n${BOLD}[$step] $1${RESET}"
    shift
    if "$@"; then
        echo -e "${GREEN}  passed${RESET}"
    else
        echo -e "${RED}  FAILED${RESET}"
        exit 1
    fi
}

cmd_fmt() {
    run_step "Format check" \
        cargo +nightly fmt --all --check
}

cmd_clippy() {
    run_step "Clippy (default features)" \
        cargo clippy --workspace --all-targets
    run_step "Clippy (MPI)" \
        cargo clippy --workspace --all-targets --features mpi
}

cmd_test() {
    run_step "Tests (default features)" \
        cargo test --workspace
    run_step "Tests (MPI)" \
        cargo test --workspace --features mpi
}

cmd_doc() {
    RUSTDOCFLAGS="-Dwarnings" run_step "Documentation" \
        cargo doc --no-deps --document-private-items
}

cmd_miri() {
    run_step "Miri (graders)" \
        cargo +nightly miri test complexes::cubical::graders
    run_step "Miri (peaks)" \
        cargo +nightly miri test homology::cubical::subgrid::peaks
}

cmd_deny() {
    run_step "Cargo deny" \
        cargo deny --all-features check
}

cmd_spellcheck() {
    run_step "Spellcheck" \
        cargo spellcheck check -m 1
}

# Parse arguments
quick=false
command=""
for arg in "$@"; do
    case "$arg" in
        --quick) quick=true ;;
        fmt|clippy|test|doc|miri|deny|spellcheck) command="$arg" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [ -n "$command" ]; then
    "cmd_$command"
else
    cmd_fmt
    cmd_clippy
    cmd_test
    cmd_doc
    if [ "$quick" = false ]; then
        cmd_miri
        cmd_deny
        cmd_spellcheck
    fi
fi

echo -e "\n${GREEN}${BOLD}All checks passed.${RESET}"
