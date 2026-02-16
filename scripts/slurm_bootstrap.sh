#!/bin/bash

# Shared helper for sbatch job scripts.
# Typical use in a job script:
#   source "/project/<account>/<user>/LUMI-AI-Guide/scripts/slurm_bootstrap.sh"
#   bootstrap_repo --require-sqsh

bootstrap_repo() {
  local require_sqsh=0
  local submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
  local search_dir="$submit_dir"
  local found_root=""
  local repo_root=""

  while [[ "$search_dir" != "/" ]]; do
    if [[ -f "$search_dir/env.sh" ]]; then
      found_root="$search_dir"
      break
    fi
    search_dir="$(dirname "$search_dir")"
  done

  if [[ -z "$found_root" ]]; then
    echo "ERROR: Could not locate env.sh above submit dir: $submit_dir" >&2
    return 1
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --require-sqsh)
        require_sqsh=1
        shift
        ;;
      *)
        echo "ERROR: Unknown bootstrap option: $1" >&2
        return 1
        ;;
    esac
  done

  repo_root="$found_root"
  source "$repo_root/env.sh"
  # Keep runtime paths consistent with the location from which the job was submitted.
  export REPO_ROOT="$repo_root"

  : "${CONTAINER:?Set CONTAINER in env.sh}"

  if [[ "$require_sqsh" -eq 1 ]]; then
    export SQSH_PATH="$repo_root/resources/visiontransformer-env.sqsh"
    if [[ ! -f "$SQSH_PATH" ]]; then
      echo "ERROR: Missing sqsh file: $SQSH_PATH" >&2
      return 1
    fi
    export SINGULARITYENV_PREPEND_PATH="/user-software/bin"
  fi

  if [[ "$submit_dir" == "$repo_root"* ]]; then
    cd "$submit_dir"
  else
    cd "$repo_root"
  fi
}
