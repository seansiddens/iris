#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
counters_file="${SCRIPT_DIR}/counters.txt"
output_prefix="${SCRIPT_DIR}/profile"

if [[ ! -f "${counters_file}" ]]; then
  echo "Warning: counters file '${counters_file}' not found." >&2
fi

rocprofv3 -i "${counters_file}" -o "${output_prefix}" -- "$@"
