#!/usr/bin/env bash
# Fails if any tracked Rust source file exceeds the hard line limit in CLAUDE.md.
set -euo pipefail

limit=425
status=0

while IFS= read -r -d '' file; do
  lines=$(wc -l < "$file")
  if (( lines > limit )); then
    echo "error: $file has $lines lines (limit $limit)"
    status=1
  fi
done < <(git ls-files -z --cached --others --exclude-standard -- '*.rs')

exit "$status"
