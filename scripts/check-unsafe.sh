#!/usr/bin/env bash
# Only the `cuda` crate may use unsafe; every other crate root must forbid it.
set -euo pipefail

status=0

while IFS= read -r -d '' file; do
  if grep -nE '\bunsafe\b' "$file" | grep -v 'forbid(unsafe_code)' > /dev/null; then
    echo "error: $file uses unsafe outside crates/cuda"
    status=1
  fi
done < <(git ls-files -z --cached --others --exclude-standard -- 'crates/*.rs' ':!:crates/cuda/**')

for root in crates/*/src/lib.rs crates/*/src/main.rs; do
  [[ -e "$root" ]] || continue
  [[ "$root" == crates/cuda/* ]] && continue
  if ! grep -q '^#!\[forbid(unsafe_code)\]' "$root"; then
    echo "error: $root is missing #![forbid(unsafe_code)]"
    status=1
  fi
done

exit "$status"
