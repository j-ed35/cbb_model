#!/usr/bin/env bash
set -e

# Rename: "Pomeroy College Basketball Ratings Archive (YYYY-MM-DD).html"
# -> "kenpom_archive_YYYY-MM-DD.html"
for f in Pomeroy\ College\ Basketball\ Ratings\ Archive\ \(*.html; do
  [ -f "$f" ] || continue
  date="$(echo "$f" | sed -E 's/.*\(([0-9]{4}-[0-9]{2}-[0-9]{2})\)\.html/\1/')"
  new="kenpom_archive_${date}.html"
  if [ "$f" != "$new" ]; then
    mv -v "$f" "$new"
  fi
done

# Remove any "*_files" directories
for d in *_files; do
  [ -d "$d" ] || continue
  rm -rf "$d"
  echo "Removed folder: $d"
done

echo "Done."
