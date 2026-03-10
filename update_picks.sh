#!/bin/bash
# Quick script to update daily picks and push to GitHub Pages
# Run from project root: ./update_picks.sh
#
# Order matters: update_history runs FIRST so it can read yesterday's
# picks.json (which contains yesterday's games) before predict_daily
# overwrites it with today's predictions.

set -e

echo "=== Updating pick history (resolving yesterday's games) ==="
python3 -m src.cbb.update_history

echo ""
echo "=== Generating today's picks ==="
python3 -m src.cbb.predict_daily --json --save

echo ""
echo "=== Pushing to GitHub Pages ==="
git add docs/data/picks.json docs/data/picks_history.json
git diff --staged --quiet && echo "No changes to push." && exit 0
git commit -m "Update daily picks $(date +%Y-%m-%d)"
git push

echo ""
echo "=== Done! Site will update in ~30 seconds ==="
echo "Visit: https://j-ed35.github.io/cbb_model/picks.html"
