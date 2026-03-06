#!/bin/bash
# Quick script to update daily picks and push to GitHub Pages
# Run from project root: ./update_picks.sh

set -e

echo "=== Generating today's picks ==="
python -m src.cbb.predict_daily --json --save

echo ""
echo "=== Pushing to GitHub Pages ==="
git add docs/data/picks.json
git diff --staged --quiet && echo "No changes to push." && exit 0
git commit -m "Update daily picks $(date +%Y-%m-%d)"
git push

echo ""
echo "=== Done! Site will update in ~30 seconds ==="
echo "Visit: https://j-ed35.github.io/cbb_model/picks.html"
