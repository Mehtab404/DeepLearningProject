#!/usr/bin/env bash
# Usage: ./scripts/auto_push.sh "feat: completed milestone X"
COMMIT_MSG=${1:-"chore: automated task checkpoint update"}

git add .
if ! git diff --cached --quiet; then
    git commit -m "$COMMIT_MSG"
    git push origin HEAD
    echo "[AUTO-PUSH] Successfully pushed to GitHub: $COMMIT_MSG"
else
    echo "[AUTO-PUSH] No file changes to commit."
fi
