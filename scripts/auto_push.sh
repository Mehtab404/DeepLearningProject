#!/usr/bin/env bash
# Usage: ./scripts/auto_push.sh "feat: completed milestone X"
# WHAT: Commit message from the first argument, or a default.
# HOW: Bash default syntax ${1:-text}.
# WHY: A message must exist.
# PROS & CONS: + Safe default. - None.
# EXAMPLE: ./scripts/auto_push.sh 'feat: x' sets COMMIT_MSG to 'feat: x'.
COMMIT_MSG=${1:-"chore: automated task checkpoint update"}

# WHAT: Stages all changes.
# HOW: git add .
# WHY: Simple for a research repository.
# PROS & CONS: + Convenient. - Adds everything not in .gitignore.
# EXAMPLE: A new PNG plot is staged.
git add .
# WHAT: Checks whether anything is staged.
# HOW: --quiet exits 1 when there are differences; ! inverts it.
# WHY: Avoids empty commits.
# PROS & CONS: + Clean history. - None.
# EXAMPLE: False when nothing changed.
if ! git diff --cached --quiet; then
    # WHAT: Creates the commit.
    # HOW: git commit -m with the message.
    # WHY: Snapshot.
    # PROS & CONS: + Traceable. - None.
    # EXAMPLE: 'feat: x'.
    git commit -m "$COMMIT_MSG"
    # WHAT: Uploads the commit.
    # HOW: Pushes the current branch.
    # WHY: Remote backup.
    # PROS & CONS: + Works on any branch. - Needs credentials.
    # EXAMPLE: pushes improve-robustness-eval when on that branch.
    git push origin HEAD
    # WHAT: Success message.
    # HOW: echo.
    # WHY: Feedback.
    # PROS & CONS: + Visible. - None.
    # EXAMPLE: [AUTO-PUSH] Successfully pushed to GitHub: feat: x
    echo "[AUTO-PUSH] Successfully pushed to GitHub: $COMMIT_MSG"
# WHAT: No changes branch.
# HOW: Prints a message.
# WHY: Tells the user nothing happened.
# PROS & CONS: + Clear. - None.
# EXAMPLE: Prints '[AUTO-PUSH] No file changes to commit.' and exits without a commit.
else
    echo "[AUTO-PUSH] No file changes to commit."
fi
