#!/usr/bin/env bash
# Run from your existing private EvoHallu checkout after copying in the files.
# Creates a separate branch and local commit; never pushes or resets branches.
set -euo pipefail

branch_name='vladimir/evohallu-preliminary'
author_name='Vladimir Urvantsev'
author_email='urvantsevve@gmail.com'

git rev-parse --show-toplevel >/dev/null

main_before=$(git rev-parse --verify refs/heads/main 2>/dev/null || true)
hallu3_before=$(git rev-parse --verify refs/heads/hallu3 2>/dev/null || true)
current_branch=$(git symbolic-ref --quiet --short HEAD || true)

if [ "$current_branch" != "$branch_name" ]; then
    if git show-ref --verify --quiet "refs/heads/$branch_name"; then
        echo "Branch $branch_name already exists. No branch was overwritten." >&2
        exit 1
    fi
    git switch -c "$branch_name"
fi

git config --local user.name "$author_name"
git config --local user.email "$author_email"
git config --local user.useConfigOnly true

git add -A
if git diff --cached --quiet; then
    echo 'No uncommitted changes to save. Existing commits were not rewritten.'
else
    GIT_AUTHOR_NAME="$author_name" \
    GIT_AUTHOR_EMAIL="$author_email" \
    GIT_COMMITTER_NAME="$author_name" \
    GIT_COMMITTER_EMAIL="$author_email" \
        git commit -m 'Add EvoHallu preliminary experiments and report'
fi

main_after=$(git rev-parse --verify refs/heads/main 2>/dev/null || true)
hallu3_after=$(git rev-parse --verify refs/heads/hallu3 2>/dev/null || true)
if [ "$main_before" != "$main_after" ] || [ "$hallu3_before" != "$hallu3_after" ]; then
    echo 'Protected branch verification failed. No push was attempted.' >&2
    exit 1
fi

git log -1 --format='Commit: %h%nAuthor: %an <%ae>%nCommitter: %cn <%ce>'
echo 'main and hallu3 branch pointers are unchanged.'
echo 'No push was performed. To publish ONLY the new branch, run:'
echo 'git push -u origin HEAD:refs/heads/vladimir/evohallu-preliminary'
