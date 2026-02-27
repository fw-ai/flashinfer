---
name: rebase-upstream-release
description: Sync fw-ai/flashinfer fork with the latest upstream flashinfer-ai/flashinfer release tag. Creates a rebase branch, opens a PR to update origin/main, and rebases feature branches onto the new base. Use when the user asks to update main, sync with upstream, rebase onto a new release, or catch up with upstream tags.
---

# Rebase origin/main onto Latest Upstream Release

## Remotes

| Remote | Repository |
|--------|-----------|
| `origin` | `fw-ai/flashinfer` (our fork) |
| `upstream` | `flashinfer-ai/flashinfer` (upstream) |

## Workflow

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Fetch upstream and find latest release tag
- [ ] Step 2: Verify origin/main needs updating
- [ ] Step 3: Create and push rebase branch
- [ ] Step 4: Create PR to update main
- [ ] Step 5: Rebase feature branches (if requested)
```

### Step 1: Fetch and find latest release

```bash
git fetch upstream --tags
git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -5
```

Ignore nightly (`nightly-v*`) and release candidate (`*rc*`) tags.

### Step 2: Verify origin/main is behind

```bash
git merge-base --is-ancestor origin/main <tag>; echo $?
# 0 = origin/main is an ancestor (needs updating)
# 1 = NOT an ancestor (already current or diverged)
```

### Step 3: Create the rebase branch

**Naming convention**: `rebase-0pXpY` — version dots replaced with `p`.

| Release | Branch |
|---------|--------|
| v0.6.2 | `rebase-0p6p2` |
| v0.6.4 | `rebase-0p6p4` |
| v0.7.0 | `rebase-0p7p0` |

```bash
git checkout -b rebase-0pXpY <tag>
git push -u origin rebase-0pXpY
```

### Step 4: Create the PR

**Important**: Use `--repo fw-ai/flashinfer` explicitly — without it, `gh pr create` may fail with "No commits between main and ..." even when there are commits.

```bash
gh pr create --repo fw-ai/flashinfer --base main --head rebase-0pXpY \
  --title "Rebase main onto upstream <tag>" \
  --body "$(cat <<'EOF'
## Summary
- Rebases main onto upstream FlashInfer <tag> release
- Brings in all upstream changes since last sync

EOF
)"
```

### Step 5: Rebase feature branches

**Determine the old base**: Feature branches are built on a previous `rebase-*` branch. Find it:

```bash
git branch -a | grep rebase
```

Then try each candidate to find which one the feature branch was built on:

```bash
git log --oneline origin/<feature-branch> --not origin/<old-rebase-branch>
```

The correct base will show only the feature's custom commits (typically a handful), not hundreds of upstream commits.

**Choose a rebase strategy:**

| Strategy | When to use |
|----------|-------------|
| Net diff (recommended) | Messy history, reverts, or intermediate conflicts |
| Commit-by-commit | Clean history you want to preserve |

#### Net diff (recommended)

Applies the combined effect of all feature commits as a single clean patch:

```bash
git checkout -b <feature>-rebase-0pXpY rebase-0pXpY
git diff origin/<old-base>..origin/<feature> | git apply --3way
git add -A
git commit  # summarize the squashed changes
git push -u origin <feature>-rebase-0pXpY
```

#### Commit-by-commit

```bash
git checkout -b <feature>-rebase-0pXpY origin/<feature>
git rebase --onto rebase-0pXpY origin/<old-base>
```

On conflict: if upstream already incorporated equivalent changes, accept the new base version with `git checkout --ours <file>`. Use `GIT_EDITOR="true" git rebase --continue` since the terminal may not have an editor configured.

#### Verify the result

The file list and change counts should match between the original and rebased branches:

```bash
git diff origin/<old-base>..origin/<feature> --stat
git diff rebase-0pXpY..<feature>-rebase-0pXpY --stat
```

## Gotchas

- **`gh pr create` "No commits" error**: Always pass `--repo fw-ai/flashinfer` explicitly.
- **`GIT_EDITOR` unset during rebase continue**: Use `GIT_EDITOR="true" git rebase --continue`.
- **Untracked files block checkout**: `git stash --include-untracked` before switching, `git stash pop` after.
- **Feature branch has upstream commits mixed in**: This means you picked the wrong old base. The correct base yields only the handful of custom commits when diffed.
