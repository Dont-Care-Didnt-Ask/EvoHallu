# Importing the EvoHallu experiment snapshot

This ZIP contains the current source files, configurations, tests, English
preliminary report, result tables, and compact campaign records. It includes
working-tree changes that were not yet committed in the original checkout.
The ZIP has one top-level directory, `mod1/`.

The `.git` directory is intentionally absent, so importing the snapshot does
not replace the history or remotes of your existing repository. Virtual
environments, model weights, raw feature caches, credentials, and the separately
supplied PopQA/evaluation CSVs are not included. Auxiliary datasets already
tracked in the original repository are retained.

If you already copied these files into an existing checkout on `hallu3`, run
the helper there to move the uncommitted work onto a new branch and commit it:

```bash
bash create_vladimir_branch.sh
git push -u origin HEAD:refs/heads/vladimir/evohallu-preliminary
```

The helper creates `vladimir/evohallu-preliminary`, sets repository-local Git
identity to **Vladimir Urvantsev <urvantsevve@gmail.com>**, and records all
uncommitted changes with that exact author and committer. It verifies that the
local `main` and `hallu3` branch pointers remain unchanged. It never pushes,
resets existing branches, or rewrites earlier commits. If everything was
already committed on `hallu3`, it creates the new branch without manufacturing
another commit; existing history keeps its original authorship.

For a fresh import, start in your existing local clone and copy the **contents**
of the extracted `mod1/` directory into its root. For example:

```bash
git switch -c vladimir/evohallu-preliminary
unzip /path/to/evohallu_repository.zip -d /tmp/evohallu-snapshot
cp -a /tmp/evohallu-snapshot/mod1/. .
bash create_vladimir_branch.sh
git push -u origin HEAD:refs/heads/vladimir/evohallu-preliminary
```

If that branch already exists, choose another branch name. Use a fresh temporary
extraction directory. Resolve any changes in your existing clone before copying
files over it.

Start reading at `evohallu_preliminary_report.md`, or open
`evohallu_preliminary_report.html` in a browser for the rendered document.
The English report links to portable copies of the audit and patches under
`results/technical_review/`. The original patches describe the earlier audited
experiment snapshot, before subsequent reporting and packaging changes.

The archive also preserves compact run records under `out/`. They remain ignored
by git. Source files, the report, result tables, and exported frozen programs
under `results/technical_review/frozen_estimators/` can be committed normally.
Result CSVs are explicitly allowed by `.gitignore`.

No git branch, commit, or remote push is performed by extracting the archive.
