# Baselines

Subdirectories here hold copied criterion baseline samples so the
`bench-diff` tool can detect regressions across commits.

## Populating a new baseline

After a release tag:

```
make bench-baseline NAME=v0.1.0
```

That writes raw samples to `<target>/criterion/**/v0.1.0/`. Mirror them
into this directory so they're versioned with the repo:

```
# From the repo root:
rsync -a \
  --include='*/' \
  --include='v0.1.0/**' \
  --exclude='*' \
  "$(cargo metadata --format-version=1 | jq -r .target_directory)/criterion/" \
  docs/bench/baselines/v0.1.0/
```

Then `git add docs/bench/baselines/v0.1.0 && git commit`.

## Regression check

```
make bench-regress NAME=v0.1.0
```

Runs the full self-benchmark suite with `--baseline v0.1.0`, then calls
the `bench-diff` binary to compute `mean.point_estimate` deltas. Any
benchmark that regressed by more than 5% triggers exit code 1.
