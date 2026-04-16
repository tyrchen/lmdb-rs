# Benchmarks

This directory holds benchmark artefacts for `lmdb-rs-core`:

- **`baselines/`** — criterion baseline samples (one subdir per saved tag).
  Committed to the repo so CI can diff against a known-good point release.
- **`reports/`** — timestamped HTML snapshots of `target/criterion/` after
  a full run. Optional; useful for historical comparison.
- **`results-<date>.md`** — curated summary of a given benchmark run,
  written by a human after interpreting the criterion output.

## Running

Full self-benchmark suite (no C comparison):

```
make bench
```

Head-to-head Rust vs C LMDB:

```
make bench-compare
```

CI smoke (10s total):

```
make bench-quick
```

Save a baseline, then later check for regressions:

```
make bench-baseline NAME=v0.1.0
# ...later...
make bench-regress NAME=v0.1.0
```

`bench-regress` reruns the benches with `--baseline`, then the
`bench-diff` tool (under `crates/bench-diff/`) parses
`<target>/criterion/**/estimates.json` and fails with exit 1 if any
benchmark's `mean.point_estimate` exceeds the baseline by more than 5%.

## CI hook

The spec (`specs/perf-benchmark-design.md` §6.3) calls for a nightly
regression gate. Wire this up with a single GitHub Actions step:

```yaml
- name: Regression check
  run: make bench-regress NAME=v0.1.0
```

The job fails when `bench-diff` exits 1, just like any other test.
