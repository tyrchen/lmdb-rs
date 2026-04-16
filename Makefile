build:
	@cargo build

test:
	@cargo nextest run --all-features

release:
	@cargo release tag --execute
	@git cliff -o CHANGELOG.md
	@git commit -a -n -m "Update CHANGELOG.md" || true
	@git push origin master
	@cargo release push --execute

update-submodule:
	@git submodule update --init --recursive --remote

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

bench: ## run full self-benchmark suite (no comparison)
	@cargo bench -p lmdb-rs-core \
	    --bench bench_read \
	    --bench bench_write \
	    --bench bench_cursor \
	    --bench bench_dupsort \
	    --bench bench_txn \
	    --bench bench_scaling \
	    --bench bench_admin

bench-compare: ## run head-to-head Rust vs C LMDB comparison
	@cargo bench -p lmdb-rs-core --features bench-compare --bench bench_compare

bench-quick: ## CI smoke — very short run
	@cargo bench -p lmdb-rs-core --bench bench_read -- --quick

bench-baseline: ## snapshot current numbers under NAME=<tag>
	@test -n "$(NAME)" || (echo "usage: make bench-baseline NAME=<tag>" && exit 1)
	@cargo bench -p lmdb-rs-core -- --save-baseline $(NAME)

bench-regress: ## compare against NAME=<tag>, fail on regression
	@test -n "$(NAME)" || (echo "usage: make bench-regress NAME=<tag>" && exit 1)
	@cargo bench -p lmdb-rs-core -- --baseline $(NAME)

.PHONY: build test release update-submodule bench bench-compare bench-quick bench-baseline bench-regress
