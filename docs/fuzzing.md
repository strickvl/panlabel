# Fuzzing guide

This repo uses `cargo-fuzz` for parser-surface fuzzing in `fuzz/fuzz_targets/` with seed corpora in `fuzz/corpus/<target>/`.

## Install once

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
```

## Discover, build, and check targets

Always run from repo root.

```bash
# List every configured target (source of truth: fuzz/Cargo.toml [[bin]] entries)
cargo +nightly fuzz list

# Build one target without running it
cargo +nightly fuzz build coco_json_parse

# Sanity-check the whole fuzz crate wiring
cargo check --manifest-path fuzz/Cargo.toml
```

## Smoke-run a target (short run)

Use a small `-runs=...` first so local checks stay fast.

```bash
# Small local smoke run
cargo +nightly fuzz run coco_json_parse -- -runs=256

# Another example
cargo +nightly fuzz run tfrecord_parse -- -runs=512
```

## Corpus hygiene (important)

`cargo fuzz run` writes findings and can grow corpora. Avoid accidental noisy commits.

- Prefer targeted staging: `git add fuzz/fuzz_targets/<target>.rs fuzz/corpus/<target>/seed_*`.
- Before committing, inspect fuzz paths explicitly: `git status -- fuzz/`.
- If local fuzzing created extra generated files you do not want to keep, remove them before commit.

## Adding new seeds intentionally

When you *do* want to keep a new seed:

1. Put it in the matching directory: `fuzz/corpus/<target>/`.
2. Keep it tiny and deterministic (minimal reproducer style).
3. Name it clearly (for example `seed-min-valid.json`, `seed-edge-empty.txt`).
4. Stage only intended files, then mention why the seed exists in the commit/PR notes.

## Deferred fuzz cases (known limitations)

Some adapters need real directory layouts or companion files, so they are intentionally deferred from simple parser-surface harnesses:

- OIDv4 full `Label/` directory walking
- WIDER Face image-dimension resolution
- YOLO Keras class/image companion files
- Edge Impulse directory-style input handling
- Marmot directory + same-stem companion-image discovery

Current fuzz targets focus on low-friction parser entry points and fuzz-only wrappers enabled by the `fuzzing` feature in `fuzz/Cargo.toml`.
