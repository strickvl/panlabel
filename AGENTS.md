# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs` is the library entry point and should hold all business logic.
- `src/main.rs` is a thin CLI wrapper that calls into the library.
- `tests/cli.rs` contains CLI integration tests using `assert_cmd`.
- `scripts/dataset_generator.py` generates synthetic datasets for testing.
- `assets/` is for generated test data and is gitignored.

## Build, Test, and Development Commands
```bash
cargo build              # Build debug version
cargo build --release    # Build optimized release version
cargo run                # Run the CLI
cargo run -- -V          # Run with CLI args
cargo check              # Fast type checking
cargo fmt                # Format code
cargo clippy             # Lint
cargo test               # All tests (unit + integration)
cargo test --test cli    # CLI integration tests only
cargo test runs          # Run a single test by name
cargo doc --open         # Build and view docs

python scripts/dataset_generator.py --num_images 1000 --annotations_per_image 10 --output_dir ./assets
```

## Coding Style & Naming Conventions
- Follow `rustfmt` defaults (4-space indentation, standard Rust formatting).
- Run `cargo fmt` and `cargo clippy` before opening a PR.
- Naming: `snake_case` for functions/modules/tests, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- Keep CLI glue in `src/main.rs`; put core behavior in `src/lib.rs` or its modules.

## Testing Guidelines
- Integration tests live in `tests/` and use `assert_cmd`.
- Add unit tests in `src/` with `#[cfg(test)]` when appropriate.
- Generated datasets belong in `assets/` (gitignored). Use the deterministic seed (42) when modifying the generator to keep data reproducible.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short and imperative (e.g., “Add basic CLI test”, “Update README”). Avoid prefixes unless needed.
- PRs should include: a clear description, test commands run, and any user-facing changes (help text, README) noted. Link an issue for larger changes.

## Configuration & Data Tips
- For the dataset generator, use a fresh Python virtual environment and install `numpy`.
- Keep generated assets out of git; only commit code and fixtures that are meant to be versioned.
