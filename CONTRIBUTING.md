# Contributing to Panlabel

Thanks for your interest in contributing! Whether you're fixing a typo, adding
a new format adapter, improving error messages, or writing tests — every
contribution helps make annotation conversion less painful for everyone.

## Getting started

```sh
git clone https://github.com/strickvl/panlabel.git
cd panlabel
cargo build
cargo test
```

If all tests pass, you're ready to go.

## Where to change what

| What you're changing | Where to look |
|---|---|
| CLI behavior, command args, auto-detection | `src/lib.rs` |
| COCO format adapter | `src/ir/io_coco_json.rs` |
| TFOD format adapter | `src/ir/io_tfod_csv.rs` |
| YOLO format adapter | `src/ir/io_yolo.rs` |
| Lossiness and conversion policy | `src/conversion/mod.rs` |
| Stable conversion issue codes | `src/conversion/report.rs` |
| CLI integration tests | `tests/cli.rs` |
| YOLO roundtrip tests | `tests/yolo_roundtrip.rs` |

## What kinds of contributions are most useful?

Here are some areas where help is especially welcome:

- **New format adapters** — Pascal VOC, Label Studio, or other common formats
- **Better error messages** — if a panlabel error confused you, that's a bug worth fixing
- **Test coverage** — especially edge cases and roundtrip tests
- **Documentation improvements** — clearer explanations, better examples, typo fixes

For larger changes (new formats, IR schema changes), please
[open an issue](https://github.com/strickvl/panlabel/issues) first so we can
discuss the approach before you invest time writing code.

## Keeping docs in sync

When your change affects user-visible behavior, please update the relevant docs
in the same PR:

| Behavior change | Update |
|---|---|
| CLI flags or commands | `docs/cli.md` and `README.md` examples |
| Format read/write behavior | `docs/formats.md` |
| Task or use-case support | `docs/tasks.md` |
| Lossiness, report schema, or issue codes | `docs/conversion.md` |

Also update `CLAUDE.md` when repository structure or workflow guidance changes,
and `ROADMAP.md` when priorities shift.

## Guidelines

- **Don't claim unsupported capabilities.** Current scope is detection bounding
  boxes only (not segmentation, pose/keypoints, OBB, or classification-only).
- **Prefer small, concrete examples** that match tested behavior.
- **Write tests** for new behavior — if it's not tested, it can silently break.
