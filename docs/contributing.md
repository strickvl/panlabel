# Contributing

This page covers high-level contribution guidance for docs and behavior changes.

## Where to change what

- CLI behavior, command args, auto-detection: `src/lib.rs`
- Format adapter behavior:
  - COCO: `src/ir/io_coco_json.rs`
  - TFOD: `src/ir/io_tfod_csv.rs`
  - YOLO: `src/ir/io_yolo.rs`
- Lossiness and conversion policy notes: `src/conversion/mod.rs`
- Stable conversion issue codes / JSON schema values: `src/conversion/report.rs`
- User-visible behavior tests:
  - `tests/cli.rs`
  - `tests/yolo_roundtrip.rs`

## Documentation update rules

When behavior changes, update docs in the same change:

1. CLI flag/command behavior changes
   - update `docs/cli.md`
   - update root `README.md` examples if needed
2. Format behavior changes
   - update `docs/formats.md`
3. Task/use-case support changes
   - update `docs/tasks.md`
4. Lossiness/report schema/code changes
   - update `docs/conversion.md`

Also update `AGENTS.md` / `CLAUDE.md` when repository structure or workflow guidance changes.

## Keep docs accurate

- Do not claim unsupported capabilities.
- Current scope is detection bboxes only (not segmentation, pose/keypoints, OBB, or classification-only flows).
- Prefer small, concrete examples that match tested behavior.

## Planned expansion pattern

As docs grow, split by domain:
- `docs/formats/<format>.md`
- `docs/tasks/<task>.md`
- `docs/providers/<provider>.md`

Only create these once implementation exists, to avoid stale placeholder docs.
