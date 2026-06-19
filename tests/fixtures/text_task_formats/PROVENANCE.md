# Text/task format fixture provenance

Checked on: 2026-06-19

These fixtures are intentionally tiny and deterministic. They are **schema/layout fixtures**, not full upstream dataset copies. Row values are synthetic unless a file says otherwise. The point is to pin the shape an adapter should recognize without pulling benchmark data into the repository.

## RLVR-HF / generic Hugging Face RLVR rows

Primary sources checked:

- AllenAI RLVR-IFeval dataset page/API: https://huggingface.co/datasets/allenai/RLVR-IFeval
  - Hugging Face dataset API SHA: `47c03c73621c4aab2b824b7818681117d662770e`
  - Dataset-server split checked: `default/train`
  - Dataset-server features checked: `messages`, `ground_truth`, `dataset`, `constraint_type`, `constraint`
- NVIDIA Nemotron RL Agentic Conversational Tool Use Pivot dataset page/API: https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1
  - Hugging Face dataset API SHA: `9643c8103d7bfbc2d7fc4d15991d6739c612ff58`
  - Dataset-server split checked: `default/train`
  - Dataset-server features checked: `trajectory_id`, `responses_create_params`, `expected_action`, `scenario`, `num_unique_actions`, `meta_info`, `qwen_235b_info`, `agent_ref`, `pass_rate`, `pass_rate_total`, `pass_rate_passed`

Conclusion: `rlvr-hf` must stay generic. Current RLVR-style HF datasets are ordinary HF tables/JSONL/parquet with task-specific columns, not one canonical RLVR schema.

Fixtures:

- `rlvr_hf/allenai_rlvr_ifeval_shape.jsonl`
- `rlvr_hf/nvidia_agentic_tool_use_shape.jsonl`

## Verifiers tasksets

Primary sources checked:

- Verifiers environments docs: https://docs.primeintellect.ai/verifiers/environments
- Verifiers overview docs: https://docs.primeintellect.ai/verifiers/overview
- Verifiers GitHub repo: https://github.com/PrimeIntellect-ai/verifiers
  - `main` commit checked: `5ed280996259056aa5a5191513b0c62387b47686`

Conclusion: the adapter should support materialized rows with `prompt` message lists or `question` strings, optional `answer`, optional `info` metadata, `system_prompt` from environment config, and taskset-ish fields such as `max_turns` when present. Python rubrics/environments are executable artifacts and should be referenced/copied, not imported or translated.

Fixtures:

- `verifiers_taskset/dataset.jsonl`
- `verifiers_taskset/rubric.py`
- `verifiers_taskset/env.py`

## Harbor

Primary sources checked:

- Harbor task docs: https://www.harborframework.com/docs/tasks
- Harbor multi-step task docs: https://www.harborframework.com/docs/tasks/multi-step
- Harbor Terminal-Bench difference docs: https://www.harborframework.com/docs/tasks/task-difference
- Harbor GitHub repo: https://github.com/harbor-framework/harbor
  - `main` commit checked: `4c2c2d1413401952fc79ee959713f2e11403d9ad`

Conclusion: current Harbor single-task directories use `instruction.md`, `task.toml`, `environment/`, `solution/`, and `tests/`. Current docs show `schema_version = "1.3"`. Harbor multi-step tasks move per-step instruction/tests/solution under `steps/<name>/` and declare ordered `[[steps]]` entries in `task.toml`.

Fixtures:

- `harbor/single_task_minimal/`
- `harbor/multi_step_minimal/`

## SWE-bench

Primary sources checked:

- SWE-bench dataset docs: https://www.swebench.com/SWE-bench/guides/datasets/
- SWE-bench Hugging Face dataset page/API: https://huggingface.co/datasets/princeton-nlp/SWE-bench
  - Hugging Face dataset API SHA: `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`
  - Dataset-server splits checked: `default/dev`, `default/test`, `default/train`
  - Dataset-server features checked: `repo`, `instance_id`, `base_commit`, `patch`, `test_patch`, `problem_statement`, `hints_text`, `created_at`, `version`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `environment_setup_commit`

Conclusion: SWE-bench import should treat issue text as task input and preserve `patch`/`test_patch` as patch artifacts. `test_patch` and `patch` should not be concatenated into model-facing task input.

Fixture:

- `swe_bench/minimal_instance.jsonl`

## Terminal-Bench compatibility

Primary sources checked:

- Terminal-Bench task overview: https://www.tbench.ai/docs/task-overview
- Terminal-Bench 2.0 docs: https://www.tbench.ai/docs
- Harbor docs for running Terminal-Bench: https://github.com/harbor-framework/harbor/blob/main/docs/content/docs/tutorials/running-terminal-bench.mdx
- Terminal-Bench GitHub repo: https://github.com/harbor-framework/terminal-bench
  - `main` commit checked: `1a6ffa9674b571da0ed040c470cb40c4d85f9b9b`
  - Code source checked: `terminal_bench/handlers/trial_handler.py`

Conclusion: current Terminal-Bench 2.x is run via Harbor dataset IDs such as `terminal-bench/terminal-bench-2`, so Harbor compatibility should be verified through Harbor-format fixtures or Harbor Hub data. The legacy/open Terminal-Bench repo still contains the older layout: `task.yaml`, `solution.sh` or `solution.yaml`, `run-tests.sh`, `docker-compose.yaml`, and `tests/`. That older layout is **not** a Harbor task directory and should not be auto-detected as Harbor.

Fixture:

- `terminal_bench_legacy/minimal_legacy_task/`
