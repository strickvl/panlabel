use std::fs;
use std::process::Output;

use assert_cmd::cargo::cargo_bin_cmd;

fn stdout_json(output: &Output) -> serde_json::Value {
    let stdout = String::from_utf8(output.stdout.clone()).expect("stdout utf8");
    serde_json::from_str(&stdout).expect("stdout json")
}

#[test]
fn text_list_formats_marks_first_batch_capabilities() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["text", "list-formats", "--output", "json"]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());
    let formats = stdout_json(&output);
    let formats = formats.as_array().expect("array");

    let rlvr = formats
        .iter()
        .find(|entry| entry["name"] == "rlvr-hf")
        .expect("rlvr");
    assert_eq!(rlvr["read"], true);
    assert_eq!(rlvr["write"], true);

    let swe = formats
        .iter()
        .find(|entry| entry["name"] == "swe-bench")
        .expect("swe");
    assert_eq!(swe["read"], true);
    assert_eq!(swe["write"], false);
}

#[test]
fn text_convert_auto_detects_rlvr_hf_to_text_ir() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output_path = temp.path().join("tasks.jsonl");
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "text",
        "convert",
        "--from",
        "auto",
        "--to",
        "text-ir-jsonl",
        "-i",
        "tests/fixtures/text_task_formats/rlvr_hf/allenai_rlvr_ifeval_shape.jsonl",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert().success();

    let row = fs::read_to_string(&output_path).expect("read text ir");
    assert!(row.contains("Describe IPv6 in lowercase"));
    assert!(row.contains("prompt_column"));
    assert!(temp.path().join("tasks.meta.json").exists());
}

#[test]
fn text_convert_reads_verifiers_directory_without_executing_python() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output_path = temp.path().join("verifiers.jsonl");
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "text",
        "convert",
        "--from",
        "auto",
        "--to",
        "text-ir-jsonl",
        "-i",
        "tests/fixtures/text_task_formats/verifiers_taskset",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert().success();

    let rows = fs::read_to_string(&output_path).expect("read text ir");
    assert!(rows.contains("verifiers.artifacts/row-000001/rubric.py"));
    assert!(rows.contains("verifiers.artifacts/row-000001/env.py"));
    assert!(rows.contains("What is 2+2?"));
    assert!(temp
        .path()
        .join("verifiers.artifacts/row-000001/rubric.py")
        .exists());
    assert!(temp
        .path()
        .join("verifiers.artifacts/row-000001/env.py")
        .exists());
}

#[test]
fn text_convert_harbor_to_rlvr_blocks_then_allows_lossy() {
    let temp = tempfile::tempdir().expect("tempdir");
    let blocked_output = temp.path().join("blocked.jsonl");
    let mut blocked = cargo_bin_cmd!("panlabel");
    blocked.args([
        "text",
        "convert",
        "--from",
        "auto",
        "--to",
        "rlvr-hf",
        "-i",
        "tests/fixtures/text_task_formats/harbor/single_task_minimal",
        "-o",
        blocked_output.to_str().unwrap(),
    ]);
    blocked
        .assert()
        .failure()
        .stderr(predicates::str::contains("Lossy text conversion"));
    assert!(!blocked_output.exists());

    let allowed_output = temp.path().join("allowed.jsonl");
    let mut allowed = cargo_bin_cmd!("panlabel");
    allowed.args([
        "text",
        "convert",
        "--from",
        "auto",
        "--to",
        "rlvr-hf",
        "-i",
        "tests/fixtures/text_task_formats/harbor/single_task_minimal",
        "-o",
        allowed_output.to_str().unwrap(),
        "--allow-lossy",
    ]);
    allowed.assert().success();
    assert!(allowed_output.exists());
    let row = fs::read_to_string(&allowed_output).expect("read rlvr output");
    assert!(row.contains("allowed.artifacts/panlabel-fixtures_minimal-harbor/tests/test.sh"));
    assert!(temp
        .path()
        .join("allowed.artifacts/panlabel-fixtures_minimal-harbor/tests/test.sh")
        .exists());
    assert!(temp
        .path()
        .join("allowed.artifacts/panlabel-fixtures_minimal-harbor/solution/solve.sh")
        .exists());
    assert!(temp
        .path()
        .join("allowed.artifacts/panlabel-fixtures_minimal-harbor/environment/Dockerfile")
        .exists());
}

#[test]
fn text_convert_harbor_to_harbor_copies_native_artifacts() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output_dir = temp.path().join("harbor_native");
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "text",
        "convert",
        "--from",
        "harbor",
        "--to",
        "harbor",
        "-i",
        "tests/fixtures/text_task_formats/harbor/single_task_minimal",
        "-o",
        output_dir.to_str().unwrap(),
    ]);
    cmd.assert().success();

    assert!(output_dir.join("task.toml").exists());
    assert!(output_dir.join("instruction.md").exists());
    assert!(output_dir.join("tests/test.sh").exists());
    assert!(output_dir.join("solution/solve.sh").exists());
    assert!(output_dir.join("environment/Dockerfile").exists());
    assert!(fs::read_to_string(output_dir.join("tests/test.sh"))
        .expect("read copied verifier")
        .contains("/app/answer.txt"));
}

#[test]
fn text_convert_rlvr_to_harbor_requires_scaffold() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output_dir = temp.path().join("harbor_out");
    let mut missing = cargo_bin_cmd!("panlabel");
    missing.args([
        "text",
        "convert",
        "--from",
        "rlvr-hf",
        "--to",
        "harbor",
        "-i",
        "tests/fixtures/text_task_formats/rlvr_hf/allenai_rlvr_ifeval_shape.jsonl",
        "-o",
        output_dir.to_str().unwrap(),
        "--allow-lossy",
    ]);
    missing
        .assert()
        .failure()
        .stderr(predicates::str::contains("--scaffold"));

    let mut scaffold = cargo_bin_cmd!("panlabel");
    scaffold.args([
        "text",
        "convert",
        "--from",
        "rlvr-hf",
        "--to",
        "harbor",
        "-i",
        "tests/fixtures/text_task_formats/rlvr_hf/allenai_rlvr_ifeval_shape.jsonl",
        "-o",
        output_dir.to_str().unwrap(),
        "--scaffold",
        "--allow-lossy",
    ]);
    scaffold
        .assert()
        .success()
        .stdout(predicates::str::contains("TASK-REWARD-STUB"));
    assert!(output_dir.join("task.toml").exists());
    assert!(output_dir.join("tests/test.sh").exists());
}

#[test]
fn text_convert_swe_bench_preserves_patches_outside_prompt() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output_path = temp.path().join("swe.jsonl");
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "text",
        "convert",
        "--from",
        "auto",
        "--to",
        "text-ir-jsonl",
        "-i",
        "tests/fixtures/text_task_formats/swe_bench/minimal_instance.jsonl",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("TASK-SWEBENCH-PATCH-PRESERVED"));

    let row = fs::read_to_string(&output_path).expect("read text ir");
    let value: serde_json::Value =
        serde_json::from_str(row.lines().next().unwrap()).expect("json row");
    let input = value["data"]["input"]["text"].as_str().expect("input text");
    assert!(input.contains("Fix the example bug"));
    assert!(!input.contains("diff --git"));
    assert!(value
        .to_string()
        .contains("swe.artifacts/example__project-1/test_patch.diff"));
    assert!(value
        .to_string()
        .contains("swe.artifacts/example__project-1/gold_patch.diff"));
    assert!(temp
        .path()
        .join("swe.artifacts/example__project-1/test_patch.diff")
        .exists());
    assert!(temp
        .path()
        .join("swe.artifacts/example__project-1/gold_patch.diff")
        .exists());
}

#[test]
fn text_validate_rejects_terminal_bench_legacy_as_auto_harbor() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "text",
        "validate",
        "tests/fixtures/text_task_formats/terminal_bench_legacy/minimal_legacy_task",
    ]);
    cmd.assert().failure().stderr(predicates::str::contains(
        "unrecognized text/task directory layout",
    ));
}

#[test]
fn text_convert_dry_run_reports_artifact_destinations_but_writes_no_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output_path = temp.path().join("dry.jsonl");
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "text",
        "convert",
        "--from",
        "auto",
        "--to",
        "text-ir-jsonl",
        "-i",
        "tests/fixtures/text_task_formats/harbor/single_task_minimal",
        "-o",
        output_path.to_str().unwrap(),
        "--dry-run",
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dry run"))
        .stdout(predicates::str::contains("artifact destinations:"))
        .stdout(predicates::str::contains(
            "dry.artifacts/panlabel-fixtures_minimal-harbor/tests/test.sh",
        ));
    assert!(!output_path.exists());
    assert!(!temp.path().join("dry.artifacts").exists());
}
