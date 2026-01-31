use assert_cmd::Command;

#[test]
fn runs() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.assert().success();
}

#[test]
fn outputs_tool_name() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.arg("-V");
    cmd.assert().success().stdout("panlabel 0.1.0\n");
}

// Validate subcommand tests

#[test]
fn validate_valid_dataset_succeeds() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.args(["validate", "tests/fixtures/sample_valid.ir.json"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_invalid_dataset_fails() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.args(["validate", "tests/fixtures/sample_invalid.ir.json"]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("error(s)"));
}

#[test]
fn validate_reports_duplicate_ids() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.args(["validate", "tests/fixtures/sample_invalid.ir.json"]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("DuplicateImageId"));
}

#[test]
fn validate_reports_missing_refs() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.args(["validate", "tests/fixtures/sample_invalid.ir.json"]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("MissingImageRef"))
        .stdout(predicates::str::contains("MissingCategoryRef"));
}

#[test]
fn validate_json_output_format() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.ir.json",
        "--output",
        "json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"error_count\": 0"))
        .stdout(predicates::str::contains("\"warning_count\": 0"));
}

#[test]
fn validate_nonexistent_file_fails() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.args(["validate", "nonexistent_file.json"]);
    cmd.assert().failure();
}

#[test]
fn validate_unsupported_format_fails() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.ir.json",
        "--format",
        "coco",
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Unsupported format"));
}
