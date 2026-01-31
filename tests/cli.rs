use assert_cmd::cargo::cargo_bin_cmd;

#[test]
fn runs() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.assert().success();
}

#[test]
fn outputs_tool_name() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.arg("-V");
    cmd.assert().success().stdout("panlabel 0.1.0\n");
}

// Validate subcommand tests

#[test]
fn validate_valid_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["validate", "tests/fixtures/sample_valid.ir.json"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_invalid_dataset_fails() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["validate", "tests/fixtures/sample_invalid.ir.json"]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("error(s)"));
}

#[test]
fn validate_reports_duplicate_ids() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["validate", "tests/fixtures/sample_invalid.ir.json"]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("DuplicateImageId"));
}

#[test]
fn validate_reports_missing_refs() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["validate", "tests/fixtures/sample_invalid.ir.json"]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("MissingImageRef"))
        .stdout(predicates::str::contains("MissingCategoryRef"));
}

#[test]
fn validate_json_output_format() {
    let mut cmd = cargo_bin_cmd!("panlabel");
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
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["validate", "nonexistent_file.json"]);
    cmd.assert().failure();
}

// COCO format tests

#[test]
fn validate_coco_valid_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.coco.json",
        "--format",
        "coco",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_coco_invalid_dataset_fails() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_invalid.coco.json",
        "--format",
        "coco",
    ]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("error(s)"));
}

#[test]
fn validate_coco_reports_duplicate_ids() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_invalid.coco.json",
        "--format",
        "coco",
    ]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("DuplicateImageId"));
}

#[test]
fn validate_coco_reports_missing_refs() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_invalid.coco.json",
        "--format",
        "coco",
    ]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("MissingImageRef"))
        .stdout(predicates::str::contains("MissingCategoryRef"));
}

#[test]
fn validate_coco_json_alias_works() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.coco.json",
        "--format",
        "coco-json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

// TFOD CSV format tests

#[test]
fn validate_tfod_valid_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.tfod.csv",
        "--format",
        "tfod",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_tfod_csv_alias_works() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.tfod.csv",
        "--format",
        "tfod-csv",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
#[ignore] // Requires large generated dataset in assets/ (not committed)
fn validate_tfod_large_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "assets/tfod_annotations.csv",
        "--format",
        "tfod",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

// Unsupported format test (uses a truly unsupported format now)

#[test]
fn validate_unsupported_format_fails() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.ir.json",
        "--format",
        "not-a-format",
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Unsupported format"));
}
