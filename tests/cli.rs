use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::PredicateBooleanExt;

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

// Convert subcommand tests

#[test]
fn convert_coco_to_ir_json_succeeds() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_coco_to_ir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"))
        .stdout(predicates::str::contains("2 images"))
        .stdout(predicates::str::contains("3 annotations"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_ir_json_to_coco_succeeds() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_ir_to_coco.json");

    // IR JSON may have info.name and attributes that COCO doesn't preserve
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ir-json",
        "-t",
        "coco",
        "-i",
        "tests/fixtures/sample_valid.ir.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_tfod_to_coco_succeeds() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_tfod_to_coco.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "tfod",
        "-t",
        "coco",
        "-i",
        "tests/fixtures/sample_valid.tfod.csv",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_coco_to_tfod_fails_without_allow_lossy() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_coco_to_tfod.csv");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "tfod",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Lossy conversion"))
        .stderr(predicates::str::contains("--allow-lossy"));
}

#[test]
fn convert_coco_to_tfod_succeeds_with_allow_lossy() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_coco_to_tfod_lossy.csv");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "tfod",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_invalid_input_fails_validation() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_invalid.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "ir-json",
        "-i",
        "tests/fixtures/sample_invalid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Validation failed"));
}

#[test]
fn convert_invalid_input_succeeds_with_no_validate() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_no_validate.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "ir-json",
        "-i",
        "tests/fixtures/sample_invalid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--no-validate",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_format_aliases_work() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_aliases.csv");

    // Test "coco-json" alias and "tfod-csv" alias
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco-json",
        "-t",
        "tfod-csv",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_nonexistent_file_fails() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "ir-json",
        "-i",
        "nonexistent_file.json",
        "-o",
        "/tmp/output.json",
    ]);
    cmd.assert().failure();
}

// ConversionReport tests

#[test]
fn convert_report_json_output_format() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_report_json.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--report",
        "json",
    ]);

    // JSON report should not contain "Converted" text, only JSON
    // Note: pretty-printed JSON has spaces after colons
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"from\": \"coco\""))
        .stdout(predicates::str::contains("\"to\": \"ir-json\""))
        .stdout(predicates::str::contains("\"input\""))
        .stdout(predicates::str::contains("\"output\""))
        .stdout(predicates::str::contains("\"issues\""))
        .stdout(predicates::str::contains("Converted").not());

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_report_json_includes_lossy_warnings() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_report_lossy.csv");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "tfod",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--report",
        "json",
    ]);

    // JSON report should include lossiness warnings
    // Note: pretty-printed JSON has spaces after colons
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"severity\": \"warning\""))
        .stdout(predicates::str::contains("\"code\":"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_report_text_shows_counts() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_report_text.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);

    // Text report should show counts
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"))
        .stdout(predicates::str::contains("images"))
        .stdout(predicates::str::contains("categories"))
        .stdout(predicates::str::contains("annotations"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_tfod_to_coco_shows_policy_notes() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_policy_notes.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "tfod",
        "-t",
        "coco",
        "-i",
        "tests/fixtures/sample_valid.tfod.csv",
        "-o",
        output_path.to_str().unwrap(),
        "--report",
        "json",
    ]);

    // JSON report should include policy notes from TFOD reader
    // Note: pretty-printed JSON has spaces after colons
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"severity\": \"info\""))
        .stdout(predicates::str::contains("tfod_reader_id_assignment"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

// Inspect subcommand tests

#[test]
fn inspect_coco_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "inspect",
        "--format",
        "coco",
        "tests/fixtures/sample_valid.coco.json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Inspection Report"))
        .stdout(predicates::str::contains("Summary"))
        .stdout(predicates::str::contains("Labels"))
        .stdout(predicates::str::contains("Bounding Boxes"));
}

#[test]
fn inspect_ir_json_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "inspect",
        "--format",
        "ir-json",
        "tests/fixtures/sample_valid.ir.json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Inspection Report"))
        .stdout(predicates::str::contains("Images"));
}

#[test]
fn inspect_tfod_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "inspect",
        "--format",
        "tfod",
        "tests/fixtures/sample_valid.tfod.csv",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Inspection Report"))
        .stdout(predicates::str::contains("Annotations"));
}

#[test]
fn inspect_shows_label_histogram() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "inspect",
        "--format",
        "coco",
        "tests/fixtures/sample_valid.coco.json",
    ]);
    // Should show category names from the dataset
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("person").or(predicates::str::contains("Labels")));
}

#[test]
fn inspect_top_flag_limits_labels() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "inspect",
        "--format",
        "coco",
        "tests/fixtures/sample_valid.coco.json",
        "--top",
        "2",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Labels"));
}

#[test]
fn inspect_nonexistent_file_fails() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "inspect",
        "--format",
        "coco",
        "nonexistent_file.json",
    ]);
    cmd.assert().failure();
}
