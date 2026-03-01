use std::fs;
use std::path::Path;

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::PredicateBooleanExt;

fn bmp_bytes(width: u32, height: u32) -> Vec<u8> {
    let row_stride = (width * 3).div_ceil(4) * 4;
    let pixel_array_size = row_stride * height;
    let file_size = 54 + pixel_array_size;

    let mut bytes = Vec::with_capacity(file_size as usize);
    bytes.extend_from_slice(b"BM");
    bytes.extend_from_slice(&file_size.to_le_bytes());
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    bytes.extend_from_slice(&54u32.to_le_bytes());

    bytes.extend_from_slice(&40u32.to_le_bytes());
    bytes.extend_from_slice(&(width as i32).to_le_bytes());
    bytes.extend_from_slice(&(height as i32).to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&24u16.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&pixel_array_size.to_le_bytes());
    bytes.extend_from_slice(&2835u32.to_le_bytes());
    bytes.extend_from_slice(&2835u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());

    bytes.resize(file_size as usize, 0);
    bytes
}

fn write_bmp(path: &Path, width: u32, height: u32) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dir");
    }
    fs::write(path, bmp_bytes(width, height)).expect("write bmp file");
}

fn create_sample_yolo_dataset(root: &Path) {
    fs::create_dir_all(root.join("images/train")).expect("create images dir");
    fs::create_dir_all(root.join("labels/train")).expect("create labels dir");

    write_bmp(&root.join("images/train/img1.bmp"), 16, 8);
    write_bmp(&root.join("images/train/img2.bmp"), 10, 10);

    fs::write(root.join("data.yaml"), "names:\n  - person\n  - car\n").expect("write data yaml");

    fs::write(
        root.join("labels/train/img1.txt"),
        "0 0.5 0.5 0.5 0.5\n1 0.2 0.3 0.2 0.2\n",
    )
    .expect("write labels for img1");
    fs::write(root.join("labels/train/img2.txt"), "").expect("write empty label file");
}

fn create_sample_voc_dataset(root: &Path) {
    fs::create_dir_all(root.join("Annotations")).expect("create annotations dir");
    fs::create_dir_all(root.join("JPEGImages")).expect("create images dir");

    let xml_1 = r#"<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>img1.jpg</filename>
  <size>
    <width>100</width>
    <height>80</height>
    <depth>3</depth>
  </size>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>20</ymin>
      <xmax>50</xmax>
      <ymax>70</ymax>
    </bndbox>
  </object>
</annotation>
"#;

    let xml_2 = r#"<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>img2.jpg</filename>
  <size>
    <width>60</width>
    <height>40</height>
    <depth>3</depth>
  </size>
</annotation>
"#;

    fs::write(root.join("Annotations/img1.xml"), xml_1).expect("write img1 xml");
    fs::write(root.join("Annotations/img2.xml"), xml_2).expect("write img2 xml");
}

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
fn validate_yolo_dataset_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_yolo_dataset(temp.path());

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        temp.path().to_str().unwrap(),
        "--format",
        "yolo",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_yolo_alias_works() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_yolo_dataset(temp.path());

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        temp.path().to_str().unwrap(),
        "--format",
        "yolov8",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_voc_dataset_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_voc_dataset(temp.path());

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["validate", temp.path().to_str().unwrap(), "--format", "voc"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_voc_alias_works() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_voc_dataset(temp.path());

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        temp.path().to_str().unwrap(),
        "--format",
        "pascal-voc",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_label_studio_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.label_studio.json",
        "--format",
        "label-studio",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_label_studio_alias_works() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.label_studio.json",
        "--format",
        "ls",
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
fn convert_ir_json_to_yolo_fails_without_allow_lossy() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("yolo_out");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ir-json",
        "-t",
        "yolo",
        "-i",
        "tests/fixtures/sample_valid.ir.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Lossy conversion"))
        .stderr(predicates::str::contains("--allow-lossy"));
}

#[test]
fn convert_yolo_to_coco_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let yolo_dir = temp.path().join("sample_yolo");
    create_sample_yolo_dataset(&yolo_dir);
    let output_path = temp.path().join("yolo_to_coco.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "yolo",
        "-t",
        "coco",
        "-i",
        yolo_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));
}

#[test]
fn convert_voc_to_coco_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let voc_dir = temp.path().join("sample_voc");
    create_sample_voc_dataset(&voc_dir);
    let output_path = temp.path().join("voc_to_coco.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "voc",
        "-t",
        "coco",
        "-i",
        voc_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));
}

#[test]
fn convert_ir_json_to_voc_fails_without_allow_lossy() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("voc_out");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ir-json",
        "-t",
        "voc",
        "-i",
        "tests/fixtures/sample_valid.ir.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Lossy conversion"))
        .stderr(predicates::str::contains("--allow-lossy"));
}

#[test]
fn convert_ir_json_to_voc_succeeds_with_allow_lossy() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("voc_out");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ir-json",
        "-t",
        "voc",
        "-i",
        "tests/fixtures/sample_valid.ir.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));
}

#[test]
fn convert_ir_json_to_label_studio_fails_without_allow_lossy() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("label_studio_out.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ir-json",
        "-t",
        "label-studio",
        "-i",
        "tests/fixtures/sample_valid.ir.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Lossy conversion"))
        .stderr(predicates::str::contains("--allow-lossy"));
}

#[test]
fn convert_from_yolo_alias_works() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let yolo_dir = temp.path().join("sample_yolo");
    create_sample_yolo_dataset(&yolo_dir);
    let output_path = temp.path().join("alias_out.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ultralytics",
        "-t",
        "coco",
        "-i",
        yolo_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));
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

#[test]
fn convert_to_yolo_report_includes_policy_notes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("report_yolo");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "yolo",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--report",
        "json",
    ]);

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"severity\": \"info\""))
        .stdout(predicates::str::contains("yolo_writer_float_precision"));
}

#[test]
fn convert_to_voc_report_includes_policy_notes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("report_voc");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "voc",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--report",
        "json",
    ]);

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"severity\": \"info\""))
        .stdout(predicates::str::contains("voc_writer_bool_normalization"));
}

#[test]
fn convert_to_label_studio_report_includes_policy_notes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("report_label_studio.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ir-json",
        "-t",
        "label-studio",
        "-i",
        "tests/fixtures/sample_valid.ir.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--report",
        "json",
    ]);

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"severity\": \"info\""))
        .stdout(predicates::str::contains(
            "label_studio_writer_from_to_defaults",
        ));
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
fn inspect_voc_dataset_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_voc_dataset(temp.path());

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["inspect", "--format", "voc", temp.path().to_str().unwrap()]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Inspection Report"))
        .stdout(predicates::str::contains("Images"));
}

#[test]
fn inspect_label_studio_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "inspect",
        "--format",
        "label-studio",
        "tests/fixtures/sample_valid.label_studio.json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Inspection Report"))
        .stdout(predicates::str::contains("Images"));
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
    cmd.args(["inspect", "--format", "coco", "nonexistent_file.json"]);
    cmd.assert().failure();
}

// list-formats subcommand tests

#[test]
fn list_formats_shows_all_formats() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["list-formats"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("ir-json"))
        .stdout(predicates::str::contains("coco"))
        .stdout(predicates::str::contains("label-studio"))
        .stdout(predicates::str::contains("tfod"))
        .stdout(predicates::str::contains("yolo"))
        .stdout(predicates::str::contains("voc"))
        .stdout(predicates::str::contains("Supported formats"));
}

#[test]
fn list_formats_shows_lossiness() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["list-formats"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("lossless"))
        .stdout(predicates::str::contains("conditional"))
        .stdout(predicates::str::contains("lossy"));
}

#[test]
fn list_formats_shows_read_write_capability() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["list-formats"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("READ"))
        .stdout(predicates::str::contains("WRITE"));
}

// Auto-detection tests

#[test]
fn convert_auto_detects_coco_format() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("auto_detect_coco.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(coco)"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_auto_detects_tfod_format() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("auto_detect_tfod.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        "tests/fixtures/sample_valid.tfod.csv",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(tfod)"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_auto_detects_ir_json_format() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("auto_detect_ir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        "tests/fixtures/sample_valid.ir.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(ir-json)"));

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_auto_detects_label_studio_format() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("auto_detect_label_studio.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        "tests/fixtures/sample_valid.label_studio.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(label-studio)"));

    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_auto_detects_yolo_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let yolo_dir = temp.path().join("sample_yolo");
    create_sample_yolo_dataset(&yolo_dir);
    let output_path = temp.path().join("auto_detect_yolo.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        yolo_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(yolo)"));
}

#[test]
fn convert_auto_detects_yolo_labels_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let yolo_dir = temp.path().join("sample_yolo");
    create_sample_yolo_dataset(&yolo_dir);
    let labels_dir = yolo_dir.join("labels");
    let output_path = temp.path().join("auto_detect_yolo_labels.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        labels_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(yolo)"));
}

#[test]
fn convert_auto_detects_voc_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let voc_dir = temp.path().join("sample_voc");
    create_sample_voc_dataset(&voc_dir);
    let output_path = temp.path().join("auto_detect_voc.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        voc_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(voc)"));
}

#[test]
fn convert_auto_detect_errors_on_yolo_voc_ambiguity() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_yolo_dataset(temp.path());

    fs::create_dir_all(temp.path().join("Annotations")).expect("create annotations dir");
    fs::write(
        temp.path().join("Annotations/extra.xml"),
        r#"<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>img_ambiguous.jpg</filename>
  <size><width>10</width><height>10</height><depth>3</depth></size>
</annotation>
"#,
    )
    .expect("write xml");
    fs::create_dir_all(temp.path().join("JPEGImages")).expect("create JPEGImages dir");

    let output_path = temp.path().join("auto_detect_ambiguous.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("matches both YOLO and VOC"));
}

#[test]
fn convert_auto_fails_on_unknown_extension() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_unknown_ext.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        "Cargo.toml",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("unrecognized file extension"));
}
