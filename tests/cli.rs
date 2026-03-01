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

// Stats subcommand tests

#[test]
fn stats_coco_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "stats",
        "--format",
        "coco",
        "tests/fixtures/sample_valid.coco.json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Stats Report"))
        .stdout(predicates::str::contains("Summary"))
        .stdout(predicates::str::contains("Labels"))
        .stdout(predicates::str::contains("Bounding Boxes"));
}

#[test]
fn stats_ir_json_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "stats",
        "--format",
        "ir-json",
        "tests/fixtures/sample_valid.ir.json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Stats Report"))
        .stdout(predicates::str::contains("Images"));
}

#[test]
fn stats_tfod_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "stats",
        "--format",
        "tfod",
        "tests/fixtures/sample_valid.tfod.csv",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Stats Report"))
        .stdout(predicates::str::contains("Annotations"));
}

#[test]
fn stats_voc_dataset_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_voc_dataset(temp.path());

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["stats", "--format", "voc", temp.path().to_str().unwrap()]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Stats Report"))
        .stdout(predicates::str::contains("Images"));
}

#[test]
fn stats_label_studio_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "stats",
        "--format",
        "label-studio",
        "tests/fixtures/sample_valid.label_studio.json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Stats Report"))
        .stdout(predicates::str::contains("Images"));
}

#[test]
fn stats_shows_label_histogram() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "stats",
        "--format",
        "coco",
        "tests/fixtures/sample_valid.coco.json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("person").or(predicates::str::contains("Labels")));
}

#[test]
fn stats_top_flag_limits_labels() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "stats",
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
fn stats_nonexistent_file_fails() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["stats", "--format", "coco", "nonexistent_file.json"]);
    cmd.assert().failure();
}

#[test]
fn stats_auto_detects_coco_when_format_omitted() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["stats", "tests/fixtures/sample_valid.coco.json"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Stats Report"));
}

#[test]
fn stats_falls_back_to_ir_json_when_detection_fails_for_json_file() {
    let temp = tempfile::tempdir().expect("tempdir");
    let p = temp.path().join("empty.ir.json");
    fs::write(
        &p,
        r#"{"info":{},"images":[],"categories":[],"annotations":[]}"#,
    )
    .expect("write");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["stats", p.to_str().unwrap()]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Stats Report"));
}

#[test]
fn stats_directory_detection_errors_without_fallback() {
    let temp = tempfile::tempdir().expect("tempdir");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["stats", temp.path().to_str().unwrap()]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("unrecognized directory layout"));
}

#[test]
fn stats_json_output_contains_expected_keys() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "stats",
        "--output",
        "json",
        "tests/fixtures/sample_valid.coco.json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"summary\""))
        .stdout(predicates::str::contains("\"labels\""))
        .stdout(predicates::str::contains("\"bboxes\""))
        .stdout(predicates::str::contains("\"cooccurrence_top_pairs\""));
}

#[test]
fn stats_html_output_contains_expected_markers() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "stats",
        "--output",
        "html",
        "tests/fixtures/sample_valid.coco.json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("<title>panlabel stats</title>"))
        .stdout(predicates::str::contains("id=\"stats-data\""))
        .stdout(predicates::str::contains("id=\"labels-chart\""));
}

// Diff subcommand tests

#[test]
fn diff_identical_ir_json_has_no_changes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let a = temp.path().join("a.ir.json");
    let b = temp.path().join("b.ir.json");
    let content = fs::read_to_string("tests/fixtures/sample_valid.ir.json").expect("read fixture");
    fs::write(&a, &content).expect("write a");
    fs::write(&b, &content).expect("write b");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "diff",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--format-a",
        "ir-json",
        "--format-b",
        "ir-json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dataset Diff"))
        .stdout(predicates::str::contains("0 only in A, 0 only in B"));
}

#[test]
fn diff_id_mode_bbox_change_within_epsilon_is_not_modified() {
    let temp = tempfile::tempdir().expect("tempdir");
    let a = temp.path().join("a.ir.json");
    let b = temp.path().join("b.ir.json");

    let a_json = r#"{"info":{},"images":[{"id":1,"file_name":"img.jpg","width":100,"height":100}],"categories":[{"id":1,"name":"cat"}],"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":{"xmin":10.0,"ymin":10.0,"xmax":20.0,"ymax":20.0}}]}"#;
    let b_json = r#"{"info":{},"images":[{"id":1,"file_name":"img.jpg","width":100,"height":100}],"categories":[{"id":1,"name":"cat"}],"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":{"xmin":10.0000005,"ymin":10.0,"xmax":20.0,"ymax":20.0}}]}"#;

    fs::write(&a, a_json).expect("write a");
    fs::write(&b, b_json).expect("write b");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "diff",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--format-a",
        "ir-json",
        "--format-b",
        "ir-json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("modified (0)"));
}

#[test]
fn diff_id_mode_bbox_change_beyond_epsilon_is_modified() {
    let temp = tempfile::tempdir().expect("tempdir");
    let a = temp.path().join("a.ir.json");
    let b = temp.path().join("b.ir.json");

    let a_json = r#"{"info":{},"images":[{"id":1,"file_name":"img.jpg","width":100,"height":100}],"categories":[{"id":1,"name":"cat"}],"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":{"xmin":10.0,"ymin":10.0,"xmax":20.0,"ymax":20.0}}]}"#;
    let b_json = r#"{"info":{},"images":[{"id":1,"file_name":"img.jpg","width":100,"height":100}],"categories":[{"id":1,"name":"cat"}],"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":{"xmin":10.0001,"ymin":10.0,"xmax":20.0,"ymax":20.0}}]}"#;

    fs::write(&a, a_json).expect("write a");
    fs::write(&b, b_json).expect("write b");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "diff",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--format-a",
        "ir-json",
        "--format-b",
        "ir-json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("modified (1)"));
}

#[test]
fn diff_detail_prints_detail_sections() {
    let temp = tempfile::tempdir().expect("tempdir");
    let a = temp.path().join("a.ir.json");
    let b = temp.path().join("b.ir.json");

    let a_json = r#"{"info":{},"images":[{"id":1,"file_name":"a.jpg","width":10,"height":10}],"categories":[{"id":1,"name":"cat"}],"annotations":[]}"#;
    let b_json = r#"{"info":{},"images":[{"id":1,"file_name":"b.jpg","width":10,"height":10}],"categories":[{"id":1,"name":"cat"}],"annotations":[]}"#;
    fs::write(&a, a_json).expect("write a");
    fs::write(&b, b_json).expect("write b");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "diff",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--format-a",
        "ir-json",
        "--format-b",
        "ir-json",
        "--detail",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Images only in A:"))
        .stdout(predicates::str::contains("Images only in B:"));
}

#[test]
fn diff_json_output_contains_expected_keys() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "diff",
        "tests/fixtures/sample_valid.ir.json",
        "tests/fixtures/sample_valid.ir.json",
        "--format-a",
        "ir-json",
        "--format-b",
        "ir-json",
        "--output",
        "json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"images\""))
        .stdout(predicates::str::contains("\"categories\""))
        .stdout(predicates::str::contains("\"annotations\""));
}

#[test]
fn diff_iou_mode_matches_different_ids() {
    let temp = tempfile::tempdir().expect("tempdir");
    let a = temp.path().join("a.ir.json");
    let b = temp.path().join("b.ir.json");

    let a_json = r#"{"info":{},"images":[{"id":1,"file_name":"img.jpg","width":100,"height":100}],"categories":[{"id":1,"name":"cat"}],"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":{"xmin":10.0,"ymin":10.0,"xmax":20.0,"ymax":20.0}}]}"#;
    let b_json = r#"{"info":{},"images":[{"id":1,"file_name":"img.jpg","width":100,"height":100}],"categories":[{"id":1,"name":"cat"}],"annotations":[{"id":999,"image_id":1,"category_id":1,"bbox":{"xmin":10.0,"ymin":10.0,"xmax":20.0,"ymax":20.0}}]}"#;

    fs::write(&a, a_json).expect("write a");
    fs::write(&b, b_json).expect("write b");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "diff",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--format-a",
        "ir-json",
        "--format-b",
        "ir-json",
        "--match-by",
        "iou",
        "--iou-threshold",
        "0.5",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Annotations:"))
        .stdout(predicates::str::contains("modified (0)"));
}

// Sample subcommand tests

#[test]
fn sample_n_writes_output_and_validates() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("out.ir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "sample",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        out.to_str().unwrap(),
        "--from",
        "coco",
        "--to",
        "ir-json",
        "-n",
        "1",
    ]);
    cmd.assert().success();

    let mut validate = cargo_bin_cmd!("panlabel");
    validate.args(["validate", out.to_str().unwrap(), "--format", "ir-json"]);
    validate
        .assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn sample_fraction_writes_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("out.ir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "sample",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        out.to_str().unwrap(),
        "--from",
        "coco",
        "--to",
        "ir-json",
        "--fraction",
        "0.5",
        "--seed",
        "42",
    ]);
    cmd.assert().success();
    assert!(out.is_file());
}

#[test]
fn sample_rejects_n_and_fraction_together() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("out.ir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "sample",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        out.to_str().unwrap(),
        "--from",
        "coco",
        "--to",
        "ir-json",
        "-n",
        "1",
        "--fraction",
        "0.5",
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Invalid sample parameters"));
}

#[test]
fn sample_seed_is_deterministic() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out1 = temp.path().join("out1.ir.json");
    let out2 = temp.path().join("out2.ir.json");

    for out in [&out1, &out2] {
        let mut cmd = cargo_bin_cmd!("panlabel");
        cmd.args([
            "sample",
            "-i",
            "tests/fixtures/sample_valid.coco.json",
            "-o",
            out.to_str().unwrap(),
            "--from",
            "coco",
            "--to",
            "ir-json",
            "-n",
            "1",
            "--seed",
            "123",
        ]);
        cmd.assert().success();
    }

    let b1 = fs::read(&out1).expect("read out1");
    let b2 = fs::read(&out2).expect("read out2");
    assert_eq!(b1, b2);
}

#[test]
fn sample_category_mode_annotations_keeps_all_categories() {
    let temp = tempfile::tempdir().expect("tempdir");
    let input = temp.path().join("in.ir.json");
    let out = temp.path().join("out.ir.json");

    let ds = r#"{
      "info": {},
      "images": [{"id":1,"file_name":"img.jpg","width":10,"height":10}],
      "categories": [{"id":1,"name":"person"},{"id":2,"name":"dog"}],
      "annotations": [
        {"id":1,"image_id":1,"category_id":1,"bbox":{"xmin":1.0,"ymin":1.0,"xmax":2.0,"ymax":2.0}},
        {"id":2,"image_id":1,"category_id":2,"bbox":{"xmin":3.0,"ymin":3.0,"xmax":4.0,"ymax":4.0}}
      ]
    }"#;
    fs::write(&input, ds).expect("write input");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "sample",
        "-i",
        input.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--from",
        "ir-json",
        "--to",
        "ir-json",
        "-n",
        "1",
        "--categories",
        "person",
        "--category-mode",
        "annotations",
        "--seed",
        "1",
    ]);
    cmd.assert().success();

    let out_s = fs::read_to_string(&out).expect("read out");
    let out_v: serde_json::Value = serde_json::from_str(&out_s).expect("parse json");

    assert_eq!(out_v["categories"].as_array().map(|v| v.len()), Some(2));
    let annotations = out_v["annotations"].as_array().expect("annotations array");
    assert!(annotations.iter().all(|ann| ann["category_id"] == 1));
}

#[test]
fn sample_to_tfod_is_blocked_without_allow_lossy() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("out.csv");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "sample",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        out.to_str().unwrap(),
        "--from",
        "coco",
        "--to",
        "tfod",
        "-n",
        "1",
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Lossy conversion"))
        .stderr(predicates::str::contains("--allow-lossy"));
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
