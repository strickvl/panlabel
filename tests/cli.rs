use std::fs;
use std::path::Path;
use std::process::Output;

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

fn create_sample_cvat_export(root: &Path) {
    fs::create_dir_all(root).expect("create cvat root");
    let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label><name>person</name><type>bbox</type></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="img1.jpg" width="100" height="80">
    <box label="person" occluded="0" xtl="10" ytl="20" xbr="50" ybr="70" z_order="0" source="manual"/>
  </image>
</annotations>
"#;
    fs::write(root.join("annotations.xml"), xml).expect("write cvat annotations.xml");
}

fn marmot_hex(value: f64) -> String {
    value
        .to_be_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn marmot_rect(x_left: f64, y_top: f64, x_right: f64, y_bottom: f64) -> String {
    [x_left, y_top, x_right, y_bottom]
        .into_iter()
        .map(marmot_hex)
        .collect::<Vec<_>>()
        .join(" ")
}

fn create_sample_marmot_dataset(root: &Path) {
    fs::create_dir_all(root).expect("create marmot root");
    write_bmp(&root.join("page1.bmp"), 200, 100);
    let xml = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Page CropBox="{}" PageNum="1">
  <Contents>
    <Composites Label="TableBody">
      <Composite BBox="{}" LID="1" Label="TableBody" />
    </Composites>
  </Contents>
</Page>
"#,
        marmot_rect(0.0, 100.0, 200.0, 0.0),
        marmot_rect(20.0, 80.0, 120.0, 30.0),
    );
    fs::write(root.join("page1.xml"), xml).expect("write marmot xml");
}

fn create_sample_cloud_annotations_dataset(root: &Path) {
    fs::create_dir_all(root).expect("create cloud annotations root");
    write_bmp(&root.join("img1.bmp"), 100, 80);
    fs::write(
        root.join("_annotations.json"),
        r#"{
            "version": "1.0",
            "type": "localization",
            "labels": ["person"],
            "annotations": {
                "img1.bmp": [{"x": 0.1, "y": 0.25, "x2": 0.5, "y2": 0.875, "label": "person"}]
            }
        }"#,
    )
    .expect("write cloud annotations json");
}

fn create_sample_vott_json_dataset(root: &Path) {
    fs::create_dir_all(root.join("vott-json-export")).expect("create vott export root");
    fs::write(
        root.join("vott-json-export/panlabel-export.json"),
        r##"{
            "name": "sample-vott-json",
            "version": "2.2.0",
            "tags": [{"name": "person", "color": "#e6194b"}],
            "assets": {
                "asset-img1": {
                    "asset": {"id": "asset-img1", "name": "img1.bmp", "path": "file:img1.bmp", "size": {"width": 100, "height": 80}, "format": "bmp"},
                    "regions": [{"id": "r1", "type": "RECTANGLE", "tags": ["person"], "boundingBox": {"left": 10, "top": 20, "width": 40, "height": 50}}],
                    "version": "2.2.0"
                }
            }
        }"##,
    )
    .expect("write vott json");
}

fn create_sample_labelbox_jsonl(path: &Path) {
    fs::write(
        path,
        r#"{"data_row":{"id":"dr-1","external_id":"img1.jpg","row_data":"s3://bucket/img1.jpg"},"media_attributes":{"width":100,"height":80},"projects":{"project-a":{"labels":[{"annotations":{"objects":[{"feature_id":"bbox-1","name":"person","annotation_kind":"ImageBoundingBox","bounding_box":{"top":20,"left":10,"height":30,"width":40}}]}}]}}}
"#,
    )
    .expect("write labelbox jsonl");
}

fn create_sample_scale_ai_dataset(root: &Path) {
    fs::create_dir_all(root.join("annotations")).expect("create scale annotations dir");
    fs::write(
        root.join("annotations/img1.json"),
        r#"{
            "task_id": "task-img1",
            "type": "imageannotation",
            "params": {"attachment": "img1.jpg", "metadata": {"width": 100, "height": 80}},
            "response": {"annotations": [
                {"type": "box", "label": "person", "left": 10, "top": 20, "width": 40, "height": 50}
            ]}
        }"#,
    )
    .expect("write scale ai json");
}

fn create_sample_unity_perception_dataset(root: &Path) {
    fs::create_dir_all(root.join("sequence.0")).expect("create unity sequence dir");
    fs::write(
        root.join("sequence.0/step0.frame_data.json"),
        r#"{
            "frame": 0,
            "sequence": 0,
            "step": 0,
            "captures": [{
                "@type": "type.unity.com/unity.solo.RGBCamera",
                "id": "camera",
                "filename": "img1.png",
                "dimension": [100, 80],
                "annotations": [{
                    "@type": "type.unity.com/unity.solo.BoundingBox2DAnnotation",
                    "id": "bbox-def",
                    "values": [{"label_id": 1, "label_name": "person", "instance_id": 1, "x": 10, "y": 20, "width": 40, "height": 50}]
                }]
            }]
        }"#,
    )
    .expect("write unity perception frame json");
}

fn create_sample_hf_dataset(root: &Path, xyxy: bool) {
    fs::create_dir_all(root).expect("create hf root");
    write_bmp(&root.join("img1.bmp"), 100, 80);
    write_bmp(&root.join("img2.bmp"), 50, 40);

    let bbox_row_1 = if xyxy {
        "[[10,20,50,70]]"
    } else {
        "[[10,20,40,50]]"
    };
    let bbox_row_2 = if xyxy {
        "[[5,5,20,20]]"
    } else {
        "[[5,5,15,15]]"
    };

    let metadata = format!(
        "{{\"file_name\":\"img1.bmp\",\"width\":100,\"height\":80,\"objects\":{{\"bbox\":{},\"categories\":[\"person\"]}}}}\n{{\"file_name\":\"img2.bmp\",\"width\":50,\"height\":40,\"objects\":{{\"bbox\":{},\"categories\":[\"car\"]}}}}\n",
        bbox_row_1, bbox_row_2
    );

    fs::write(root.join("metadata.jsonl"), metadata).expect("write hf metadata");
}

fn stdout_json(output: &Output) -> (String, serde_json::Value) {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let parsed = serde_json::from_str(&stdout).expect("stdout should be valid JSON");
    (stdout, parsed)
}

fn assert_compact_json(stdout: &str) {
    assert!(
        stdout.ends_with('\n'),
        "JSON stdout should end with a newline"
    );
    assert_eq!(
        stdout.lines().count(),
        1,
        "captured JSON should be a single line"
    );
    assert!(
        !stdout.starts_with("{\n") && !stdout.starts_with("[\n"),
        "captured JSON should be compact rather than pretty-printed"
    );
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
    let expected = format!("panlabel {}\n", env!("CARGO_PKG_VERSION"));
    cmd.assert().success().stdout(expected);
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
        "--output-format",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["error_count"], 0);
    assert_eq!(parsed["warning_count"], 0);
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
fn validate_scaled_yolov4_alias_works() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_yolo_dataset(temp.path());

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        temp.path().to_str().unwrap(),
        "--format",
        "scaled-yolov4",
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
fn validate_superannotate_alias_works() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.superannotate.json",
        "--format",
        "superannotate-json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_supervisely_alias_works() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.supervisely.json",
        "--format",
        "supervisely-json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_cityscapes_alias_works() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.cityscapes.json",
        "--format",
        "cityscapes-json",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_marmot_alias_works() {
    let temp = tempfile::tempdir().expect("tempdir");
    create_sample_marmot_dataset(temp.path());

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        temp.path().to_str().unwrap(),
        "--format",
        "marmot-xml",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_cvat_dataset_succeeds() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.cvat.xml",
        "--format",
        "cvat",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_cvat_alias_works() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.cvat.xml",
        "--format",
        "cvat-xml",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Validation passed"));
}

#[test]
fn validate_hf_dataset_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_hf_dataset(temp.path(), false);

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["validate", temp.path().to_str().unwrap(), "--format", "hf"]);
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
        .stderr(predicates::str::contains("invalid value 'not-a-format'"))
        .stderr(predicates::str::contains("possible values"));
}

#[test]
fn validate_invalid_output_format_fails() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "validate",
        "tests/fixtures/sample_valid.ir.json",
        "--output",
        "yaml",
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("invalid value 'yaml'"))
        .stderr(predicates::str::contains("possible values"));
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
        .stderr(predicates::str::contains("--allow-lossy"))
        // Blocked path now emits the full report to stdout
        .stdout(predicates::str::contains("[drop_dataset_info]"))
        .stdout(predicates::str::contains("Warnings"))
        .stdout(predicates::str::contains("Notes"));
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
fn blocked_convert_json_emits_valid_json_to_stdout() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_blocked_json.csv");

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
        "--output-format",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(!output.status.success());

    // stdout should be valid JSON with issue codes
    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["from"], "coco");
    assert_eq!(parsed["to"], "tfod");
    let issues = parsed["issues"]
        .as_array()
        .expect("issues should be an array");
    assert!(issues.iter().any(|i| i["code"] == "drop_dataset_info"));

    // stderr should have the concise error
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Lossy conversion"));
    assert!(stderr.contains("--allow-lossy"));
}

#[test]
fn blocked_convert_text_shows_stable_codes() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_blocked_text_codes.csv");

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
        // Text output now includes stable codes in brackets
        .stdout(predicates::str::contains("[drop_dataset_info]"))
        .stdout(predicates::str::contains("[drop_licenses]"))
        .stdout(predicates::str::contains("[tfod_writer_row_order]"))
        // Counts are still shown
        .stdout(predicates::str::contains("images"))
        .stdout(predicates::str::contains("annotations"));
}

#[test]
fn success_convert_text_shows_stable_codes() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_success_text_codes.csv");

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
        .stdout(predicates::str::contains("Converted"))
        // Stable codes in text output
        .stdout(predicates::str::contains("[drop_dataset_info]"))
        .stdout(predicates::str::contains("[tfod_writer_row_order]"));

    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_dry_run_text_does_not_overwrite_existing_output() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("existing.ir.json");
    fs::write(&output_path, "keep me").expect("seed output");

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
        "--dry-run",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dry run: would convert"))
        .stdout(predicates::str::contains("images"))
        .stdout(predicates::str::contains("annotations"));

    assert_eq!(
        fs::read_to_string(&output_path).expect("read output"),
        "keep me"
    );
}

#[test]
fn convert_dry_run_json_emits_compact_report_only_and_skips_write() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("dry_run.ir.json");

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
        "--dry-run",
        "--output-format",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());
    assert!(!output_path.exists());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["from"], "coco");
    assert_eq!(parsed["to"], "ir-json");
    assert!(parsed.get("issues").is_some());
    assert!(!stdout.contains("Converted"));
    assert!(!stdout.contains("Dry run"));
}

#[test]
fn convert_dry_run_blocked_lossy_still_errors_without_writing() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("dry_run.csv");

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
        "--dry-run",
    ]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("[drop_dataset_info]"))
        .stderr(predicates::str::contains("Lossy conversion"));

    assert!(!output_path.exists());
}

#[test]
fn convert_dry_run_directory_target_does_not_create_output_dir() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("dry_run_yolo");

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
        "--dry-run",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dry run: would convert"));

    assert!(!output_path.exists());
}

#[test]
fn convert_dry_run_blocked_json_still_emits_report_without_writing() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("dry_run.csv");

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
        "--dry-run",
        "--output-format",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(!output.status.success());
    assert!(!output_path.exists());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["from"], "coco");
    assert_eq!(parsed["to"], "tfod");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Lossy conversion"));
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
fn convert_ir_json_to_cvat_fails_without_allow_lossy() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let input_path = temp.path().join("in.ir.json");
    let output_path = temp.path().join("out.cvat.xml");

    let ir = r#"{
      "info": {"name": "needs-lossy-opt-in"},
      "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 80}],
      "categories": [{"id": 1, "name": "person"}],
      "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": {"xmin": 10.0, "ymin": 20.0, "xmax": 50.0, "ymax": 70.0}}]
    }"#;
    fs::write(&input_path, ir).expect("write input");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ir-json",
        "-t",
        "cvat",
        "-i",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Lossy conversion"))
        .stderr(predicates::str::contains("--allow-lossy"));
}

#[test]
fn convert_ir_json_to_cvat_succeeds_with_allow_lossy_and_report_has_policy_note() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let input_path = temp.path().join("in.ir.json");
    let output_path = temp.path().join("out.cvat.xml");

    let ir = r#"{
      "info": {"name": "needs-lossy-opt-in"},
      "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 80}],
      "categories": [{"id": 1, "name": "person"}],
      "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": {"xmin": 10.0, "ymin": 20.0, "xmax": 50.0, "ymax": 70.0}}]
    }"#;
    fs::write(&input_path, ir).expect("write input");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ir-json",
        "-t",
        "cvat",
        "-i",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--report",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["to"], "cvat");
    assert!(parsed["issues"].is_array());
    assert!(stdout.contains("cvat_writer_meta_defaults"));
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
fn convert_from_scaled_yolov4_txt_alias_works() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let yolo_dir = temp.path().join("sample_yolo");
    create_sample_yolo_dataset(&yolo_dir);
    let output_path = temp.path().join("scaled_alias_out.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "scaled-yolov4-txt",
        "-t",
        "coco",
        "-i",
        yolo_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"))
        .stdout(predicates::str::contains("(yolo)"));
}

#[test]
fn convert_hf_to_ir_json_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let hf_dir = temp.path().join("sample_hf");
    create_sample_hf_dataset(&hf_dir, false);
    let output_path = temp.path().join("hf_to_ir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "hf",
        "-t",
        "ir-json",
        "-i",
        hf_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(hf)"));
}

#[test]
fn convert_hf_xyxy_bbox_flag_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let hf_dir = temp.path().join("sample_hf_xyxy");
    create_sample_hf_dataset(&hf_dir, true);
    let output_path = temp.path().join("hf_xyxy_to_ir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "hf",
        "-t",
        "ir-json",
        "-i",
        hf_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
        "--hf-bbox-format",
        "xyxy",
    ]);
    cmd.assert().success();
}

#[test]
fn convert_hf_specific_flags_fail_for_non_hf_formats() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("bad_flags.json");

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
        "--hf-objects-column",
        "objects",
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("HF-specific flags"));
}

#[test]
fn convert_hf_repo_requires_hf_remote_feature() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("hf_repo.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "hf",
        "-t",
        "ir-json",
        "--hf-repo",
        "org/dataset",
        "-o",
        output_path.to_str().unwrap(),
    ]);

    #[cfg(feature = "hf-remote")]
    {
        // With hf-remote enabled this will attempt network access and fail on missing repo.
        cmd.assert().failure();
    }

    #[cfg(not(feature = "hf-remote"))]
    {
        cmd.assert()
            .failure()
            .stderr(predicates::str::contains("hf-remote"));
    }
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
fn convert_sagemaker_format_aliases_work() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_convert_sagemaker_alias.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "groundtruth",
        "-t",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.sagemaker.manifest",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));

    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_superannotate_format_aliases_work() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("superannotate_alias.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "sa",
        "-t",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.superannotate.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));
}

#[test]
fn convert_supervisely_format_aliases_work() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("supervisely_alias.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "sly",
        "-t",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.supervisely.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Converted"));
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
        "--output-format",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["from"], "coco");
    assert_eq!(parsed["to"], "ir-json");
    assert!(parsed.get("input").is_some());
    assert!(parsed.get("output").is_some());
    assert!(parsed.get("issues").is_some());
    assert!(!stdout.contains("Converted"));

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
        "--output-format",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let issues = parsed["issues"]
        .as_array()
        .expect("issues should be an array");
    assert!(issues.iter().any(|issue| issue["severity"] == "warning"));
    assert!(issues.iter().all(|issue| issue.get("code").is_some()));

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
    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let issues = parsed["issues"].as_array().expect("issues array");
    assert!(issues.iter().any(|issue| issue["severity"] == "info"));
    assert!(stdout.contains("tfod_reader_id_assignment"));

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

    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let issues = parsed["issues"].as_array().expect("issues array");
    assert!(issues.iter().any(|issue| issue["severity"] == "info"));
    assert!(stdout.contains("yolo_writer_float_precision"));
}

#[test]
fn convert_from_sagemaker_report_includes_reader_policy_notes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("report_sagemaker_source.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "sagemaker",
        "-t",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.sagemaker.manifest",
        "-o",
        output_path.to_str().unwrap(),
        "--report",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let issues = parsed["issues"].as_array().expect("issues array");
    assert!(issues.iter().any(|issue| issue["severity"] == "info"));
    assert!(stdout.contains("sagemaker_reader_id_assignment"));
    assert!(stdout.contains("sagemaker_reader_label_attribute_detection"));
}

#[test]
fn convert_to_sagemaker_report_includes_policy_notes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("report_sagemaker.manifest");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "sagemaker",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--report",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let issues = parsed["issues"].as_array().expect("issues array");
    assert!(issues.iter().any(|issue| issue["severity"] == "info"));
    assert!(stdout.contains("sagemaker_writer_class_map_policy"));
    assert!(stdout.contains("sagemaker_writer_metadata_defaults"));
}

#[test]
fn convert_to_sagemaker_blocks_lossy_without_allow_lossy() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("blocked_sagemaker.manifest");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "sagemaker",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--report",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(!output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let issues = parsed["issues"].as_array().expect("issues array");
    assert!(issues
        .iter()
        .any(|issue| issue["code"] == "drop_dataset_info"));
    assert!(issues.iter().any(|issue| issue["code"] == "drop_licenses"));
    assert!(issues
        .iter()
        .any(|issue| issue["code"] == "drop_category_supercategory"));
    assert!(stdout.contains("sagemaker_writer_class_map_policy"));
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

    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let issues = parsed["issues"].as_array().expect("issues array");
    assert!(issues.iter().any(|issue| issue["severity"] == "info"));
    assert!(stdout.contains("voc_writer_bool_normalization"));
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

    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let issues = parsed["issues"].as_array().expect("issues array");
    assert!(issues.iter().any(|issue| issue["severity"] == "info"));
    assert!(stdout.contains("label_studio_writer_from_to_defaults"));
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
        "--output-format",
        "json",
        "tests/fixtures/sample_valid.coco.json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert!(parsed.get("summary").is_some());
    assert!(parsed.get("labels").is_some());
    assert!(parsed.get("bboxes").is_some());
    assert!(parsed.get("cooccurrence_top_pairs").is_some());
}

#[test]
fn stats_text_output_is_plain_when_stdout_is_captured() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["stats", "tests/fixtures/sample_valid.coco.json"]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Dataset Stats Report"));
    assert!(stdout.contains("Summary\n-------"));
    assert!(stdout.contains("Bounding Boxes"));
    assert!(!stdout.contains('📊'));
    assert!(!stdout.contains('╭'));
    assert!(!stdout.contains('█'));
    assert!(!stdout.contains('…'));
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
        "--output-format",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert!(parsed.get("images").is_some());
    assert!(parsed.get("categories").is_some());
    assert!(parsed.get("annotations").is_some());
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
        .stderr(predicates::str::contains("--allow-lossy"))
        // Blocked sample path also emits the full report to stdout
        .stdout(predicates::str::contains("[drop_dataset_info]"))
        .stdout(predicates::str::contains("Warnings"));
}

#[test]
fn sample_json_output_format_emits_report_only() {
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
        "--output-format",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());
    assert!(out.is_file());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["from"], "coco");
    assert_eq!(parsed["to"], "ir-json");
    assert!(parsed.get("input").is_some());
    assert!(parsed.get("output").is_some());
    assert!(parsed.get("issues").is_some());
    assert!(!stdout.contains("Sampled"));
}

#[test]
fn sample_dry_run_text_does_not_overwrite_existing_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("existing.ir.json");
    fs::write(&out, "keep me").expect("seed output");

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
        "42",
        "--dry-run",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dry run: would sample"))
        .stdout(predicates::str::contains("images"))
        .stdout(predicates::str::contains("annotations"));

    assert_eq!(fs::read_to_string(&out).expect("read output"), "keep me");
}

#[test]
fn sample_dry_run_json_emits_report_only_and_skips_write() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("dry_run.ir.json");

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
        "42",
        "--dry-run",
        "--output-format",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());
    assert!(!out.exists());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["from"], "coco");
    assert_eq!(parsed["to"], "ir-json");
    assert!(parsed.get("issues").is_some());
    assert!(!stdout.contains("Sampled"));
    assert!(!stdout.contains("Dry run"));
}

#[test]
fn sample_dry_run_blocked_lossy_still_errors_without_writing() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("dry_run.csv");

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
        "--seed",
        "42",
        "--dry-run",
    ]);
    cmd.assert()
        .failure()
        .stdout(predicates::str::contains("[drop_dataset_info]"))
        .stderr(predicates::str::contains("Lossy conversion"));

    assert!(!out.exists());
}

#[test]
fn sample_dry_run_directory_target_does_not_create_output_dir() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("dry_run_yolo");

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
        "yolo",
        "-n",
        "1",
        "--seed",
        "42",
        "--allow-lossy",
        "--dry-run",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Dry run: would sample"));

    assert!(!out.exists());
}

#[test]
fn sample_dry_run_blocked_json_still_emits_report_without_writing() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("dry_run.csv");

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
        "--seed",
        "42",
        "--dry-run",
        "--output-format",
        "json",
    ]);
    let output = cmd.output().expect("run command");
    assert!(!output.status.success());
    assert!(!out.exists());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["from"], "coco");
    assert_eq!(parsed["to"], "tfod");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Lossy conversion"));
}

#[test]
fn sample_blocked_json_emits_valid_json_to_stdout() {
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
        "--output-format",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(!output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["from"], "coco");
    assert_eq!(parsed["to"], "tfod");
    assert!(parsed["issues"].is_array());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Lossy conversion"));
    assert!(stderr.contains("--allow-lossy"));
}

#[test]
fn sample_report_alias_json_works() {
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
        "--report",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert_eq!(parsed["to"], "ir-json");
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
        .stdout(predicates::str::contains("ibm-cloud-annotations"))
        .stdout(predicates::str::contains("cvat"))
        .stdout(predicates::str::contains("label-studio"))
        .stdout(predicates::str::contains("labelbox"))
        .stdout(predicates::str::contains("scale-ai"))
        .stdout(predicates::str::contains("unity-perception"))
        .stdout(predicates::str::contains("tfod"))
        .stdout(predicates::str::contains("vott-csv"))
        .stdout(predicates::str::contains("vott-json"))
        .stdout(predicates::str::contains("yolo"))
        .stdout(predicates::str::contains("voc"))
        .stdout(predicates::str::contains("hf"))
        .stdout(predicates::str::contains("sagemaker"))
        .stdout(predicates::str::contains("superannotate"))
        .stdout(predicates::str::contains("supervisely"))
        .stdout(predicates::str::contains("cityscapes"))
        .stdout(predicates::str::contains("marmot"))
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

#[test]
fn list_formats_json_output_has_expected_schema() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["list-formats", "--output", "json"]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let formats = parsed.as_array().expect("top-level array");
    assert_eq!(formats.len(), 30);

    let label_studio = formats
        .iter()
        .find(|entry| entry["name"] == "label-studio")
        .expect("label-studio entry");
    let label_studio_aliases = label_studio["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(label_studio_aliases.contains(&"label-studio-json"));
    assert!(label_studio_aliases.contains(&"ls"));

    let labelbox = formats
        .iter()
        .find(|entry| entry["name"] == "labelbox")
        .expect("labelbox entry");
    let labelbox_aliases = labelbox["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(labelbox_aliases.contains(&"labelbox-json"));
    assert!(labelbox_aliases.contains(&"labelbox-ndjson"));
    assert_eq!(labelbox["file_based"], true);
    assert_eq!(labelbox["directory_based"], false);
    assert_eq!(labelbox["lossiness"], "lossy");

    let scale_ai = formats
        .iter()
        .find(|entry| entry["name"] == "scale-ai")
        .expect("scale-ai entry");
    let scale_ai_aliases = scale_ai["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(scale_ai_aliases.contains(&"scale"));
    assert!(scale_ai_aliases.contains(&"scale-ai-json"));
    assert_eq!(scale_ai["file_based"], true);
    assert_eq!(scale_ai["directory_based"], true);
    assert_eq!(scale_ai["lossiness"], "lossy");

    let unity = formats
        .iter()
        .find(|entry| entry["name"] == "unity-perception")
        .expect("unity-perception entry");
    let unity_aliases = unity["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(unity_aliases.contains(&"unity"));
    assert!(unity_aliases.contains(&"unity-perception-json"));
    assert!(unity_aliases.contains(&"solo"));
    assert_eq!(unity["file_based"], true);
    assert_eq!(unity["directory_based"], true);
    assert_eq!(unity["lossiness"], "lossy");

    let yolo = formats
        .iter()
        .find(|entry| entry["name"] == "yolo")
        .expect("yolo entry");
    let yolo_aliases = yolo["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(yolo_aliases.contains(&"scaled-yolov4"));
    assert!(yolo_aliases.contains(&"scaled-yolov4-txt"));
    assert_eq!(yolo["directory_based"], true);
    assert_eq!(yolo["file_based"], false);

    let yolo_keras = formats
        .iter()
        .find(|entry| entry["name"] == "yolo-keras")
        .expect("yolo-keras entry");
    let yolo_keras_aliases = yolo_keras["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(yolo_keras_aliases.contains(&"yolo-keras-txt"));
    assert_eq!(yolo_keras["directory_based"], true);
    assert_eq!(yolo_keras["file_based"], true);
    assert_eq!(yolo_keras["lossiness"], "lossy");

    let yolov4_pytorch = formats
        .iter()
        .find(|entry| entry["name"] == "yolov4-pytorch")
        .expect("yolov4-pytorch entry");
    let yolov4_pytorch_aliases = yolov4_pytorch["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(yolov4_pytorch_aliases.contains(&"yolov4-pytorch-txt"));
    assert_eq!(yolov4_pytorch["directory_based"], true);
    assert_eq!(yolov4_pytorch["file_based"], true);
    assert_eq!(yolov4_pytorch["lossiness"], "lossy");

    let sagemaker = formats
        .iter()
        .find(|entry| entry["name"] == "sagemaker")
        .expect("sagemaker entry");
    let sagemaker_aliases = sagemaker["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(sagemaker_aliases.contains(&"sagemaker-manifest"));
    assert!(sagemaker_aliases.contains(&"groundtruth"));
    assert_eq!(sagemaker["file_based"], true);
    assert_eq!(sagemaker["directory_based"], false);
    assert_eq!(sagemaker["lossiness"], "lossy");

    let superannotate = formats
        .iter()
        .find(|entry| entry["name"] == "superannotate")
        .expect("superannotate entry");
    let superannotate_aliases = superannotate["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(superannotate_aliases.contains(&"superannotate-json"));
    assert!(superannotate_aliases.contains(&"sa"));
    assert_eq!(superannotate["file_based"], true);
    assert_eq!(superannotate["directory_based"], true);
    assert_eq!(superannotate["lossiness"], "lossy");

    let supervisely = formats
        .iter()
        .find(|entry| entry["name"] == "supervisely")
        .expect("supervisely entry");
    let supervisely_aliases = supervisely["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(supervisely_aliases.contains(&"supervisely-json"));
    assert!(supervisely_aliases.contains(&"sly"));
    assert_eq!(supervisely["file_based"], true);
    assert_eq!(supervisely["directory_based"], true);
    assert_eq!(supervisely["lossiness"], "lossy");

    let cityscapes = formats
        .iter()
        .find(|entry| entry["name"] == "cityscapes")
        .expect("cityscapes entry");
    let cityscapes_aliases = cityscapes["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(cityscapes_aliases.contains(&"cityscapes-json"));
    assert_eq!(cityscapes["file_based"], true);
    assert_eq!(cityscapes["directory_based"], true);
    assert_eq!(cityscapes["lossiness"], "lossy");

    let marmot = formats
        .iter()
        .find(|entry| entry["name"] == "marmot")
        .expect("marmot entry");
    let marmot_aliases = marmot["aliases"]
        .as_array()
        .expect("aliases array")
        .iter()
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>();
    assert!(marmot_aliases.contains(&"marmot-xml"));
    assert_eq!(marmot["file_based"], true);
    assert_eq!(marmot["directory_based"], true);
    assert_eq!(marmot["lossiness"], "lossy");

    let coco = formats
        .iter()
        .find(|entry| entry["name"] == "coco")
        .expect("coco entry");
    assert_eq!(coco["file_based"], true);
    assert_eq!(coco["directory_based"], false);
    assert_eq!(coco["lossiness"], "conditional");

    let ir_json = formats
        .iter()
        .find(|entry| entry["name"] == "ir-json")
        .expect("ir-json entry");
    assert_eq!(ir_json["lossiness"], "lossless");
    assert_eq!(ir_json["read"], true);
    assert_eq!(ir_json["write"], true);

    for name in [
        "tfod",
        "vott-csv",
        "vott-json",
        "ibm-cloud-annotations",
        "labelbox",
        "scale-ai",
        "unity-perception",
        "yolo",
        "yolo-keras",
        "yolov4-pytorch",
        "voc",
        "hf",
        "sagemaker",
        "superannotate",
        "supervisely",
        "cityscapes",
        "marmot",
    ] {
        let entry = formats
            .iter()
            .find(|format| format["name"] == name)
            .unwrap_or_else(|| panic!("missing {name} entry"));
        assert_eq!(entry["lossiness"], "lossy");
    }
}

#[test]
fn list_formats_output_format_alias_works() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["list-formats", "--output-format", "json"]);
    let output = cmd.output().expect("run command");
    assert!(output.status.success());

    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    let formats = parsed.as_array().expect("top-level array");
    assert!(formats.iter().any(|entry| entry["name"] == "coco"));
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
fn convert_auto_detects_vott_csv_format() {
    let temp = tempfile::tempdir().expect("tempdir");
    write_bmp(&temp.path().join("img1.bmp"), 100, 80);
    let csv_path = temp.path().join("annotations.csv");
    fs::write(
        &csv_path,
        "image,xmin,ymin,xmax,ymax,label\nimg1.bmp,10,20,50,70,person\n",
    )
    .expect("write vott csv");
    let output_path = temp.path().join("auto_detect_vott.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        csv_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(vott-csv)"));
}

#[test]
fn convert_auto_detects_vott_json_file() {
    let temp = tempfile::tempdir().expect("tempdir");
    create_sample_vott_json_dataset(temp.path());
    let json_path = temp.path().join("vott-json-export/panlabel-export.json");
    let output_path = temp.path().join("auto_detect_vott_json_file.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        json_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(vott-json)"));
}

#[test]
fn convert_auto_detects_vott_json_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    create_sample_vott_json_dataset(temp.path());
    let output_path = temp.path().join("auto_detect_vott_json_dir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(vott-json)"));
}

#[test]
fn convert_auto_detects_cloud_annotations_file() {
    let temp = tempfile::tempdir().expect("tempdir");
    create_sample_cloud_annotations_dataset(temp.path());
    let output_path = temp.path().join("auto_detect_cloud_annotations.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().join("_annotations.json").to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(ibm-cloud-annotations)"));
}

#[test]
fn convert_auto_detects_cloud_annotations_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    create_sample_cloud_annotations_dataset(temp.path());
    let output_path = temp.path().join("auto_detect_cloud_annotations_dir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(ibm-cloud-annotations)"));
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
fn convert_auto_detects_superannotate_json_format() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("auto_detect_superannotate.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.superannotate.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(superannotate)"));
}

#[test]
fn convert_auto_detects_supervisely_json_format() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("auto_detect_supervisely.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.supervisely.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(supervisely)"));
}

#[test]
fn convert_auto_detects_cityscapes_json_format() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("auto_detect_cityscapes.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.cityscapes.json",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(cityscapes)"));
}

#[test]
fn convert_auto_detects_marmot_xml_file() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_marmot_dataset(temp.path());
    let output_path = temp.path().join("auto_detect_marmot.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().join("page1.xml").to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(marmot)"));
}

#[test]
fn convert_auto_detects_marmot_xml_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_marmot_dataset(temp.path());
    let output_path = temp.path().join("auto_detect_marmot_dir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(marmot)"));
}

#[test]
fn convert_auto_detects_sagemaker_manifest_format() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("auto_detect_sagemaker.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.sagemaker.manifest",
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(sagemaker)"));

    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn convert_auto_detects_sagemaker_jsonl_format() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let jsonl_path = temp.path().join("sample.jsonl");
    fs::copy(
        "tests/fixtures/sample_valid.sagemaker.manifest",
        &jsonl_path,
    )
    .expect("copy manifest fixture to jsonl");
    let output_path = temp.path().join("auto_detect_sagemaker_jsonl.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        jsonl_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(sagemaker)"));
}

#[test]
fn convert_auto_detects_labelbox_jsonl_before_sagemaker() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let jsonl_path = temp.path().join("sample.jsonl");
    create_sample_labelbox_jsonl(&jsonl_path);
    let output_path = temp.path().join("auto_detect_labelbox_jsonl.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        jsonl_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(labelbox)"));
}

#[test]
fn convert_auto_detects_scale_ai_json_file() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("auto_detect_scale_ai_file.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.scale_ai.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(scale-ai)"));
}

#[test]
fn convert_auto_detects_scale_ai_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_scale_ai_dataset(temp.path());
    let output_path = temp.path().join("auto_detect_scale_ai_dir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(scale-ai)"));
}

#[test]
fn convert_to_scale_ai_report_includes_policy_notes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("report_scale_ai.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "scale-ai",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--output-format",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());
    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert!(parsed["issues"].as_array().unwrap().iter().any(|issue| {
        issue["code"] == "scale_ai_writer_deterministic_order"
            || issue["code"] == "scale_ai_writer_rectangle_policy"
    }));
    assert!(output_path.is_file());
}

#[test]
fn convert_auto_detects_unity_perception_json_file() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("auto_detect_unity_file.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        "tests/fixtures/sample_valid.unity_perception.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(unity-perception)"));
}

#[test]
fn convert_auto_detects_unity_perception_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_unity_perception_dataset(temp.path());
    let output_path = temp.path().join("auto_detect_unity_dir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(unity-perception)"));
}

#[test]
fn convert_to_unity_perception_report_includes_policy_notes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("report_unity_perception");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "unity-perception",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--output-format",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());
    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert!(parsed["issues"].as_array().unwrap().iter().any(|issue| {
        issue["code"] == "unity_perception_writer_directory_layout"
            || issue["code"] == "unity_perception_writer_rectangle_policy"
    }));
    assert!(output_path
        .join("sequence.0/step0.frame_data.json")
        .is_file());
}

#[test]
fn convert_to_labelbox_report_includes_policy_notes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("report_labelbox.ndjson");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "labelbox",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--output-format",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());
    let (stdout, parsed) = stdout_json(&output);
    assert_compact_json(&stdout);
    assert!(parsed["issues"].as_array().unwrap().iter().any(|issue| {
        issue["code"] == "labelbox_writer_format_policy"
            || issue["code"] == "labelbox_writer_rectangle_policy"
    }));
    assert!(output_path.is_file());
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
fn convert_auto_detects_yolo_with_unrelated_cityscapes_like_json_as_yolo() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let yolo_dir = temp.path().join("sample_yolo");
    create_sample_yolo_dataset(&yolo_dir);
    fs::write(
        yolo_dir.join("notes.json"),
        r#"{"imgWidth": 10, "imgHeight": 10, "objects": []}"#,
    )
    .expect("write unrelated json");
    let output_path = temp.path().join("auto_detect_yolo_with_notes.json");

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
fn convert_auto_detects_yolo_keras_named_txt_file() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let txt_path = temp.path().join("yolo_keras.txt");
    fs::write(&txt_path, "img.bmp 1,2,10,20,0\n").expect("write annotations");
    fs::write(temp.path().join("classes.txt"), "object\n").expect("write classes");
    write_bmp(&temp.path().join("img.bmp"), 30, 30);
    let output_path = temp.path().join("auto_detect_yolo_keras.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        txt_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(yolo-keras)"));
}

#[test]
fn convert_auto_detect_errors_on_generic_train_txt_ambiguity() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let txt_path = temp.path().join("train.txt");
    fs::write(&txt_path, "img.bmp 1,2,10,20,0\n").expect("write annotations");
    write_bmp(&temp.path().join("img.bmp"), 30, 30);
    let output_path = temp.path().join("ambiguous_train.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        txt_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert().failure().stderr(predicates::str::contains(
        "matches both yolo-keras and yolov4-pytorch",
    ));
}

#[test]
fn convert_to_yolov4_pytorch_report_includes_policy_notes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_path = temp.path().join("yolov4_pytorch_out");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "coco",
        "-t",
        "yolov4-pytorch",
        "-i",
        "tests/fixtures/sample_valid.coco.json",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
        "--output-format",
        "json",
    ]);

    let output = cmd.output().expect("run command");
    assert!(output.status.success());
    let (_stdout, parsed) = stdout_json(&output);
    assert!(parsed["issues"].as_array().unwrap().iter().any(|issue| {
        issue["code"] == "yolo_keras_txt_writer_class_order"
            || issue["code"] == "yolo_keras_txt_writer_deterministic_order"
    }));
    assert!(output_path.join("yolov4_pytorch.txt").is_file());
    assert!(output_path.join("classes.txt").is_file());
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
fn convert_auto_detects_cvat_xml_file() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output_path = temp.path().join("auto_detect_cvat_file.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        "tests/fixtures/sample_valid.cvat.xml",
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(cvat)"));
}

#[test]
fn convert_auto_detects_cvat_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    let cvat_dir = temp.path().join("sample_cvat");
    create_sample_cvat_export(&cvat_dir);
    let output_path = temp.path().join("auto_detect_cvat_dir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "coco",
        "-i",
        cvat_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(cvat)"));
}

#[test]
fn convert_auto_detects_hf_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    let hf_dir = temp.path().join("sample_hf");
    create_sample_hf_dataset(&hf_dir, false);
    let output_path = temp.path().join("auto_detect_hf.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        hf_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(hf)"));
}

#[test]
fn convert_auto_detects_superannotate_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let source_dir = temp.path().join("sample_superannotate");
    fs::create_dir_all(source_dir.join("annotations")).expect("create annotations dir");
    fs::copy(
        "tests/fixtures/sample_valid.superannotate.json",
        source_dir.join("annotations/sa_image.json"),
    )
    .expect("copy fixture");
    let output_path = temp.path().join("auto_detect_superannotate_dir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        source_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(superannotate)"));
}

#[test]
fn convert_auto_does_not_detect_superannotate_from_nested_child_only() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let nested_dir = temp.path().join("archive/export/annotations");
    fs::create_dir_all(&nested_dir).expect("create nested annotations dir");
    fs::copy(
        "tests/fixtures/sample_valid.superannotate.json",
        nested_dir.join("sa_image.json"),
    )
    .expect("copy fixture");
    let output_path = temp
        .path()
        .join("should_not_detect_nested_superannotate.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("unrecognized directory layout"));
}

#[test]
fn convert_auto_detects_supervisely_project_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let project_dir = temp.path().join("sample_supervisely");
    fs::create_dir_all(project_dir.join("dataset_01/ann")).expect("create ann dir");
    fs::write(
        project_dir.join("meta.json"),
        r#"{"classes": [{"title": "cat"}]}"#,
    )
    .expect("write meta");
    fs::copy(
        "tests/fixtures/sample_valid.supervisely.json",
        project_dir.join("dataset_01/ann/sample.jpg.json"),
    )
    .expect("copy fixture");
    let output_path = temp.path().join("auto_detect_supervisely_project.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        project_dir.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(supervisely)"));
}

#[test]
fn convert_auto_detects_cityscapes_dataset_root() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ann_dir = temp.path().join("gtFine/train/aachen");
    fs::create_dir_all(&ann_dir).expect("create cityscapes ann dir");
    fs::copy(
        "tests/fixtures/sample_valid.cityscapes.json",
        ann_dir.join("aachen_000001_000019_gtFine_polygons.json"),
    )
    .expect("copy fixture");
    let output_path = temp.path().join("auto_detect_cityscapes_dir.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(cityscapes)"));
}

#[test]
fn convert_auto_detects_yolo_split_with_train_txt_image_list_as_yolo() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("images/train")).expect("create images");
    fs::create_dir_all(temp.path().join("labels/train")).expect("create labels");
    write_bmp(&temp.path().join("images/train/img.bmp"), 20, 10);
    fs::write(
        temp.path().join("labels/train/img.txt"),
        "0 0.5 0.5 0.5 0.5\n",
    )
    .expect("write labels");
    fs::write(temp.path().join("train.txt"), "images/train/img.bmp\n").expect("write image list");
    fs::write(
        temp.path().join("data.yaml"),
        "names:\n  - person\ntrain: train.txt\n",
    )
    .expect("write data yaml");
    let output_path = temp.path().join("auto_detect_yolo_split.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(yolo)"));
}

#[test]
fn convert_auto_detect_errors_on_hf_yolo_ambiguity() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_hf_dataset(temp.path(), false);
    // Need both labels/ with .txt AND images/ for YOLO to be a complete match.
    fs::create_dir_all(temp.path().join("labels")).expect("create labels dir");
    fs::create_dir_all(temp.path().join("images")).expect("create images dir");
    fs::write(temp.path().join("labels/img1.txt"), "0 0.5 0.5 0.2 0.2\n").expect("write label");

    let output_path = temp.path().join("auto_detect_ambiguous_hf_yolo.json");

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
        .stderr(predicates::str::contains("matches both YOLO and HF"));
}

#[test]
fn convert_auto_detect_partial_yolo_without_images_detects_hf() {
    // labels/ without images/ is a *partial* YOLO layout.
    // If HF markers are also present, HF should win (only complete match).
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_hf_dataset(temp.path(), false);
    fs::create_dir_all(temp.path().join("labels")).expect("create labels dir");
    fs::write(temp.path().join("labels/img1.txt"), "0 0.5 0.5 0.2 0.2\n").expect("write label");

    let output_path = temp.path().join("partial_yolo_hf.json");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(hf)"));
}

#[test]
fn convert_auto_detect_partial_yolo_gives_helpful_error() {
    // labels/ without images/ and no other complete format should report
    // the partial match with guidance.
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("labels")).expect("create labels dir");
    fs::write(temp.path().join("labels/img1.txt"), "0 0.5 0.5 0.2 0.2\n").expect("write label");

    let output_path = temp.path().join("partial_yolo.json");

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
        .stderr(predicates::str::contains("YOLO"))
        .stderr(predicates::str::contains("images/ directory"));
}

#[test]
fn convert_auto_detect_partial_marmot_gives_helpful_error() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_marmot_dataset(temp.path());
    fs::remove_file(temp.path().join("page1.bmp")).expect("remove companion image");

    let output_path = temp.path().join("partial_marmot.json");
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "--from",
        "auto",
        "--to",
        "ir-json",
        "-i",
        temp.path().to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Marmot"))
        .stderr(predicates::str::contains("same-stem companion image"));
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

#[test]
fn convert_auto_detects_voc_without_jpegimages() {
    // VOC reader accepts Annotations/ without JPEGImages/, so auto-detect should too.
    let temp = tempfile::tempdir().expect("create temp dir");
    let voc_dir = temp.path().join("voc_no_images");
    fs::create_dir_all(voc_dir.join("Annotations")).expect("create annotations dir");
    // No JPEGImages/ directory — reader treats it as optional.

    let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>img1.jpg</filename>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <object>
    <name>person</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>70</ymax></bndbox>
  </object>
</annotation>
"#;
    fs::write(voc_dir.join("Annotations/img1.xml"), xml).expect("write xml");

    let output_path = temp.path().join("voc_no_images.json");

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
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("(voc)"));
}

#[test]
fn stats_malformed_json_does_not_fall_back_to_ir() {
    // Malformed JSON should surface the parse error directly,
    // not be silently retried as IR JSON.
    let temp = tempfile::tempdir().expect("tempdir");
    let p = temp.path().join("bad.json");
    fs::write(&p, "{ this is not valid json }").expect("write");

    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["stats", p.to_str().unwrap()]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("parse JSON"));
}

#[test]
fn stats_ambiguous_json_falls_back_to_ir() {
    // Valid JSON that detection can't classify should still fall back
    // to IR JSON for stats (the narrow fallback).
    let temp = tempfile::tempdir().expect("tempdir");
    let p = temp.path().join("empty_annotations.json");
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
fn convert_auto_detect_unrecognized_dir_lists_expected_layouts() {
    // Empty directory should give a helpful error listing expected layouts.
    let temp = tempfile::tempdir().expect("tempdir");

    let output_path = temp.path().join("output.json");
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
        .stderr(predicates::str::contains("unrecognized directory layout"))
        .stderr(predicates::str::contains("YOLO"))
        .stderr(predicates::str::contains("VOC"))
        .stderr(predicates::str::contains("CVAT"))
        .stderr(predicates::str::contains("HF"));
}

#[test]
fn convert_auto_detect_ambiguity_shows_evidence() {
    // Ambiguity messages should include what markers were found.
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_hf_dataset(temp.path(), false);
    fs::create_dir_all(temp.path().join("labels")).expect("create labels dir");
    fs::create_dir_all(temp.path().join("images")).expect("create images dir");
    fs::write(temp.path().join("labels/img1.txt"), "0 0.5 0.5 0.2 0.2\n").expect("write label");

    let output_path = temp.path().join("evidence.json");

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
        // Evidence should be listed.
        .stderr(predicates::str::contains("labels/"))
        .stderr(predicates::str::contains("metadata"));
}
