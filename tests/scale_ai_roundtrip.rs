//! Integration tests for Scale AI JSON support.

mod common;

use common::write_bmp;
use panlabel::ir::io_scale_ai_json::{
    from_scale_ai_json_str, read_scale_ai_json, to_scale_ai_json_string, write_scale_ai_json,
    ATTR_ENVELOPED, ATTR_POLYGON_ENVELOPES, ATTR_ROTATED_BOX_ENVELOPES, ATTR_UUID,
};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};
use serde_json::Value;
use std::path::Path;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {actual} to be close to {expected}"
    );
}

#[test]
fn reads_fixture_with_box_polygon_and_rotated_box_envelopes() {
    let dataset = read_scale_ai_json(Path::new("tests/fixtures/sample_valid.scale_ai.json"))
        .expect("read Scale AI fixture");

    assert_eq!(dataset.images.len(), 2);
    assert_eq!(dataset.categories.len(), 3);
    assert_eq!(dataset.annotations.len(), 3);
    assert_eq!(dataset.images[0].file_name, "img1.jpg");
    assert_eq!(dataset.images[1].file_name, "img2.jpg");
    assert_eq!(
        dataset.info.attributes.get(ATTR_POLYGON_ENVELOPES),
        Some(&"1".to_string())
    );
    assert_eq!(
        dataset.info.attributes.get(ATTR_ROTATED_BOX_ENVELOPES),
        Some(&"1".to_string())
    );

    let bbox_ann = dataset
        .annotations
        .iter()
        .find(|ann| ann.attributes.get(ATTR_UUID).map(String::as_str) == Some("box-1"))
        .expect("box annotation");
    assert_close(bbox_ann.bbox.xmin(), 10.0);
    assert_close(bbox_ann.bbox.ymin(), 20.0);
    assert_close(bbox_ann.bbox.xmax(), 40.0);
    assert_close(bbox_ann.bbox.ymax(), 60.0);

    let polygon_ann = dataset
        .annotations
        .iter()
        .find(|ann| ann.attributes.get(ATTR_UUID).map(String::as_str) == Some("poly-1"))
        .expect("polygon annotation");
    assert_eq!(
        polygon_ann
            .attributes
            .get(ATTR_ENVELOPED)
            .map(String::as_str),
        Some("true")
    );
    assert_close(polygon_ann.bbox.xmin(), 5.0);
    assert_close(polygon_ann.bbox.ymin(), 4.0);
    assert_close(polygon_ann.bbox.xmax(), 30.0);
    assert_close(polygon_ann.bbox.ymax(), 40.0);
}

#[test]
fn reads_response_object_without_task_using_geometry_extents_for_dimensions() {
    let json = r#"{
        "annotations": [
            {"type": "box", "label": "cat", "left": 2, "top": 3, "width": 8, "height": 9}
        ]
    }"#;

    let dataset = from_scale_ai_json_str(json).expect("read response-only JSON");
    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].width, 10);
    assert_eq!(dataset.images[0].height, 12);
    assert_eq!(dataset.annotations.len(), 1);
}

#[test]
fn write_then_read_file_roundtrip_preserves_detection_semantics() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("scale-ai.json");
    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "img1.jpg", 100, 80),
            Image::new(2u64, "img2.jpg", 120, 90),
        ],
        categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
        annotations: vec![
            Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 50.0, 60.0),
            ),
            Annotation::new(
                2u64,
                2u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(5.0, 4.0, 30.0, 40.0),
            ),
        ],
        ..Default::default()
    };

    write_scale_ai_json(&json_path, &dataset).expect("write Scale AI JSON");
    let raw = std::fs::read_to_string(&json_path).expect("read output JSON");
    let value: Value = serde_json::from_str(&raw).expect("parse output JSON");
    assert!(value.is_array());
    assert_eq!(value[0]["type"], "imageannotation");
    assert_eq!(value[0]["response"]["annotations"][0]["type"], "box");

    let restored = read_scale_ai_json(&json_path).expect("read written JSON");
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
    assert_close(restored.annotations[0].bbox.xmin(), 10.0);
    assert_close(restored.annotations[0].bbox.ymax(), 60.0);
}

#[test]
fn directory_input_accepts_annotations_subdir_and_probes_local_image_dimensions() {
    let temp = tempfile::tempdir().expect("create temp dir");
    std::fs::create_dir_all(temp.path().join("annotations")).expect("create annotations dir");
    write_bmp(&temp.path().join("images/img.bmp"), 320, 240);
    std::fs::write(
        temp.path().join("annotations/img.json"),
        r#"{
            "task_id":"task-img",
            "params":{"attachment":"img.bmp"},
            "response":{"annotations":[{"type":"box","label":"obj","left":1,"top":2,"width":30,"height":40}]}
        }"#,
    )
    .expect("write annotation JSON");

    let dataset = read_scale_ai_json(temp.path()).expect("read directory");
    assert_eq!(dataset.images[0].width, 320);
    assert_eq!(dataset.images[0].height, 240);
    assert_eq!(dataset.annotations.len(), 1);
}

#[test]
fn directory_output_writes_annotation_files_and_image_placeholder() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_dir = temp.path().join("out");
    let dataset = Dataset {
        images: vec![Image::new(1u64, "img.jpg", 100, 80)],
        categories: vec![Category::new(1u64, "person")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 50.0, 70.0),
        )],
        ..Default::default()
    };

    write_scale_ai_json(&output_dir, &dataset).expect("write directory");
    assert!(output_dir.join("annotations/img.json").is_file());
    assert!(output_dir.join("images/README.txt").is_file());
    let restored = read_scale_ai_json(&output_dir).expect("read directory");
    assert_eq!(restored.annotations.len(), 1);
}

#[test]
fn rejects_unsupported_geometry_clearly() {
    let err = from_scale_ai_json_str(
        r#"{"annotations":[{"type":"line","label":"lane","vertices":[{"x":1,"y":2},{"x":3,"y":4}]}]}"#,
    )
    .expect_err("unsupported geometry should fail");
    assert!(err
        .to_string()
        .contains("unsupported Scale AI geometry type 'line'"));
}

#[test]
fn to_string_writes_single_task_object_for_single_image_dataset() {
    let dataset = Dataset {
        images: vec![Image::new(1u64, "img.jpg", 100, 80)],
        categories: vec![Category::new(1u64, "person")],
        annotations: vec![],
        ..Default::default()
    };
    let json = to_scale_ai_json_string(&dataset).expect("serialize");
    let value: Value = serde_json::from_str(&json).expect("parse serialized JSON");
    assert!(value.is_object());
    assert_eq!(value["params"]["metadata"]["file_name"], "img.jpg");
}
