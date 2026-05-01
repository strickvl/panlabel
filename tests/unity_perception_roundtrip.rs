//! Integration tests for Unity Perception / SOLO JSON support.

mod common;

use common::write_bmp;
use panlabel::ir::io_unity_perception_json::{
    from_unity_perception_json_str, read_unity_perception_json, write_unity_perception_json,
    ATTR_INSTANCE_ID, ATTR_SKIPPED_ANNOTATIONS,
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
fn reads_fixture_with_bbox_values_and_skipped_non_bbox_annotation() {
    let dataset = read_unity_perception_json(Path::new(
        "tests/fixtures/sample_valid.unity_perception.json",
    ))
    .expect("read Unity fixture");

    assert_eq!(dataset.images.len(), 2);
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.annotations.len(), 2);
    assert_eq!(dataset.images[0].file_name, "img1.png");
    assert_eq!(dataset.images[0].width, 100);
    assert_eq!(dataset.images[1].file_name, "img2.png");
    assert_eq!(
        dataset.info.attributes.get(ATTR_SKIPPED_ANNOTATIONS),
        Some(&"1".to_string())
    );

    let first = dataset
        .annotations
        .iter()
        .find(|ann| ann.attributes.get(ATTR_INSTANCE_ID).map(String::as_str) == Some("10"))
        .expect("first bbox");
    assert_close(first.bbox.xmin(), 10.0);
    assert_close(first.bbox.ymin(), 20.0);
    assert_close(first.bbox.xmax(), 50.0);
    assert_close(first.bbox.ymax(), 50.0);

    let origin_dimension = dataset
        .annotations
        .iter()
        .find(|ann| ann.attributes.get(ATTR_INSTANCE_ID).map(String::as_str) == Some("crate-1"))
        .expect("origin/dimension bbox");
    assert_close(origin_dimension.bbox.xmin(), 5.0);
    assert_close(origin_dimension.bbox.ymin(), 4.0);
    assert_close(origin_dimension.bbox.xmax(), 30.0);
    assert_close(origin_dimension.bbox.ymax(), 40.0);
}

#[test]
fn reads_narrow_legacy_captures_json() {
    let json = r#"{
        "version": "0.0.1",
        "captures": [{
            "id": "capture-1",
            "sequence_id": "seq-1",
            "step": 7,
            "timestamp": 12.5,
            "filename": "legacy.png",
            "format": "png",
            "annotations": [{
                "id": "bbox-def",
                "annotation_definition": "bbox-def",
                "values": [{
                    "label_id": 3,
                    "label_name": "box",
                    "instance_id": "inst-1",
                    "x": 2,
                    "y": 3,
                    "width": 8,
                    "height": 9
                }]
            }]
        }]
    }"#;

    let dataset = from_unity_perception_json_str(json).expect("read legacy captures JSON");
    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "legacy.png");
    assert_eq!(dataset.images[0].width, 10);
    assert_eq!(dataset.images[0].height, 12);
    assert_eq!(dataset.annotations.len(), 1);
}

#[test]
fn directory_input_reads_solo_frames_and_annotation_definitions() {
    let temp = tempfile::tempdir().expect("create temp dir");
    std::fs::create_dir_all(temp.path().join("sequence.0")).expect("create sequence dir");
    write_bmp(&temp.path().join("images/img.bmp"), 320, 240);
    std::fs::write(
        temp.path().join("annotation_definitions.json"),
        r#"{
            "annotationDefinitions": [{
                "@type": "type.unity.com/unity.solo.BoundingBox2DAnnotationDefinition",
                "id": "bbox-def",
                "spec": [{"label_id": 5, "label_name": "mapped"}]
            }]
        }"#,
    )
    .expect("write definitions");
    std::fs::write(
        temp.path().join("sequence.0/step0.frame_data.json"),
        r#"{
            "frame": 0,
            "sequence": 0,
            "step": 0,
            "captures": [{
                "id": "camera",
                "filename": "img.bmp",
                "annotations": [{
                    "@type": "type.unity.com/unity.solo.BoundingBox2DAnnotation",
                    "id": "bbox-def",
                    "values": [{"label_id": 5, "instance_id": 1, "x": 1, "y": 2, "width": 30, "height": 40}]
                }]
            }]
        }"#,
    )
    .expect("write frame");

    let dataset = read_unity_perception_json(temp.path()).expect("read directory");
    assert_eq!(dataset.images[0].width, 320);
    assert_eq!(dataset.images[0].height, 240);
    assert_eq!(dataset.categories[0].name, "mapped");
    assert_eq!(dataset.annotations.len(), 1);
}

#[test]
fn directory_output_writes_minimal_solo_layout_and_roundtrips() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_dir = temp.path().join("unity-out");
    let dataset = Dataset {
        images: vec![Image::new(1u64, "img.png", 100, 80)],
        categories: vec![Category::new(1u64, "person")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 50.0, 70.0),
        )],
        ..Default::default()
    };

    write_unity_perception_json(&output_dir, &dataset).expect("write directory");
    assert!(output_dir.join("annotation_definitions.json").is_file());
    assert!(output_dir
        .join("sequence.0/step0.frame_data.json")
        .is_file());
    assert!(output_dir.join("images/README.txt").is_file());

    let raw = std::fs::read_to_string(output_dir.join("sequence.0/step0.frame_data.json"))
        .expect("read frame");
    let value: Value = serde_json::from_str(&raw).expect("parse frame");
    assert_eq!(
        value["captures"][0]["annotations"][0]["@type"],
        "type.unity.com/unity.solo.BoundingBox2DAnnotation"
    );

    let restored = read_unity_perception_json(&output_dir).expect("read output directory");
    assert_eq!(restored.images.len(), 1);
    assert_eq!(restored.annotations.len(), 1);
    assert_close(restored.annotations[0].bbox.xmin(), 10.0);
    assert_close(restored.annotations[0].bbox.ymax(), 70.0);
}

#[test]
fn empty_bbox_values_are_not_reported_as_skipped_annotations() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_dir = temp.path().join("unity-empty");
    let dataset = Dataset {
        images: vec![Image::new(1u64, "empty.png", 100, 80)],
        categories: vec![Category::new(1u64, "person")],
        annotations: vec![],
        ..Default::default()
    };

    write_unity_perception_json(&output_dir, &dataset).expect("write directory");
    let restored = read_unity_perception_json(&output_dir).expect("read output directory");
    assert_eq!(restored.images.len(), 1);
    assert_eq!(restored.annotations.len(), 0);
    assert!(!restored
        .info
        .attributes
        .contains_key(ATTR_SKIPPED_ANNOTATIONS));
}

#[test]
fn writer_removes_stale_panlabel_frame_files() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_dir = temp.path().join("unity-stale");
    std::fs::create_dir_all(output_dir.join("sequence.0")).expect("create sequence dir");
    std::fs::write(
        output_dir.join("sequence.0/step99.frame_data.json"),
        r#"{"captures":[{"filename":"stale.png","dimension":[1,1],"annotations":[]}]}"#,
    )
    .expect("write stale frame");

    let dataset = Dataset {
        images: vec![Image::new(1u64, "fresh.png", 100, 80)],
        categories: vec![Category::new(1u64, "person")],
        annotations: vec![],
        ..Default::default()
    };

    write_unity_perception_json(&output_dir, &dataset).expect("write directory");
    assert!(!output_dir
        .join("sequence.0/step99.frame_data.json")
        .exists());
    let restored = read_unity_perception_json(&output_dir).expect("read output directory");
    assert_eq!(restored.images.len(), 1);
    assert_eq!(restored.images[0].file_name, "fresh.png");
}

#[test]
fn rejects_file_output_as_ambiguous() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let dataset = Dataset {
        images: vec![Image::new(1u64, "img.png", 100, 80)],
        categories: vec![Category::new(1u64, "person")],
        annotations: vec![],
        ..Default::default()
    };
    let err = write_unity_perception_json(&temp.path().join("out.json"), &dataset)
        .expect_err("file output should be rejected");
    assert!(err.to_string().contains("directory datasets only"));
}
