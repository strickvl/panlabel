//! Integration tests for Labelbox JSON/NDJSON support.

use std::path::Path;

use panlabel::ir::io_labelbox_json::{
    from_labelbox_json_str, read_labelbox_json, to_labelbox_ndjson_string, write_labelbox_json,
    ATTR_FEATURE_ID, ATTR_POLYGON_ENVELOPED, ATTR_SKIPPED_OBJECTS,
};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};
use serde_json::Value;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {actual} to be close to {expected}"
    );
}

#[test]
fn reads_ndjson_fixture_with_bbox_polygon_envelope_and_skipped_object() {
    let dataset = read_labelbox_json(Path::new("tests/fixtures/sample_valid.labelbox.ndjson"))
        .expect("read Labelbox fixture");

    assert_eq!(dataset.images.len(), 2);
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.annotations.len(), 2);
    assert_eq!(dataset.images[0].file_name, "img1.jpg");
    assert_eq!(dataset.images[1].file_name, "img2.jpg");
    assert_eq!(
        dataset.info.attributes.get(ATTR_SKIPPED_OBJECTS),
        Some(&"1".to_string())
    );

    let bbox_ann = dataset
        .annotations
        .iter()
        .find(|ann| ann.attributes.get(ATTR_FEATURE_ID).map(String::as_str) == Some("bbox-1"))
        .expect("bbox annotation");
    assert_close(bbox_ann.bbox.xmin(), 10.0);
    assert_close(bbox_ann.bbox.ymin(), 20.0);
    assert_close(bbox_ann.bbox.xmax(), 50.0);
    assert_close(bbox_ann.bbox.ymax(), 50.0);

    let polygon_ann = dataset
        .annotations
        .iter()
        .find(|ann| {
            ann.attributes
                .get(ATTR_POLYGON_ENVELOPED)
                .map(String::as_str)
                == Some("true")
        })
        .expect("polygon annotation");
    assert_close(polygon_ann.bbox.xmin(), 5.0);
    assert_close(polygon_ann.bbox.ymin(), 4.0);
    assert_close(polygon_ann.bbox.xmax(), 30.0);
    assert_close(polygon_ann.bbox.ymax(), 40.0);
}

#[test]
fn reads_single_json_row_and_preserves_image_with_no_supported_objects() {
    let json = r#"{
        "data_row": {"external_id": "empty.jpg"},
        "media_attributes": {"width": 20, "height": 10},
        "projects": {"project-a": {"labels": [{"annotations": {"objects": [
            {"feature_id":"mask-1", "name":"mask", "annotation_kind":"ImageSegmentationMask", "mask": {}}
        ]}}]}}
    }"#;

    let dataset = from_labelbox_json_str(json).expect("read single row");
    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "empty.jpg");
    assert_eq!(dataset.annotations.len(), 0);
    assert_eq!(
        dataset.info.attributes.get(ATTR_SKIPPED_OBJECTS),
        Some(&"1".to_string())
    );
}

#[test]
fn write_then_read_json_array_roundtrip_preserves_detection_semantics() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("labelbox.json");
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
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 50.0, 50.0),
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

    write_labelbox_json(&json_path, &dataset).expect("write Labelbox JSON");
    let raw = std::fs::read_to_string(&json_path).expect("read output JSON");
    let rows: Value = serde_json::from_str(&raw).expect("parse output JSON");
    assert!(rows.is_array());
    assert_eq!(rows[0]["data_row"]["external_id"], "img1.jpg");
    assert_eq!(
        rows[0]["projects"]["panlabel-project"]["labels"][0]["annotations"]["objects"][0]
            ["annotation_kind"],
        "ImageBoundingBox"
    );

    let restored = read_labelbox_json(&json_path).expect("read written JSON");
    assert_eq!(restored.images.len(), dataset.images.len());
    assert_eq!(restored.categories.len(), dataset.categories.len());
    assert_eq!(restored.annotations.len(), dataset.annotations.len());
    assert_close(restored.annotations[0].bbox.xmin(), 10.0);
    assert_close(restored.annotations[0].bbox.ymax(), 50.0);
}

#[test]
fn writer_emits_ndjson_for_jsonl_paths() {
    let dataset = Dataset {
        images: vec![Image::new(1u64, "img.jpg", 100, 80)],
        categories: vec![Category::new(1u64, "cat")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 50.0, 60.0),
        )],
        ..Default::default()
    };

    let ndjson = to_labelbox_ndjson_string(&dataset).expect("serialize NDJSON");
    assert_eq!(ndjson.lines().count(), 1);
    let row: Value = serde_json::from_str(ndjson.lines().next().unwrap()).expect("row JSON");
    assert_eq!(row["data_row"]["external_id"], "img.jpg");
}
