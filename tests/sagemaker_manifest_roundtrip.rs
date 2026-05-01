//! Integration tests for SageMaker Ground Truth manifest support.

use std::path::Path;

use panlabel::ir::io_sagemaker_manifest::{
    from_sagemaker_manifest_str, read_sagemaker_manifest, to_sagemaker_manifest_string,
    write_sagemaker_manifest, ATTR_LABEL_ATTRIBUTE_NAME, ATTR_SOURCE_REF,
};
use panlabel::ir::BBoxXYXY;
use panlabel::PanlabelError;
use serde_json::Value;

fn fixture_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sample_valid.sagemaker.manifest"
    ))
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {actual} to be close to {expected}"
    );
}

#[test]
fn read_fixture_maps_source_refs_bboxes_confidence_and_categories() {
    let dataset = read_sagemaker_manifest(fixture_path()).expect("read SageMaker fixture");

    assert_eq!(dataset.images.len(), 2);
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.annotations.len(), 2);
    assert_eq!(
        dataset.info.attributes.get(ATTR_LABEL_ATTRIBUTE_NAME),
        Some(&"bounding-box".to_string())
    );

    assert_eq!(dataset.categories[0].name, "cat");
    assert_eq!(dataset.categories[1].name, "dog");

    let img_a = &dataset.images[0];
    assert_eq!(img_a.file_name, "images/img_a.jpg");
    assert_eq!(img_a.width, 640);
    assert_eq!(img_a.height, 480);
    assert_eq!(
        img_a.attributes.get(ATTR_SOURCE_REF),
        Some(&"s3://example-bucket/images/img_a.jpg".to_string())
    );

    let first = &dataset.annotations[0];
    assert_close(first.bbox.xmin(), 10.0);
    assert_close(first.bbox.ymin(), 20.0);
    assert_close(first.bbox.xmax(), 40.0);
    assert_close(first.bbox.ymax(), 60.0);
    assert_eq!(first.confidence, Some(0.95));
}

#[test]
fn write_then_read_preserves_semantics() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out_path = temp.path().join("roundtrip.manifest");

    let dataset = read_sagemaker_manifest(fixture_path()).expect("read input fixture");
    write_sagemaker_manifest(&out_path, &dataset).expect("write manifest");
    let restored = read_sagemaker_manifest(&out_path).expect("read written manifest");

    assert_eq!(restored.images.len(), dataset.images.len());
    assert_eq!(restored.categories.len(), dataset.categories.len());
    assert_eq!(restored.annotations.len(), dataset.annotations.len());
    assert!(restored
        .images
        .iter()
        .any(|image| image.file_name == "images/img_b.jpg"));

    let original = &dataset.annotations[0];
    let roundtripped = &restored.annotations[0];
    assert_close(roundtripped.bbox.xmin(), original.bbox.xmin());
    assert_close(roundtripped.bbox.ymin(), original.bbox.ymin());
    assert_close(roundtripped.bbox.xmax(), original.bbox.xmax());
    assert_close(roundtripped.bbox.ymax(), original.bbox.ymax());
    assert_eq!(roundtripped.confidence, original.confidence);
}

#[test]
fn writer_emits_default_key_source_refs_class_map_and_sorted_rows() {
    let mut dataset = read_sagemaker_manifest(fixture_path()).expect("read input fixture");
    dataset.info.attributes.remove(ATTR_LABEL_ATTRIBUTE_NAME);
    dataset.images.reverse();

    let manifest = to_sagemaker_manifest_string(&dataset).expect("serialize manifest");
    let rows: Vec<Value> = manifest
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid row JSON"))
        .collect();

    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[0]["source-ref"],
        "s3://example-bucket/images/img_a.jpg"
    );
    assert!(rows[0].get("bounding-box").is_some());
    assert_eq!(rows[0]["bounding-box-metadata"]["class-map"]["0"], "cat");
    assert_eq!(rows[0]["bounding-box-metadata"]["class-map"]["1"], "dog");
}

#[test]
fn reader_rejects_key_invalid_cases() {
    let cases = [
        (
            "unsupported metadata type",
            r#"{"source-ref":"s3://bucket/img.jpg","bounding-box":{"annotations":[],"image_size":[{"width":10,"height":10}]},"bounding-box-metadata":{"objects":[],"type":"groundtruth/semantic-segmentation"}}"#,
            "unsupported SageMaker metadata type",
        ),
        (
            "multiple candidates",
            r#"{"source-ref":"s3://bucket/img.jpg","boxes_a":{"annotations":[],"image_size":[{"width":10,"height":10}]},"boxes_a-metadata":{"objects":[],"type":"groundtruth/object-detection"},"boxes_b":{"annotations":[],"image_size":[{"width":10,"height":10}]},"boxes_b-metadata":{"objects":[],"type":"groundtruth/object-detection"}}"#,
            "expected exactly one object-detection label attribute",
        ),
        (
            "mixed label attributes",
            "{\"source-ref\":\"s3://bucket/a.jpg\",\"boxes_a\":{\"annotations\":[],\"image_size\":[{\"width\":10,\"height\":10}]},\"boxes_a-metadata\":{\"objects\":[],\"type\":\"groundtruth/object-detection\"}}\n{\"source-ref\":\"s3://bucket/b.jpg\",\"boxes_b\":{\"annotations\":[],\"image_size\":[{\"width\":10,\"height\":10}]},\"boxes_b-metadata\":{\"objects\":[],\"type\":\"groundtruth/object-detection\"}}",
            "mixed label attribute names",
        ),
        (
            "missing image_size",
            r#"{"source-ref":"s3://bucket/img.jpg","bounding-box":{"annotations":[]},"bounding-box-metadata":{"objects":[],"type":"groundtruth/object-detection"}}"#,
            "expected exactly one object-detection label attribute",
        ),
        (
            "mismatched metadata objects length",
            r#"{"source-ref":"s3://bucket/img.jpg","bounding-box":{"annotations":[{"class_id":0,"left":1,"top":2,"width":3,"height":4}],"image_size":[{"width":10,"height":10}]},"bounding-box-metadata":{"objects":[{"confidence":0.9},{"confidence":0.8}],"type":"groundtruth/object-detection"}}"#,
            "metadata.objects length",
        ),
        (
            "malformed metadata object",
            r#"{"source-ref":"s3://bucket/img.jpg","bounding-box":{"annotations":[],"image_size":[{"width":10,"height":10}]},"bounding-box-metadata":[]}"#,
            "must be a JSON object",
        ),
        (
            "conflicting class-map names",
            "{\"source-ref\":\"s3://bucket/a.jpg\",\"bounding-box\":{\"annotations\":[],\"image_size\":[{\"width\":10,\"height\":10}]},\"bounding-box-metadata\":{\"objects\":[],\"class-map\":{\"0\":\"cat\"},\"type\":\"groundtruth/object-detection\"}}\n{\"source-ref\":\"s3://bucket/b.jpg\",\"bounding-box\":{\"annotations\":[],\"image_size\":[{\"width\":10,\"height\":10}]},\"bounding-box-metadata\":{\"objects\":[],\"class-map\":{\"0\":\"dog\"},\"type\":\"groundtruth/object-detection\"}}",
            "conflicting class-map names",
        ),
    ];

    for (name, manifest, expected_message) in cases {
        let err = from_sagemaker_manifest_str(manifest).expect_err(name);
        match err {
            PanlabelError::SageMakerManifestParse { message, .. }
            | PanlabelError::SageMakerManifestInvalid { message, .. } => {
                assert!(
                    message.contains(expected_message),
                    "case '{name}' expected message containing '{expected_message}', got '{message}'"
                );
            }
            other => panic!("case '{name}' returned unexpected error: {other:?}"),
        }
    }
}

#[test]
fn writer_rejects_negative_bbox_dimensions() {
    let mut dataset = read_sagemaker_manifest(fixture_path()).expect("read input fixture");
    dataset.annotations[0].bbox = BBoxXYXY::from_xyxy(40.0, 20.0, 10.0, 60.0);

    let err = to_sagemaker_manifest_string(&dataset).expect_err("negative width should fail");
    match err {
        PanlabelError::SageMakerManifestWrite { message, .. } => {
            assert!(message.contains("negative bbox width/height"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
