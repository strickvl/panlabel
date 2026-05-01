//! Integration tests for Microsoft VoTT JSON support.

mod common;

use common::write_bmp;
use panlabel::ir::io_vott_json::{read_vott_json, write_vott_json};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

#[test]
fn reads_fixture_with_rectangle_and_polygon_envelope() {
    let dataset = read_vott_json(std::path::Path::new(
        "tests/fixtures/sample_valid.vott.json",
    ))
    .expect("read fixture");

    assert_eq!(dataset.images.len(), 2);
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.annotations.len(), 2);
    assert_eq!(dataset.images[0].file_name, "img1.bmp");
    assert_eq!(dataset.images[0].width, 100);
    assert_eq!(dataset.images[1].height, 40);

    let polygon_ann = dataset
        .annotations
        .iter()
        .find(|ann| {
            ann.attributes
                .get("vott_geometry_enveloped")
                .map(String::as_str)
                == Some("true")
        })
        .expect("polygon envelope annotation");
    assert_eq!(polygon_ann.bbox.xmin(), 5.0);
    assert_eq!(polygon_ann.bbox.ymin(), 4.0);
    assert_eq!(polygon_ann.bbox.xmax(), 30.0);
    assert_eq!(polygon_ann.bbox.ymax(), 20.0);
}

#[test]
fn write_then_read_file_roundtrip() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("annotations.json");

    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "img1.bmp", 640, 480),
            Image::new(2u64, "img2.bmp", 800, 600),
        ],
        categories: vec![Category::new(1u64, "car"), Category::new(2u64, "person")],
        annotations: vec![
            Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(50.0, 30.0, 200.0, 180.0),
            ),
            Annotation::new(
                2u64,
                2u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 400.0, 300.0),
            ),
        ],
        ..Default::default()
    };

    write_vott_json(&json_path, &dataset).expect("write");
    let json = std::fs::read_to_string(&json_path).expect("read json");
    assert!(json.contains("\"assets\""));
    assert!(json.contains("\"RECTANGLE\""));

    let restored = read_vott_json(&json_path).expect("read");
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
    assert_eq!(restored.images[0].file_name, "img1.bmp");

    let ann = restored
        .annotations
        .iter()
        .find(|ann| ann.bbox.xmin() == 50.0)
        .expect("first bbox");
    assert_eq!(ann.bbox.ymin(), 30.0);
    assert_eq!(ann.bbox.xmax(), 200.0);
    assert_eq!(ann.bbox.ymax(), 180.0);
}

#[test]
fn directory_output_uses_canonical_export_layout_with_image_placeholder() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_dir = temp.path().join("out");
    let dataset = Dataset {
        images: vec![Image::new(1u64, "img.bmp", 100, 80)],
        categories: vec![Category::new(1u64, "person")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 50.0, 70.0),
        )],
        ..Default::default()
    };

    write_vott_json(&output_dir, &dataset).expect("write directory");
    assert!(output_dir
        .join("vott-json-export/panlabel-export.json")
        .is_file());
    assert!(output_dir
        .join("vott-json-export/images/README.txt")
        .is_file());

    let restored = read_vott_json(&output_dir).expect("read directory");
    assert_eq!(restored.annotations.len(), 1);
}

#[test]
fn reads_per_asset_json_directory_and_probes_image_dimensions_when_size_missing() {
    let temp = tempfile::tempdir().expect("create temp dir");
    write_bmp(&temp.path().join("img.bmp"), 320, 240);
    std::fs::write(
        temp.path().join("asset.json"),
        r#"{
            "asset": {"id": "asset-1", "name": "img.bmp", "path": "file:img.bmp"},
            "regions": [{"id": "r1", "type": "RECTANGLE", "tags": ["obj"], "boundingBox": {"left": 1, "top": 2, "width": 30, "height": 40}}]
        }"#,
    )
    .expect("write asset json");

    let dataset = read_vott_json(temp.path()).expect("read directory");
    assert_eq!(dataset.images[0].width, 320);
    assert_eq!(dataset.images[0].height, 240);
    assert_eq!(dataset.annotations.len(), 1);
}

#[test]
fn multi_tag_region_expands_to_multiple_annotations() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("multi.json");
    std::fs::write(
        &json_path,
        r#"{
            "asset": {"name": "img.bmp", "size": {"width": 100, "height": 80}},
            "regions": [{"id": "r1", "type": "RECTANGLE", "tags": ["cat", "animal"], "boundingBox": {"left": 1, "top": 2, "width": 3, "height": 4}}]
        }"#,
    )
    .expect("write json");

    let dataset = read_vott_json(&json_path).expect("read");
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.annotations.len(), 2);
}
