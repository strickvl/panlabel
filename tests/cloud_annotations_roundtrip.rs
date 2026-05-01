//! Integration tests for IBM Cloud Annotations JSON support.

mod common;

use common::write_bmp;
use panlabel::ir::io_cloud_annotations_json::{
    read_cloud_annotations_json, write_cloud_annotations_json,
};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

#[test]
fn write_then_read_roundtrip_with_normalized_coordinates() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("_annotations.json");

    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "img1.bmp", 200, 100),
            Image::new(2u64, "img2.bmp", 400, 200),
        ],
        categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
        annotations: vec![
            Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(20.0, 10.0, 100.0, 50.0),
            ),
            Annotation::new(
                2u64,
                2u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(40.0, 20.0, 200.0, 100.0),
            ),
        ],
        ..Default::default()
    };

    write_cloud_annotations_json(&json_path, &dataset).expect("write");
    let written = std::fs::read_to_string(&json_path).expect("read written json");
    let value: serde_json::Value = serde_json::from_str(&written).expect("parse written json");
    assert_eq!(value["type"], "localization");
    assert_eq!(value["annotations"]["img1.bmp"][0]["x"], 0.1);
    assert_eq!(value["annotations"]["img1.bmp"][0]["x2"], 0.5);

    write_bmp(&temp.path().join("img1.bmp"), 200, 100);
    write_bmp(&temp.path().join("img2.bmp"), 400, 200);

    let restored = read_cloud_annotations_json(&json_path).expect("read");
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
    assert_eq!(restored.images[0].file_name, "img1.bmp");

    let ann = &restored.annotations[0];
    assert!((ann.bbox.xmin() - 20.0).abs() < 1e-9);
    assert!((ann.bbox.ymin() - 10.0).abs() < 1e-9);
    assert!((ann.bbox.xmax() - 100.0).abs() < 1e-9);
    assert!((ann.bbox.ymax() - 50.0).abs() < 1e-9);
}

#[test]
fn directory_path_reads_annotations_file_and_images_subdir() {
    let temp = tempfile::tempdir().expect("create temp dir");
    std::fs::create_dir_all(temp.path().join("images")).unwrap();
    write_bmp(&temp.path().join("images/nested.bmp"), 320, 240);
    std::fs::write(
        temp.path().join("_annotations.json"),
        r#"{
            "version": "1.0",
            "type": "localization",
            "labels": ["thing"],
            "annotations": {
                "nested.bmp": [{"x": 0.25, "y": 0.25, "x2": 0.75, "y2": 0.5, "label": "thing"}]
            }
        }"#,
    )
    .unwrap();

    let dataset = read_cloud_annotations_json(temp.path()).expect("read directory");
    assert_eq!(dataset.images[0].width, 320);
    assert_eq!(dataset.images[0].height, 240);
    assert_eq!(dataset.annotations[0].bbox.xmin(), 80.0);
    assert_eq!(dataset.annotations[0].bbox.ymax(), 120.0);
}

#[test]
fn unannotated_images_are_preserved_as_empty_arrays() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let dataset = Dataset {
        images: vec![Image::new(1u64, "empty.bmp", 100, 100)],
        categories: vec![Category::new(1u64, "obj")],
        annotations: vec![],
        ..Default::default()
    };

    write_cloud_annotations_json(temp.path(), &dataset).expect("write dir");
    let annotations_path = temp.path().join("_annotations.json");
    assert!(annotations_path.is_file());
    assert!(temp.path().join("images/README.txt").is_file());

    write_bmp(&temp.path().join("empty.bmp"), 100, 100);
    let restored = read_cloud_annotations_json(temp.path()).expect("read");
    assert_eq!(restored.images.len(), 1);
    assert_eq!(restored.annotations.len(), 0);
}
