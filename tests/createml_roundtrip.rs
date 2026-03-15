mod common;

use common::write_bmp;
use panlabel::ir::io_createml_json::*;
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};
use std::path::Path;

/// Helper: stage BMP images for a dataset into a temp directory
fn stage_images(dir: &Path, dataset: &Dataset) {
    for img in &dataset.images {
        write_bmp(&dir.join(&img.file_name), img.width, img.height);
    }
}

#[test]
fn roundtrip_with_staged_images() {
    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "a.bmp", 640, 480),
            Image::new(2u64, "b.bmp", 800, 600),
        ],
        categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
        annotations: vec![
            Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(110.0, 170.0, 190.0, 230.0),
            ),
            Annotation::new(
                2u64,
                2u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(100.0, 100.0, 300.0, 250.0),
            ),
        ],
        ..Default::default()
    };

    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("annotations.json");

    // Write CreateML JSON
    write_createml_json(&json_path, &dataset).expect("write");

    // Stage image files for dimension resolution
    stage_images(temp.path(), &dataset);

    // Read back
    let restored = read_createml_json(&json_path).expect("read");

    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 2);

    // Images sorted by filename
    assert_eq!(restored.images[0].file_name, "a.bmp");
    assert_eq!(restored.images[0].width, 640);
    assert_eq!(restored.images[0].height, 480);
    assert_eq!(restored.images[1].file_name, "b.bmp");

    // Check bbox round-trip: xyxy(110,170,190,230) -> cxcywh(150,200,80,60) -> xyxy(110,170,190,230)
    let ann = &restored.annotations[0];
    assert!((ann.bbox.xmin() - 110.0).abs() < 1e-9);
    assert!((ann.bbox.ymin() - 170.0).abs() < 1e-9);
    assert!((ann.bbox.xmax() - 190.0).abs() < 1e-9);
    assert!((ann.bbox.ymax() - 230.0).abs() < 1e-9);
}

#[test]
fn image_resolution_from_images_subdir() {
    let temp = tempfile::tempdir().expect("create temp dir");

    // Stage images in images/ subdirectory
    let images_dir = temp.path().join("images");
    std::fs::create_dir_all(&images_dir).unwrap();
    write_bmp(&images_dir.join("test.bmp"), 320, 240);

    // Write a CreateML JSON
    let json = r#"[
        {
            "image": "test.bmp",
            "annotations": [
                {
                    "label": "obj",
                    "coordinates": { "x": 50.0, "y": 50.0, "width": 40.0, "height": 30.0 }
                }
            ]
        }
    ]"#;

    let json_path = temp.path().join("data.json");
    std::fs::write(&json_path, json).unwrap();

    let dataset = read_createml_json(&json_path).expect("read");
    assert_eq!(dataset.images[0].width, 320);
    assert_eq!(dataset.images[0].height, 240);
}

#[test]
fn unannotated_images_preserved() {
    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "with.bmp", 100, 100),
            Image::new(2u64, "without.bmp", 200, 200),
        ],
        categories: vec![Category::new(1u64, "obj")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
        )],
        ..Default::default()
    };

    let json = to_createml_string(&dataset).expect("serialize");
    let rows: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();

    // Both images present, even the one without annotations
    assert_eq!(rows.len(), 2);
    let without = rows
        .iter()
        .find(|r| r["image"] == "without.bmp")
        .expect("unannotated image");
    assert_eq!(without["annotations"].as_array().unwrap().len(), 0);
}

#[test]
fn duplicate_image_rejected() {
    let temp = tempfile::tempdir().expect("create temp dir");
    write_bmp(&temp.path().join("dup.bmp"), 100, 100);

    let json = r#"[
        { "image": "dup.bmp", "annotations": [] },
        { "image": "dup.bmp", "annotations": [] }
    ]"#;

    let json_path = temp.path().join("data.json");
    std::fs::write(&json_path, json).unwrap();

    let result = read_createml_json(&json_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("duplicate"));
}

#[test]
fn missing_image_file_errors() {
    let temp = tempfile::tempdir().expect("create temp dir");
    // Don't stage any images

    let json = r#"[
        {
            "image": "missing.bmp",
            "annotations": [
                { "label": "x", "coordinates": { "x": 10, "y": 10, "width": 5, "height": 5 } }
            ]
        }
    ]"#;

    let json_path = temp.path().join("data.json");
    std::fs::write(&json_path, json).unwrap();

    let result = read_createml_json(&json_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn center_coordinate_conversion_correct() {
    // Write a dataset with known XYXY and verify the CreateML JSON has correct center coords
    let dataset = Dataset {
        images: vec![Image::new(1u64, "test.bmp", 200, 200)],
        categories: vec![Category::new(1u64, "box")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            // xyxy(20, 30, 80, 70) -> center=(50,50), w=60, h=40
            BBoxXYXY::<Pixel>::from_xyxy(20.0, 30.0, 80.0, 70.0),
        )],
        ..Default::default()
    };

    let json = to_createml_string(&dataset).expect("serialize");
    let rows: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    let coords = &rows[0]["annotations"][0]["coordinates"];

    assert!((coords["x"].as_f64().unwrap() - 50.0).abs() < 1e-9);
    assert!((coords["y"].as_f64().unwrap() - 50.0).abs() < 1e-9);
    assert!((coords["width"].as_f64().unwrap() - 60.0).abs() < 1e-9);
    assert!((coords["height"].as_f64().unwrap() - 40.0).abs() < 1e-9);
}
