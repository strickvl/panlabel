//! Integration tests for RetinaNet CSV format support (write-then-read roundtrip).

mod common;

use common::write_bmp;
use panlabel::ir::io_retinanet_csv::{read_retinanet_csv, write_retinanet_csv};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

#[test]
fn write_then_read_roundtrip() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let csv_path = temp.path().join("annotations.csv");

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
                1u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(300.0, 100.0, 450.0, 350.0),
            ),
            Annotation::new(
                3u64,
                2u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 400.0, 300.0),
            ),
        ],
        ..Default::default()
    };

    // Write the CSV
    write_retinanet_csv(&csv_path, &dataset).expect("write retinanet csv");

    // Create tiny BMP files next to the CSV so the reader can resolve dimensions
    write_bmp(&temp.path().join("img1.bmp"), 640, 480);
    write_bmp(&temp.path().join("img2.bmp"), 800, 600);

    // Read back
    let restored = read_retinanet_csv(&csv_path).expect("read retinanet csv");

    // Basic counts
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 3);

    // Image dimensions resolved from BMP files
    let img1 = restored
        .images
        .iter()
        .find(|i| i.file_name == "img1.bmp")
        .expect("img1 present");
    assert_eq!(img1.width, 640);
    assert_eq!(img1.height, 480);

    let img2 = restored
        .images
        .iter()
        .find(|i| i.file_name == "img2.bmp")
        .expect("img2 present");
    assert_eq!(img2.width, 800);
    assert_eq!(img2.height, 600);

    // Bbox values roundtrip
    let mut orig_bboxes: Vec<(f64, f64, f64, f64)> = dataset
        .annotations
        .iter()
        .map(|a| (a.bbox.xmin(), a.bbox.ymin(), a.bbox.xmax(), a.bbox.ymax()))
        .collect();
    orig_bboxes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut rest_bboxes: Vec<(f64, f64, f64, f64)> = restored
        .annotations
        .iter()
        .map(|a| (a.bbox.xmin(), a.bbox.ymin(), a.bbox.xmax(), a.bbox.ymax()))
        .collect();
    rest_bboxes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(orig_bboxes.len(), rest_bboxes.len());
    for (orig, rest) in orig_bboxes.iter().zip(rest_bboxes.iter()) {
        assert!((orig.0 - rest.0).abs() < 1e-9, "xmin mismatch");
        assert!((orig.1 - rest.1).abs() < 1e-9, "ymin mismatch");
        assert!((orig.2 - rest.2).abs() < 1e-9, "xmax mismatch");
        assert!((orig.3 - rest.3).abs() < 1e-9, "ymax mismatch");
    }

    // Category names roundtrip
    let mut orig_cats: Vec<&str> = dataset.categories.iter().map(|c| c.name.as_str()).collect();
    orig_cats.sort();
    let mut rest_cats: Vec<&str> = restored
        .categories
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    rest_cats.sort();
    assert_eq!(orig_cats, rest_cats);
}

#[test]
fn unannotated_images_roundtrip() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let csv_path = temp.path().join("annotations.csv");

    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "annotated.bmp", 640, 480),
            Image::new(2u64, "empty.bmp", 800, 600),
        ],
        categories: vec![Category::new(1u64, "car")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(20.0, 30.0, 200.0, 180.0),
        )],
        ..Default::default()
    };

    write_retinanet_csv(&csv_path, &dataset).expect("write");

    // Verify the CSV contains the empty row for the unannotated image
    let csv_content = std::fs::read_to_string(&csv_path).expect("read csv");
    let lines: Vec<&str> = csv_content.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(
        lines.iter().any(|l| *l == "empty.bmp,,,,,"),
        "unannotated image should produce empty row"
    );

    // Create image files and read back
    write_bmp(&temp.path().join("annotated.bmp"), 640, 480);
    write_bmp(&temp.path().join("empty.bmp"), 800, 600);

    let restored = read_retinanet_csv(&csv_path).expect("read");

    // Both images present
    assert_eq!(restored.images.len(), 2);
    // Only one annotation
    assert_eq!(restored.annotations.len(), 1);

    let names: Vec<&str> = restored
        .images
        .iter()
        .map(|i| i.file_name.as_str())
        .collect();
    assert!(names.contains(&"annotated.bmp"));
    assert!(names.contains(&"empty.bmp"));
}

#[test]
fn deterministic_ids() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let csv_path = temp.path().join("annotations.csv");

    // Write images out of alphabetical order
    let dataset = Dataset {
        images: vec![
            Image::new(2u64, "z_img.bmp", 640, 480),
            Image::new(1u64, "a_img.bmp", 800, 600),
        ],
        categories: vec![Category::new(1u64, "obj")],
        annotations: vec![
            Annotation::new(
                1u64,
                2u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 200.0),
            ),
            Annotation::new(
                2u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 150.0, 250.0),
            ),
        ],
        ..Default::default()
    };

    write_retinanet_csv(&csv_path, &dataset).expect("write");
    write_bmp(&temp.path().join("a_img.bmp"), 800, 600);
    write_bmp(&temp.path().join("z_img.bmp"), 640, 480);

    let restored = read_retinanet_csv(&csv_path).expect("read");

    // Reader assigns IDs by sorted image path: a_img gets 1, z_img gets 2
    assert_eq!(restored.images[0].file_name, "a_img.bmp");
    assert_eq!(restored.images[0].id.as_u64(), 1);
    assert_eq!(restored.images[1].file_name, "z_img.bmp");
    assert_eq!(restored.images[1].id.as_u64(), 2);

    // Category IDs are also deterministic
    assert_eq!(restored.categories[0].name, "obj");
    assert_eq!(restored.categories[0].id.as_u64(), 1);

    // Annotation IDs assigned in row order (CSV writer sorts by image file_name)
    assert_eq!(restored.annotations[0].id.as_u64(), 1);
    assert_eq!(restored.annotations[1].id.as_u64(), 2);
}
