mod common;

use panlabel::ir::io_labelme_json::*;
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

#[test]
fn single_file_roundtrip() {
    let dataset = Dataset {
        images: vec![Image::new(1u64, "test.jpg", 640, 480)],
        categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
        annotations: vec![
            Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 80.0),
            ),
            Annotation::new(
                2u64,
                1u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 200.0, 200.0),
            ),
        ],
        ..Default::default()
    };

    let json = to_labelme_string(&dataset).expect("serialize");
    let restored = from_labelme_str(&json).expect("parse");

    assert_eq!(restored.images.len(), 1);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 2);

    // Bbox preserved
    assert_eq!(restored.annotations[0].bbox.xmin(), 10.0);
    assert_eq!(restored.annotations[0].bbox.ymax(), 80.0);
    assert_eq!(restored.annotations[1].bbox.xmin(), 50.0);
    assert_eq!(restored.annotations[1].bbox.ymax(), 200.0);
}

#[test]
fn directory_roundtrip() {
    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "a.jpg", 640, 480),
            Image::new(2u64, "b.png", 800, 600),
        ],
        categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
        annotations: vec![
            Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 80.0),
            ),
            Annotation::new(
                2u64,
                2u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 200.0, 200.0),
            ),
        ],
        ..Default::default()
    };

    let temp = tempfile::tempdir().expect("create temp dir");
    write_labelme_json(temp.path(), &dataset).expect("write");

    // Verify structure
    assert!(temp.path().join("annotations/a.json").is_file());
    assert!(temp.path().join("annotations/b.json").is_file());
    assert!(temp.path().join("images/README.txt").is_file());

    let restored = read_labelme_json(temp.path()).expect("read");
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
    assert_eq!(restored.categories.len(), 2);

    // Images sorted by file_name
    assert_eq!(restored.images[0].file_name, "a.jpg");
    assert_eq!(restored.images[1].file_name, "b.png");
}

#[test]
fn polygon_converted_to_bbox_envelope() {
    let json = r#"{
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": "region",
                "points": [[10.0, 20.0], [100.0, 30.0], [80.0, 90.0], [5.0, 70.0]],
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": "test.jpg",
        "imageHeight": 100,
        "imageWidth": 200
    }"#;

    let dataset = from_labelme_str(json).expect("parse");
    assert_eq!(dataset.annotations.len(), 1);

    let ann = &dataset.annotations[0];
    // Envelope of [[10,20],[100,30],[80,90],[5,70]] = xyxy(5,20,100,90)
    assert_eq!(ann.bbox.xmin(), 5.0);
    assert_eq!(ann.bbox.ymin(), 20.0);
    assert_eq!(ann.bbox.xmax(), 100.0);
    assert_eq!(ann.bbox.ymax(), 90.0);
    assert_eq!(
        ann.attributes.get("labelme_shape_type"),
        Some(&"polygon".to_string())
    );
}

#[test]
fn unannotated_images_preserved() {
    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "with_ann.jpg", 100, 100),
            Image::new(2u64, "no_ann.jpg", 100, 100),
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

    let temp = tempfile::tempdir().expect("create temp dir");
    write_labelme_json(temp.path(), &dataset).expect("write");

    // Both annotation files should exist (one with shapes, one empty)
    assert!(temp.path().join("annotations/with_ann.json").is_file());
    assert!(temp.path().join("annotations/no_ann.json").is_file());

    let restored = read_labelme_json(temp.path()).expect("read");
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.annotations.len(), 1);
}

#[test]
fn deterministic_ids() {
    // Write images out of alphabetical order, verify reader sorts them
    let dataset = Dataset {
        images: vec![
            Image::new(2u64, "z.jpg", 100, 100),
            Image::new(1u64, "a.jpg", 100, 100),
        ],
        categories: vec![Category::new(1u64, "obj")],
        annotations: vec![
            Annotation::new(
                1u64,
                2u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
            ),
            Annotation::new(
                2u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
            ),
        ],
        ..Default::default()
    };

    let temp = tempfile::tempdir().expect("create temp dir");
    write_labelme_json(temp.path(), &dataset).expect("write");

    let restored = read_labelme_json(temp.path()).expect("read");
    // Images should be sorted by file_name: a.jpg gets ID 1, z.jpg gets ID 2
    assert_eq!(restored.images[0].file_name, "a.jpg");
    assert_eq!(restored.images[0].id.as_u64(), 1);
    assert_eq!(restored.images[1].file_name, "z.jpg");
    assert_eq!(restored.images[1].id.as_u64(), 2);
}
