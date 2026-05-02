//! Integration tests for KITTI format support (write-then-read roundtrip).

mod common;

use common::write_bmp;
use panlabel::ir::io_kitti::{read_kitti_dir, write_kitti_dir};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};
use std::fs;

#[test]
fn write_then_read_roundtrip() {
    let temp = tempfile::tempdir().expect("create temp dir");

    let mut ann1 = Annotation::new(
        1u64,
        1u64,
        1u64,
        BBoxXYXY::<Pixel>::from_xyxy(50.0, 30.0, 200.0, 180.0),
    );
    ann1.attributes
        .insert("kitti_truncated".to_string(), "0.50".to_string());
    ann1.attributes
        .insert("kitti_occluded".to_string(), "1".to_string());
    ann1.attributes
        .insert("kitti_alpha".to_string(), "-1.57".to_string());
    ann1.attributes
        .insert("kitti_dim_height".to_string(), "1.52".to_string());
    ann1.attributes
        .insert("kitti_dim_width".to_string(), "1.60".to_string());
    ann1.attributes
        .insert("kitti_dim_length".to_string(), "3.23".to_string());
    ann1.attributes
        .insert("kitti_loc_x".to_string(), "1.51".to_string());
    ann1.attributes
        .insert("kitti_loc_y".to_string(), "1.65".to_string());
    ann1.attributes
        .insert("kitti_loc_z".to_string(), "13.73".to_string());
    ann1.attributes
        .insert("kitti_rotation_y".to_string(), "-1.59".to_string());
    ann1.confidence = Some(0.95);

    let ann2 = Annotation::new(
        2u64,
        1u64,
        2u64,
        BBoxXYXY::<Pixel>::from_xyxy(300.0, 100.0, 450.0, 350.0),
    )
    .with_attribute("kitti_truncated", "0.00")
    .with_attribute("kitti_occluded", "0")
    .with_attribute("kitti_alpha", "0.20")
    .with_attribute("kitti_dim_height", "1.70")
    .with_attribute("kitti_dim_width", "0.60")
    .with_attribute("kitti_dim_length", "0.80")
    .with_attribute("kitti_loc_x", "-5.00")
    .with_attribute("kitti_loc_y", "1.80")
    .with_attribute("kitti_loc_z", "20.00")
    .with_attribute("kitti_rotation_y", "-0.50");

    let ann3 = Annotation::new(
        3u64,
        2u64,
        1u64,
        BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 400.0, 300.0),
    )
    .with_attribute("kitti_truncated", "0.25")
    .with_attribute("kitti_occluded", "2")
    .with_attribute("kitti_alpha", "-0.10")
    .with_attribute("kitti_dim_height", "1.80")
    .with_attribute("kitti_dim_width", "0.70")
    .with_attribute("kitti_dim_length", "4.50")
    .with_attribute("kitti_loc_x", "2.00")
    .with_attribute("kitti_loc_y", "1.50")
    .with_attribute("kitti_loc_z", "8.00")
    .with_attribute("kitti_rotation_y", "0.30");

    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "img1.bmp", 640, 480),
            Image::new(2u64, "img2.bmp", 800, 600),
        ],
        categories: vec![
            Category::new(1u64, "Car"),
            Category::new(2u64, "Pedestrian"),
        ],
        annotations: vec![ann1, ann2, ann3],
        ..Default::default()
    };

    // Write the dataset
    write_kitti_dir(temp.path(), &dataset).expect("write kitti dir");

    // Verify the directory structure
    assert!(temp.path().join("label_2/img1.txt").is_file());
    assert!(temp.path().join("label_2/img2.txt").is_file());
    assert!(temp.path().join("image_2").is_dir());

    // Create tiny BMP image files so the reader can resolve dimensions
    write_bmp(&temp.path().join("image_2/img1.bmp"), 640, 480);
    write_bmp(&temp.path().join("image_2/img2.bmp"), 800, 600);

    // Read back
    let restored = read_kitti_dir(temp.path()).expect("read kitti dir");

    // Basic counts
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 3);

    // Image dimensions resolved from BMP files
    assert_eq!(restored.images[0].width, 640);
    assert_eq!(restored.images[0].height, 480);
    assert_eq!(restored.images[1].width, 800);
    assert_eq!(restored.images[1].height, 600);

    // Sort annotations by bbox xmin for deterministic comparison
    let mut orig_anns: Vec<_> = dataset
        .annotations
        .iter()
        .map(|a| (a.bbox.xmin(), a.bbox.ymin(), a.bbox.xmax(), a.bbox.ymax()))
        .collect();
    orig_anns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut rest_anns: Vec<_> = restored
        .annotations
        .iter()
        .map(|a| (a.bbox.xmin(), a.bbox.ymin(), a.bbox.xmax(), a.bbox.ymax()))
        .collect();
    rest_anns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (orig, rest) in orig_anns.iter().zip(rest_anns.iter()) {
        assert!((orig.0 - rest.0).abs() < 0.01, "xmin mismatch");
        assert!((orig.1 - rest.1).abs() < 0.01, "ymin mismatch");
        assert!((orig.2 - rest.2).abs() < 0.01, "xmax mismatch");
        assert!((orig.3 - rest.3).abs() < 0.01, "ymax mismatch");
    }
}

#[test]
fn kitti_attributes_roundtrip() {
    let temp = tempfile::tempdir().expect("create temp dir");

    let ann = Annotation::new(
        1u64,
        1u64,
        1u64,
        BBoxXYXY::<Pixel>::from_xyxy(100.0, 50.0, 300.0, 250.0),
    )
    .with_attribute("kitti_truncated", "0.75")
    .with_attribute("kitti_occluded", "2")
    .with_attribute("kitti_alpha", "-1.20")
    .with_attribute("kitti_dim_height", "1.65")
    .with_attribute("kitti_dim_width", "0.55")
    .with_attribute("kitti_dim_length", "4.10")
    .with_attribute("kitti_loc_x", "3.50")
    .with_attribute("kitti_loc_y", "1.45")
    .with_attribute("kitti_loc_z", "15.00")
    .with_attribute("kitti_rotation_y", "0.80");

    let dataset = Dataset {
        images: vec![Image::new(1u64, "test.bmp", 640, 480)],
        categories: vec![Category::new(1u64, "Car")],
        annotations: vec![ann],
        ..Default::default()
    };

    write_kitti_dir(temp.path(), &dataset).expect("write");
    write_bmp(&temp.path().join("image_2/test.bmp"), 640, 480);

    let restored = read_kitti_dir(temp.path()).expect("read");
    let rann = &restored.annotations[0];

    // Verify all KITTI-specific attributes roundtrip
    let check_attr = |key: &str, expected: f64| {
        let val: f64 = rann.attributes.get(key).unwrap().parse().unwrap();
        assert!(
            (val - expected).abs() < 0.01,
            "{key}: expected {expected}, got {val}"
        );
    };

    check_attr("kitti_truncated", 0.75);
    check_attr("kitti_alpha", -1.20);
    check_attr("kitti_dim_height", 1.65);
    check_attr("kitti_dim_width", 0.55);
    check_attr("kitti_dim_length", 4.10);
    check_attr("kitti_loc_x", 3.50);
    check_attr("kitti_loc_y", 1.45);
    check_attr("kitti_loc_z", 15.00);
    check_attr("kitti_rotation_y", 0.80);

    let occluded: u8 = rann
        .attributes
        .get("kitti_occluded")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(occluded, 2);
}

#[test]
fn confidence_roundtrips_through_score() {
    let temp = tempfile::tempdir().expect("create temp dir");

    let mut ann = Annotation::new(
        1u64,
        1u64,
        1u64,
        BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 200.0, 300.0),
    );
    ann.confidence = Some(0.87);

    let dataset = Dataset {
        images: vec![Image::new(1u64, "scored.bmp", 640, 480)],
        categories: vec![Category::new(1u64, "Car")],
        annotations: vec![ann],
        ..Default::default()
    };

    write_kitti_dir(temp.path(), &dataset).expect("write");
    write_bmp(&temp.path().join("image_2/scored.bmp"), 640, 480);

    let restored = read_kitti_dir(temp.path()).expect("read");
    let rann = &restored.annotations[0];

    assert!(rann.confidence.is_some());
    assert!(
        (rann.confidence.unwrap() - 0.87).abs() < 0.01,
        "confidence should roundtrip"
    );
}

#[test]
fn unannotated_images_produce_empty_txt() {
    let temp = tempfile::tempdir().expect("create temp dir");

    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "annotated.bmp", 640, 480),
            Image::new(2u64, "empty.bmp", 800, 600),
        ],
        categories: vec![Category::new(1u64, "Car")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 200.0),
        )],
        ..Default::default()
    };

    write_kitti_dir(temp.path(), &dataset).expect("write");

    // Unannotated image should have an empty label file
    let empty_label =
        fs::read_to_string(temp.path().join("label_2/empty.txt")).expect("read empty label");
    assert!(
        empty_label.is_empty(),
        "unannotated image should produce empty .txt"
    );

    // Annotated image should have content
    let annotated_label = fs::read_to_string(temp.path().join("label_2/annotated.txt"))
        .expect("read annotated label");
    assert!(
        !annotated_label.is_empty(),
        "annotated image should produce non-empty .txt"
    );

    // Read back: both images present, only one annotation
    write_bmp(&temp.path().join("image_2/annotated.bmp"), 640, 480);
    write_bmp(&temp.path().join("image_2/empty.bmp"), 800, 600);

    let restored = read_kitti_dir(temp.path()).expect("read");
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.annotations.len(), 1);
}

#[test]
fn duplicate_image_ids_write_annotations_only_for_first_filename_sorted_image() {
    let temp = tempfile::tempdir().expect("create temp dir");

    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "b_duplicate.bmp", 100, 100),
            Image::new(1u64, "a_duplicate.bmp", 100, 100),
        ],
        categories: vec![Category::new(1u64, "Car")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(1.0, 2.0, 3.0, 4.0),
        )],
        ..Default::default()
    };

    write_kitti_dir(temp.path(), &dataset).expect("write kitti");

    let first_label = fs::read_to_string(temp.path().join("label_2/a_duplicate.txt"))
        .expect("read first duplicate label");
    let second_label = fs::read_to_string(temp.path().join("label_2/b_duplicate.txt"))
        .expect("read second duplicate label");

    assert!(!first_label.is_empty());
    assert!(second_label.is_empty());
}

#[test]
fn deterministic_ids() {
    let temp = tempfile::tempdir().expect("create temp dir");

    // Write images out of alphabetical order
    let dataset = Dataset {
        images: vec![
            Image::new(2u64, "z_img.bmp", 640, 480),
            Image::new(1u64, "a_img.bmp", 800, 600),
        ],
        categories: vec![Category::new(1u64, "Car")],
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

    write_kitti_dir(temp.path(), &dataset).expect("write");
    write_bmp(&temp.path().join("image_2/a_img.bmp"), 800, 600);
    write_bmp(&temp.path().join("image_2/z_img.bmp"), 640, 480);

    let restored = read_kitti_dir(temp.path()).expect("read");

    // Reader assigns IDs by sorted label-file order: a_img gets 1, z_img gets 2
    assert_eq!(restored.images[0].file_name, "a_img.bmp");
    assert_eq!(restored.images[0].id.as_u64(), 1);
    assert_eq!(restored.images[1].file_name, "z_img.bmp");
    assert_eq!(restored.images[1].id.as_u64(), 2);

    // Categories sorted alphabetically
    assert_eq!(restored.categories[0].name, "Car");
    assert_eq!(restored.categories[0].id.as_u64(), 1);
}
