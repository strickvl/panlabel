//! Integration tests for YOLO format support.

use std::fs;
use std::path::Path;

use panlabel::ir::io_yolo::{read_yolo_dir, write_yolo_dir};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image};
use panlabel::PanlabelError;

mod common;
use common::write_bmp;

fn create_sample_dataset(root: &Path) {
    fs::create_dir_all(root.join("images/train")).expect("create images dir");
    fs::create_dir_all(root.join("labels/train")).expect("create labels dir");

    write_bmp(&root.join("images/train/img_b.bmp"), 12, 8);
    write_bmp(&root.join("images/train/img_a.bmp"), 20, 10);
    write_bmp(&root.join("images/train/img_c.bmp"), 6, 6);

    fs::write(root.join("data.yaml"), "names:\n  - person\n  - bicycle\n")
        .expect("write data yaml");

    fs::write(
        root.join("labels/train/img_a.txt"),
        "0 0.5 0.5 0.4 0.4\n1 0.2 0.3 0.1 0.2\n",
    )
    .expect("write label file a");
    fs::write(root.join("labels/train/img_b.txt"), "1 0.5 0.5 0.5 0.5\n")
        .expect("write label file b");
    // img_c intentionally has no label file.
}

fn copy_images_for_reread(src_root: &Path, dst_root: &Path) {
    for image_name in ["img_a.bmp", "img_b.bmp", "img_c.bmp"] {
        let src = src_root.join("images/train").join(image_name);
        let dst = dst_root.join("images/train").join(image_name);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).expect("create output image dir");
        }
        fs::copy(src, dst).expect("copy image for reread");
    }
}

#[test]
fn read_yolo_from_root_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_dataset(temp.path());

    let dataset = read_yolo_dir(temp.path()).expect("read yolo dataset");

    assert_eq!(dataset.images.len(), 3);
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.annotations.len(), 3);

    // Images should be assigned deterministically by relative path.
    assert_eq!(dataset.images[0].file_name, "train/img_a.bmp");
    assert_eq!(dataset.images[0].id.as_u64(), 1);
    assert_eq!(dataset.images[1].file_name, "train/img_b.bmp");
    assert_eq!(dataset.images[1].id.as_u64(), 2);
    assert_eq!(dataset.images[2].file_name, "train/img_c.bmp");
    assert_eq!(dataset.images[2].id.as_u64(), 3);

    // Check one bbox conversion for img_a (20x10).
    // 0.5,0.5 center and 0.4x0.4 size => xmin=6,xmax=14,ymin=3,ymax=7
    let bbox = &dataset.annotations[0].bbox;
    assert!((bbox.xmin() - 6.0).abs() < 1e-6);
    assert!((bbox.ymin() - 3.0).abs() < 1e-6);
    assert!((bbox.xmax() - 14.0).abs() < 1e-6);
    assert!((bbox.ymax() - 7.0).abs() < 1e-6);
}

#[test]
fn read_yolo_from_labels_dir_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_dataset(temp.path());

    let dataset = read_yolo_dir(&temp.path().join("labels")).expect("read yolo dataset");
    assert_eq!(dataset.images.len(), 3);
    assert_eq!(dataset.annotations.len(), 3);
}

#[test]
fn read_yolo_rejects_segmentation_rows() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("images")).expect("create images dir");
    fs::create_dir_all(temp.path().join("labels")).expect("create labels dir");

    write_bmp(&temp.path().join("images/img1.bmp"), 8, 8);
    fs::write(
        temp.path().join("labels/img1.txt"),
        "0 0.1 0.2 0.3 0.4 0.5 0.6\n",
    )
    .expect("write invalid label row");

    let err = read_yolo_dir(temp.path()).unwrap_err();
    match err {
        PanlabelError::YoloLabelParse { message, .. } => {
            assert!(message.contains("segmentation/pose"));
        }
        other => panic!("expected YoloLabelParse, got {other:?}"),
    }
}

#[test]
fn yolo_write_then_read_roundtrip_semantic() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let input_root = temp.path().join("input_yolo");
    let output_root = temp.path().join("output_yolo");

    create_sample_dataset(&input_root);

    let input_dataset = read_yolo_dir(&input_root).expect("read input dataset");
    write_yolo_dir(&output_root, &input_dataset).expect("write yolo dataset");

    // Writer should create empty label file for img_c (no annotations).
    let img_c_label = output_root.join("labels/train/img_c.txt");
    assert!(img_c_label.is_file());
    let img_c_contents = fs::read_to_string(&img_c_label).expect("read img_c label");
    assert!(img_c_contents.is_empty());

    // Writer intentionally does not copy images.
    assert!(output_root.join("images").is_dir());
    assert!(!output_root.join("images/train/img_a.bmp").exists());

    // Copy images in place to allow read-back.
    copy_images_for_reread(&input_root, &output_root);

    let restored = read_yolo_dir(&output_root).expect("read restored dataset");

    assert_eq!(restored.images.len(), input_dataset.images.len());
    assert_eq!(restored.categories.len(), input_dataset.categories.len());
    assert_eq!(restored.annotations.len(), input_dataset.annotations.len());

    for (left, right) in input_dataset
        .annotations
        .iter()
        .zip(restored.annotations.iter())
    {
        assert!((left.bbox.xmin() - right.bbox.xmin()).abs() < 1e-3);
        assert!((left.bbox.ymin() - right.bbox.ymin()).abs() < 1e-3);
        assert!((left.bbox.xmax() - right.bbox.xmax()).abs() < 1e-3);
        assert!((left.bbox.ymax() - right.bbox.ymax()).abs() < 1e-3);
    }
}

#[test]
fn yolo_confidence_roundtrip() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let output_root = temp.path().join("yolo_conf");

    let dataset = Dataset {
        images: vec![Image::new(1u64, "img.bmp", 100, 100)],
        categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
        annotations: vec![
            Annotation::new(1u64, 1u64, 1u64, BBoxXYXY::from_xyxy(10.0, 20.0, 60.0, 70.0))
                .with_confidence(0.95),
            Annotation::new(2u64, 1u64, 2u64, BBoxXYXY::from_xyxy(30.0, 40.0, 80.0, 90.0)),
        ],
        ..Default::default()
    };

    write_yolo_dir(&output_root, &dataset).expect("write yolo");

    // Materialize image for re-read
    write_bmp(&output_root.join("images/img.bmp"), 100, 100);

    let restored = read_yolo_dir(&output_root).expect("read yolo");
    assert_eq!(restored.annotations.len(), 2);

    // First annotation had confidence — should survive roundtrip
    assert!(
        restored.annotations[0].confidence.is_some(),
        "confidence should be preserved"
    );
    assert!(
        (restored.annotations[0].confidence.unwrap() - 0.95).abs() < 1e-6,
        "confidence value should match"
    );

    // Second annotation had no confidence — should remain None
    assert_eq!(restored.annotations[1].confidence, None);
}

#[test]
fn yolo_6_token_row_accepted_7_rejected() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("images")).expect("create images dir");
    fs::create_dir_all(temp.path().join("labels")).expect("create labels dir");

    write_bmp(&temp.path().join("images/ok.bmp"), 10, 10);
    // 6 tokens = valid (bbox + confidence)
    fs::write(
        temp.path().join("labels/ok.txt"),
        "0 0.5 0.5 0.3 0.3 0.99\n",
    )
    .expect("write 6-token label");

    let dataset = read_yolo_dir(temp.path()).expect("6-token row should be accepted");
    assert_eq!(dataset.annotations.len(), 1);
    assert_eq!(dataset.annotations[0].confidence, Some(0.99));

    // Now try 7 tokens — should fail
    let temp2 = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp2.path().join("images")).expect("create images dir");
    fs::create_dir_all(temp2.path().join("labels")).expect("create labels dir");
    write_bmp(&temp2.path().join("images/bad.bmp"), 10, 10);
    fs::write(
        temp2.path().join("labels/bad.txt"),
        "0 0.1 0.2 0.3 0.4 0.5 0.6\n",
    )
    .expect("write 7-token label");

    let err = read_yolo_dir(temp2.path()).unwrap_err();
    match err {
        PanlabelError::YoloLabelParse { message, .. } => {
            assert!(message.contains("segmentation/pose"));
        }
        other => panic!("expected YoloLabelParse, got {other:?}"),
    }
}

#[test]
fn flat_layout_no_data_yaml_with_classes_txt() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("images")).expect("create images dir");
    fs::create_dir_all(temp.path().join("labels")).expect("create labels dir");

    write_bmp(&temp.path().join("images/a.bmp"), 50, 50);
    fs::write(temp.path().join("classes.txt"), "person\ndog\n").expect("write classes.txt");
    fs::write(
        temp.path().join("labels/a.txt"),
        "1 0.5 0.5 0.3 0.3\n",
    )
    .expect("write label");

    let dataset = read_yolo_dir(temp.path()).expect("read flat layout with classes.txt");
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.categories[0].name, "person");
    assert_eq!(dataset.categories[1].name, "dog");
    assert_eq!(dataset.annotations.len(), 1);
}

#[test]
fn flat_layout_no_data_yaml_no_classes_txt() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("images")).expect("create images dir");
    fs::create_dir_all(temp.path().join("labels")).expect("create labels dir");

    write_bmp(&temp.path().join("images/a.bmp"), 50, 50);
    // No data.yaml, no classes.txt
    fs::write(
        temp.path().join("labels/a.txt"),
        "2 0.5 0.5 0.3 0.3\n0 0.2 0.2 0.1 0.1\n",
    )
    .expect("write label");

    let dataset = read_yolo_dir(temp.path()).expect("read flat layout inferred names");
    assert_eq!(dataset.categories.len(), 3);
    assert_eq!(dataset.categories[0].name, "class_0");
    assert_eq!(dataset.categories[1].name, "class_1");
    assert_eq!(dataset.categories[2].name, "class_2");
}
