//! Integration tests for shared YOLO Keras / YOLOv4 PyTorch TXT support.

mod common;

use common::write_bmp;
use panlabel::ir::io_yolo_keras_txt::{
    read_yolo_keras_txt, read_yolov4_pytorch_txt, write_yolo_keras_txt, write_yolov4_pytorch_txt,
};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

#[test]
fn yolo_keras_write_then_read_roundtrip_preserves_boxes_and_empty_rows() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let txt_path = temp.path().join("yolo_keras.txt");

    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "img1.bmp", 640, 480),
            Image::new(2u64, "empty.bmp", 320, 240),
        ],
        categories: vec![Category::new(1u64, "car"), Category::new(2u64, "person")],
        annotations: vec![
            Annotation::new(
                1u64,
                1u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(50.0, 30.0, 200.0, 180.0),
            ),
            Annotation::new(
                2u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 40.0, 60.0),
            ),
        ],
        ..Default::default()
    };

    write_yolo_keras_txt(&txt_path, &dataset).expect("write yolo keras txt");
    let txt = std::fs::read_to_string(&txt_path).expect("read txt");
    assert_eq!(txt, "empty.bmp\nimg1.bmp 50,30,200,180,1 10,20,40,60,0\n");

    write_bmp(&temp.path().join("img1.bmp"), 640, 480);
    write_bmp(&temp.path().join("empty.bmp"), 320, 240);

    let restored = read_yolo_keras_txt(&txt_path).expect("read yolo keras txt");
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
    assert!(restored
        .images
        .iter()
        .any(|image| image.file_name == "empty.bmp"));
    assert_eq!(
        restored
            .categories
            .iter()
            .map(|category| category.name.as_str())
            .collect::<Vec<_>>(),
        vec!["car", "person"]
    );
}

#[test]
fn yolov4_pytorch_directory_input_uses_canonical_files_and_images_subdir() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let root = temp.path();
    std::fs::write(
        root.join("yolov4_pytorch.txt"),
        "images/nested/img.bmp 1,2,30,40,0\nempty.bmp\n",
    )
    .expect("write annotations");
    std::fs::write(root.join("classes.txt"), "person\n").expect("write classes");
    write_bmp(&root.join("images/nested/img.bmp"), 100, 80);
    write_bmp(&root.join("images/empty.bmp"), 10, 20);

    let dataset = read_yolov4_pytorch_txt(root).expect("read directory");
    assert_eq!(dataset.images.len(), 2);
    assert_eq!(dataset.annotations.len(), 1);
    let nested = dataset
        .images
        .iter()
        .find(|image| image.file_name == "images/nested/img.bmp")
        .expect("nested image");
    assert_eq!(nested.width, 100);
    assert_eq!(nested.height, 80);
    let empty = dataset
        .images
        .iter()
        .find(|image| image.file_name == "empty.bmp")
        .expect("empty image");
    assert_eq!(empty.width, 10);
    assert_eq!(empty.height, 20);
}

#[test]
fn yolov4_pytorch_writer_uses_canonical_directory_output() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let dataset = Dataset {
        images: vec![Image::new(1u64, "img.bmp", 10, 10)],
        categories: vec![Category::new(1u64, "object")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 5.0, 6.0),
        )],
        ..Default::default()
    };

    write_yolov4_pytorch_txt(temp.path(), &dataset).expect("write directory");
    assert_eq!(
        std::fs::read_to_string(temp.path().join("yolov4_pytorch.txt")).expect("txt"),
        "img.bmp 0,0,5,6,0\n"
    );
    assert_eq!(
        std::fs::read_to_string(temp.path().join("classes.txt")).expect("classes"),
        "object\n"
    );
}

#[test]
fn class_file_blank_lines_preserve_class_indices() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let txt_path = temp.path().join("yolo_keras.txt");
    std::fs::write(&txt_path, "img.bmp 1,2,3,4,2\n").expect("write annotations");
    std::fs::write(temp.path().join("classes.txt"), "cat\n\ndog\n").expect("write classes");
    write_bmp(&temp.path().join("img.bmp"), 20, 20);

    let dataset = read_yolo_keras_txt(&txt_path).expect("read");
    assert_eq!(
        dataset
            .categories
            .iter()
            .map(|category| category.name.as_str())
            .collect::<Vec<_>>(),
        vec!["cat", "class_1", "dog"]
    );
}

#[test]
fn writer_rejects_unrepresentable_image_refs() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let dataset = Dataset {
        images: vec![Image::new(1u64, "my image.bmp", 10, 10)],
        categories: vec![],
        annotations: vec![],
        ..Default::default()
    };

    let err = write_yolo_keras_txt(temp.path(), &dataset)
        .unwrap_err()
        .to_string();
    assert!(err.contains("cannot be represented"));
    assert!(err.contains("whitespace/commas"));
}

#[test]
fn missing_image_error_names_image_ref() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let txt_path = temp.path().join("yolo_keras.txt");
    std::fs::write(&txt_path, "missing.bmp 1,2,3,4,0\n").expect("write annotations");

    let err = read_yolo_keras_txt(&txt_path).unwrap_err().to_string();
    assert!(err.contains("missing.bmp"));
    assert!(err.contains("image not found"));
}

#[test]
fn malformed_box_error_is_line_specific() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let txt_path = temp.path().join("yolo_keras.txt");
    std::fs::write(&txt_path, "img.bmp 10,2,3,4,0\n").expect("write annotations");
    write_bmp(&temp.path().join("img.bmp"), 20, 20);

    let err = read_yolo_keras_txt(&txt_path).unwrap_err().to_string();
    assert!(err.contains("yolo_keras.txt:1"));
    assert!(err.contains("malformed box"));
}
