//! Integration tests for Microsoft VoTT CSV support.

mod common;

use common::write_bmp;
use panlabel::ir::io_vott_csv::{read_vott_csv, write_vott_csv};
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
                2u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 400.0, 300.0),
            ),
        ],
        ..Default::default()
    };

    write_vott_csv(&csv_path, &dataset).expect("write");
    let csv = std::fs::read_to_string(&csv_path).expect("read csv");
    assert!(csv.starts_with("image,xmin,ymin,xmax,ymax,label\n"));

    write_bmp(&temp.path().join("img1.bmp"), 640, 480);
    write_bmp(&temp.path().join("img2.bmp"), 800, 600);

    let restored = read_vott_csv(&csv_path).expect("read");
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
    assert_eq!(restored.images[0].file_name, "img1.bmp");
    assert_eq!(restored.images[0].width, 640);
    assert_eq!(restored.images[1].height, 600);

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
fn reads_images_from_images_subdir() {
    let temp = tempfile::tempdir().expect("create temp dir");
    write_bmp(&temp.path().join("images/img.bmp"), 320, 240);
    std::fs::write(
        temp.path().join("annotations.csv"),
        "image,xmin,ymin,xmax,ymax,label\nimg.bmp,1,2,30,40,obj\n",
    )
    .unwrap();

    let dataset = read_vott_csv(&temp.path().join("annotations.csv")).expect("read");
    assert_eq!(dataset.images[0].width, 320);
    assert_eq!(dataset.images[0].height, 240);
}

#[test]
fn headerless_six_column_csv_is_rejected_by_explicit_reader() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let csv_path = temp.path().join("annotations.csv");
    std::fs::write(&csv_path, "img.bmp,1,2,3,4,obj\n").unwrap();

    let err = read_vott_csv(&csv_path).unwrap_err().to_string();
    assert!(err.contains("expected header"));
}
