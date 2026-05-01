mod common;

use std::fs;

use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};
use tempfile::tempdir;

fn sample_dataset() -> Dataset {
    Dataset {
        images: vec![
            Image::new(2u64, "b.bmp", 40, 30),
            Image::new(1u64, "a.bmp", 100, 80),
        ],
        categories: vec![Category::new(2u64, "dog"), Category::new(1u64, "cat")],
        annotations: vec![
            Annotation::new(
                2u64,
                2u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(5.0, 6.0, 25.0, 26.0),
            ),
            Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 12.0, 30.0, 32.0),
            )
            .with_confidence(0.7),
        ],
        ..Default::default()
    }
}

fn assert_counts_and_bbox(dataset: &Dataset) {
    assert_eq!(dataset.images.len(), 2);
    assert_eq!(dataset.annotations.len(), 2);
    assert!(dataset
        .annotations
        .iter()
        .any(|a| (a.bbox.xmin() - 10.0).abs() < 1e-9 && (a.bbox.ymax() - 32.0).abs() < 1e-9));
}

#[test]
fn datumaro_json_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("datumaro.json");
    panlabel::ir::io_datumaro_json::write_datumaro_json(&path, &sample_dataset()).unwrap();
    let read = panlabel::ir::io_datumaro_json::read_datumaro_json(&path).unwrap();
    assert_counts_and_bbox(&read);
}

#[test]
fn bdd100k_json_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bdd.json");
    panlabel::ir::io_bdd100k_json::write_bdd100k_json(&path, &sample_dataset()).unwrap();
    let read = panlabel::ir::io_bdd100k_json::read_bdd100k_json(&path).unwrap();
    assert_counts_and_bbox(&read);
}

#[test]
fn v7_darwin_json_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("darwin.json");
    panlabel::ir::io_v7_darwin_json::write_v7_darwin_json(&path, &sample_dataset()).unwrap();
    let read = panlabel::ir::io_v7_darwin_json::read_v7_darwin_json(&path).unwrap();
    assert_counts_and_bbox(&read);
}

#[test]
fn edge_impulse_labels_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path();
    panlabel::ir::io_edge_impulse_labels::write_edge_impulse_labels(path, &sample_dataset())
        .unwrap();
    let read = panlabel::ir::io_edge_impulse_labels::read_edge_impulse_labels(path).unwrap();
    assert_counts_and_bbox(&read);
}

#[test]
fn openlabel_json_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("openlabel.json");
    panlabel::ir::io_openlabel_json::write_openlabel_json(&path, &sample_dataset()).unwrap();
    let read = panlabel::ir::io_openlabel_json::read_openlabel_json(&path).unwrap();
    assert_counts_and_bbox(&read);
}

#[test]
fn via_csv_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("via.csv");
    panlabel::ir::io_via_csv::write_via_csv(&path, &sample_dataset()).unwrap();
    common::write_bmp(&dir.path().join("a.bmp"), 100, 80);
    common::write_bmp(&dir.path().join("b.bmp"), 40, 30);
    let read = panlabel::ir::io_via_csv::read_via_csv(&path).unwrap();
    assert_counts_and_bbox(&read);
}

#[test]
fn wider_face_txt_roundtrip_collapses_to_face() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wider.txt");
    panlabel::ir::io_wider_face_txt::write_wider_face_txt(&path, &sample_dataset()).unwrap();
    common::write_bmp(&dir.path().join("a.bmp"), 100, 80);
    common::write_bmp(&dir.path().join("b.bmp"), 40, 30);
    let read = panlabel::ir::io_wider_face_txt::read_wider_face_txt(&path).unwrap();
    assert_eq!(read.images.len(), 2);
    assert_eq!(read.categories.len(), 1);
    assert_eq!(read.categories[0].name, "face");
    assert_eq!(read.annotations.len(), 2);
}

#[test]
fn oidv4_txt_directory_roundtrip() {
    let dir = tempdir().unwrap();
    let out = dir.path().join("oid");
    panlabel::ir::io_oidv4_txt::write_oidv4_txt(&out, &sample_dataset()).unwrap();
    fs::write(out.join("images").join("README.txt"), "placeholder").unwrap();
    common::write_bmp(&out.join("a.bmp"), 100, 80);
    common::write_bmp(&out.join("b.bmp"), 40, 30);
    let read = panlabel::ir::io_oidv4_txt::read_oidv4_txt(&out).unwrap();
    assert_counts_and_bbox(&read);
}
