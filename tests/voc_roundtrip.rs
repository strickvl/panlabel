//! Integration tests for Pascal VOC format support.

use std::fs;
use std::path::Path;

use panlabel::ir::io_voc_xml::{read_voc_dir, write_voc_dir};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image};

fn create_sample_voc_dataset(root: &Path) {
    fs::create_dir_all(root.join("Annotations")).expect("create annotations dir");
    fs::create_dir_all(root.join("JPEGImages")).expect("create images dir");

    let xml_a = r#"<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>img_b.jpg</filename>
  <size>
    <width>120</width>
    <height>80</height>
    <depth>1</depth>
  </size>
  <object>
    <name>dog</name>
    <truncated>yes</truncated>
    <bndbox>
      <xmin>10</xmin>
      <ymin>12</ymin>
      <xmax>60</xmax>
      <ymax>70</ymax>
    </bndbox>
  </object>
</annotation>
"#;

    let xml_b = r#"<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>img_a.jpg</filename>
  <size>
    <width>100</width>
    <height>50</height>
    <depth>3</depth>
  </size>
  <object>
    <name>cat</name>
    <pose>Sitting</pose>
    <difficult>0</difficult>
    <bndbox>
      <xmin>1</xmin>
      <ymin>2</ymin>
      <xmax>30</xmax>
      <ymax>40</ymax>
    </bndbox>
  </object>
  <object>
    <name>dog</name>
    <occluded>1</occluded>
    <bndbox>
      <xmin>31</xmin>
      <ymin>4</ymin>
      <xmax>80</xmax>
      <ymax>45</ymax>
    </bndbox>
  </object>
</annotation>
"#;

    let xml_c = r#"<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>img_c.jpg</filename>
  <size>
    <width>64</width>
    <height>64</height>
    <depth>3</depth>
  </size>
</annotation>
"#;

    fs::write(root.join("Annotations/a.xml"), xml_a).expect("write a.xml");
    fs::write(root.join("Annotations/b.xml"), xml_b).expect("write b.xml");
    fs::write(root.join("Annotations/c.xml"), xml_c).expect("write c.xml");
}

#[test]
fn read_voc_assigns_deterministic_ids_and_preserves_fields() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_voc_dataset(temp.path());

    let dataset = read_voc_dir(temp.path()).expect("read voc dataset");

    assert_eq!(dataset.images.len(), 3);
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.annotations.len(), 3);

    assert_eq!(dataset.images[0].file_name, "img_a.jpg");
    assert_eq!(dataset.images[0].id.as_u64(), 1);
    assert_eq!(dataset.images[1].file_name, "img_b.jpg");
    assert_eq!(dataset.images[1].id.as_u64(), 2);
    assert_eq!(dataset.images[2].file_name, "img_c.jpg");
    assert_eq!(dataset.images[2].id.as_u64(), 3);

    assert_eq!(
        dataset.images[1].attributes.get("depth"),
        Some(&"1".to_string())
    );

    assert_eq!(dataset.categories[0].name, "cat");
    assert_eq!(dataset.categories[0].id.as_u64(), 1);
    assert_eq!(dataset.categories[1].name, "dog");
    assert_eq!(dataset.categories[1].id.as_u64(), 2);

    // Annotation IDs: file order (a.xml then b.xml), then object order.
    assert_eq!(dataset.annotations[0].id.as_u64(), 1);
    assert_eq!(dataset.annotations[0].image_id.as_u64(), 2); // img_b.jpg
    assert_eq!(dataset.annotations[0].category_id.as_u64(), 2); // dog

    assert_eq!(dataset.annotations[1].id.as_u64(), 2);
    assert_eq!(dataset.annotations[1].image_id.as_u64(), 1); // img_a.jpg
    assert_eq!(dataset.annotations[1].category_id.as_u64(), 1); // cat
    assert_eq!(
        dataset.annotations[1].attributes.get("pose"),
        Some(&"Sitting".to_string())
    );

    // Coordinate policy: raw values are read as-is.
    let bbox = &dataset.annotations[1].bbox;
    assert!((bbox.xmin() - 1.0).abs() < 1e-9);
    assert!((bbox.ymin() - 2.0).abs() < 1e-9);
    assert!((bbox.xmax() - 30.0).abs() < 1e-9);
    assert!((bbox.ymax() - 40.0).abs() < 1e-9);
}

#[test]
fn voc_write_then_read_roundtrip_semantic() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let input_root = temp.path().join("input_voc");
    let output_root = temp.path().join("output_voc");

    create_sample_voc_dataset(&input_root);

    let input_dataset = read_voc_dir(&input_root).expect("read input dataset");
    write_voc_dir(&output_root, &input_dataset).expect("write voc dataset");

    assert!(output_root.join("Annotations").is_dir());
    assert!(output_root.join("JPEGImages").is_dir());
    assert!(output_root.join("JPEGImages/README.txt").is_file());

    let restored = read_voc_dir(&output_root).expect("read restored dataset");

    assert_eq!(restored.images.len(), input_dataset.images.len());
    assert_eq!(restored.categories.len(), input_dataset.categories.len());
    assert_eq!(restored.annotations.len(), input_dataset.annotations.len());

    let mut left_annotations: Vec<_> = input_dataset
        .annotations
        .iter()
        .map(|ann| {
            (
                ann.image_id.as_u64(),
                ann.category_id.as_u64(),
                ann.bbox.xmin(),
                ann.bbox.ymin(),
                ann.bbox.xmax(),
                ann.bbox.ymax(),
            )
        })
        .collect();
    left_annotations.sort_by(|a, b| a.partial_cmp(b).expect("finite bbox values"));

    let mut right_annotations: Vec<_> = restored
        .annotations
        .iter()
        .map(|ann| {
            (
                ann.image_id.as_u64(),
                ann.category_id.as_u64(),
                ann.bbox.xmin(),
                ann.bbox.ymin(),
                ann.bbox.xmax(),
                ann.bbox.ymax(),
            )
        })
        .collect();
    right_annotations.sort_by(|a, b| a.partial_cmp(b).expect("finite bbox values"));

    assert_eq!(left_annotations, right_annotations);
}

#[test]
fn write_voc_preserves_subdirectory_structure() {
    let temp = tempfile::tempdir().expect("create temp dir");

    let dataset = Dataset {
        images: vec![Image::new(1u64, "train/img1.jpg", 32, 32)],
        categories: vec![Category::new(1u64, "cat")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::from_xyxy(1.0, 2.0, 20.0, 25.0),
        )],
        ..Default::default()
    };

    write_voc_dir(temp.path(), &dataset).expect("write voc");

    assert!(temp.path().join("Annotations/train/img1.xml").is_file());
}

#[test]
fn write_voc_normalizes_boolean_attributes() {
    let temp = tempfile::tempdir().expect("create temp dir");

    let mut annotation = Annotation::new(1u64, 1u64, 1u64, BBoxXYXY::from_xyxy(1.0, 2.0, 3.0, 4.0));
    annotation
        .attributes
        .insert("truncated".to_string(), "true".to_string());
    annotation
        .attributes
        .insert("difficult".to_string(), "no".to_string());
    annotation
        .attributes
        .insert("occluded".to_string(), "maybe".to_string());

    let dataset = Dataset {
        images: vec![Image::new(1u64, "img_bool.jpg", 10, 10)],
        categories: vec![Category::new(1u64, "cat")],
        annotations: vec![annotation],
        ..Default::default()
    };

    write_voc_dir(temp.path(), &dataset).expect("write voc");

    let xml =
        fs::read_to_string(temp.path().join("Annotations/img_bool.xml")).expect("read written xml");
    assert!(xml.contains("<truncated>1</truncated>"));
    assert!(xml.contains("<difficult>0</difficult>"));
    assert!(!xml.contains("<occluded>"));
}

#[test]
fn read_voc_from_annotations_dir_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    create_sample_voc_dataset(temp.path());

    let dataset =
        read_voc_dir(&temp.path().join("Annotations")).expect("read voc from annotations");
    assert_eq!(dataset.images.len(), 3);
    assert_eq!(dataset.annotations.len(), 3);
}
