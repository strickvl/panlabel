mod common;

use std::fs;

use panlabel::ir::{io_marmot_xml, Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

fn hex(value: f64) -> String {
    value
        .to_be_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn rect(x_left: f64, y_top: f64, x_right: f64, y_bottom: f64) -> String {
    [x_left, y_top, x_right, y_bottom]
        .into_iter()
        .map(hex)
        .collect::<Vec<_>>()
        .join(" ")
}

fn sample_marmot_xml() -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Page CropBox="{}" PageNum="1">
  <Contents>
    <Leafs Label="Char">
      <Leaf BBox="{}" Label="Char" />
    </Leafs>
    <Composite BBox="{}" Label="Outside" />
    <Composites Label="Figure">
      <Composite BBox="{}" LID="7" Label="Figure" />
    </Composites>
    <Composites Label="TableBody">
      <Composite BBox="{}" LID="8" />
    </Composites>
  </Contents>
</Page>
"#,
        rect(0.0, 100.0, 200.0, 0.0),
        rect(1.0, 99.0, 2.0, 98.0),
        rect(0.0, 100.0, 10.0, 90.0),
        rect(20.0, 80.0, 120.0, 30.0),
        rect(50.0, 60.0, 150.0, 10.0),
    )
}

#[test]
fn reads_composites_under_composites_and_flips_y_axis() {
    let temp = tempfile::tempdir().expect("tempdir");
    let xml_path = temp.path().join("page1.xml");
    fs::write(&xml_path, sample_marmot_xml()).expect("write xml");
    common::write_bmp(&temp.path().join("page1.bmp"), 200, 100);

    let dataset = io_marmot_xml::read_marmot_xml(&xml_path).expect("read marmot");
    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "page1.bmp");
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.annotations.len(), 2);

    let figure = dataset
        .annotations
        .iter()
        .find(|ann| ann.attributes.get("marmot_lid").map(String::as_str) == Some("7"))
        .expect("figure annotation");
    assert_eq!(figure.bbox.xmin(), 20.0);
    assert_eq!(figure.bbox.ymin(), 20.0);
    assert_eq!(figure.bbox.xmax(), 120.0);
    assert_eq!(figure.bbox.ymax(), 70.0);
}

#[test]
fn reads_directory_and_requires_companion_images() {
    let temp = tempfile::tempdir().expect("tempdir");
    fs::write(temp.path().join("page1.xml"), sample_marmot_xml()).expect("write xml");
    common::write_bmp(&temp.path().join("page1.bmp"), 200, 100);

    let dataset = io_marmot_xml::read_marmot_xml(temp.path()).expect("read directory");
    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.annotations.len(), 2);

    let missing = tempfile::tempdir().expect("missing tempdir");
    fs::write(missing.path().join("page1.xml"), sample_marmot_xml()).expect("write xml");
    let err = io_marmot_xml::read_marmot_xml(missing.path())
        .unwrap_err()
        .to_string();
    assert!(err.contains("companion image not found"));
}

#[test]
fn writer_emits_deterministic_minimal_xml() {
    let dataset = Dataset {
        images: vec![Image::new(1u64, "page1.bmp", 200, 100)],
        categories: vec![Category::new(1u64, "TableBody")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(20.0, 20.0, 120.0, 70.0),
        )],
        ..Default::default()
    };

    let xml = io_marmot_xml::to_marmot_xml_string(&dataset).expect("write xml string");
    assert!(xml.contains(
        "<Page CropBox=\"0000000000000000 4059000000000000 4069000000000000 0000000000000000\">"
    ));
    assert!(xml.contains("<Composites Label=\"TableBody\">"));
    assert!(xml.contains("<Composite BBox=\"4034000000000000 4054000000000000 405e000000000000 403e000000000000\" LID=\"1\" Label=\"TableBody\" />"));
}

#[test]
fn directory_read_reports_malformed_page_cropbox() {
    let temp = tempfile::tempdir().expect("tempdir");
    fs::write(
        temp.path().join("page1.xml"),
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Page CropBox="not-hex">
  <Composites Label="TableBody" />
</Page>
"#,
    )
    .expect("write malformed xml");
    common::write_bmp(&temp.path().join("page1.bmp"), 200, 100);

    let err = io_marmot_xml::read_marmot_xml(temp.path())
        .unwrap_err()
        .to_string();
    assert!(err.contains("must be exactly 16 hexadecimal characters"));
}

#[test]
fn directory_read_does_not_skip_malformed_page_when_valid_page_exists() {
    let temp = tempfile::tempdir().expect("tempdir");
    fs::write(temp.path().join("page1.xml"), sample_marmot_xml()).expect("write valid xml");
    common::write_bmp(&temp.path().join("page1.bmp"), 200, 100);
    fs::write(
        temp.path().join("page2.xml"),
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Page CropBox="not-hex">
  <Composites Label="TableBody" />
</Page>
"#,
    )
    .expect("write malformed xml");
    common::write_bmp(&temp.path().join("page2.bmp"), 200, 100);

    let err = io_marmot_xml::read_marmot_xml(temp.path())
        .unwrap_err()
        .to_string();
    assert!(err.contains("must be exactly 16 hexadecimal characters"));
}

#[test]
fn directory_writer_rejects_unsafe_image_paths() {
    let temp = tempfile::tempdir().expect("tempdir");
    let dataset = Dataset {
        images: vec![Image::new(1u64, "../escape.bmp", 200, 100)],
        categories: vec![Category::new(1u64, "TableBody")],
        annotations: vec![],
        ..Default::default()
    };

    let err = io_marmot_xml::write_marmot_xml(temp.path(), &dataset)
        .unwrap_err()
        .to_string();
    assert!(err.contains("safe relative path"));
    assert!(!temp.path().parent().unwrap().join("escape.xml").exists());
}
