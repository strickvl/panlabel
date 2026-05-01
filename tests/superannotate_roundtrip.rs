use std::fs;

use panlabel::ir::io_superannotate_json::{
    from_superannotate_str, read_superannotate_json, to_superannotate_string,
    write_superannotate_json, ATTR_GEOMETRY_TYPE, ATTR_INSTANCE_ID,
};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

fn assert_bbox(bbox: BBoxXYXY<Pixel>, expected: (f64, f64, f64, f64)) {
    assert_eq!(bbox.xmin(), expected.0);
    assert_eq!(bbox.ymin(), expected.1);
    assert_eq!(bbox.xmax(), expected.2);
    assert_eq!(bbox.ymax(), expected.3);
}

fn category_names(dataset: &Dataset) -> Vec<&str> {
    dataset
        .categories
        .iter()
        .map(|category| category.name.as_str())
        .collect()
}

#[test]
fn parses_fixture_with_bbox_polygon_and_oriented_envelopes() {
    let dataset =
        read_superannotate_json("tests/fixtures/sample_valid.superannotate.json".as_ref())
            .expect("parse fixture");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "sa_image.jpg");
    assert_eq!(dataset.images[0].width, 640);
    assert_eq!(dataset.images[0].height, 480);
    assert_eq!(category_names(&dataset), vec!["car", "cat", "dog"]);
    assert_eq!(dataset.annotations.len(), 3);

    assert_bbox(dataset.annotations[0].bbox, (10.0, 20.0, 100.0, 80.0));
    assert_eq!(dataset.annotations[0].confidence, Some(0.91));
    assert_eq!(
        dataset.annotations[0].attributes.get(ATTR_GEOMETRY_TYPE),
        Some(&"bbox".to_string())
    );
    assert_eq!(
        dataset.annotations[0].attributes.get(ATTR_INSTANCE_ID),
        Some(&"101".to_string())
    );

    assert_bbox(dataset.annotations[1].bbox, (50.0, 60.0, 200.0, 210.0));
    assert_eq!(
        dataset.annotations[1].attributes.get(ATTR_GEOMETRY_TYPE),
        Some(&"polygon".to_string())
    );

    assert_bbox(dataset.annotations[2].bbox, (280.0, 100.0, 360.0, 180.0));
    assert_eq!(
        dataset.annotations[2].attributes.get(ATTR_GEOMETRY_TYPE),
        Some(&"rotated_bbox".to_string())
    );
}

#[test]
fn write_read_single_file_semantic_roundtrip() {
    let dataset = Dataset {
        images: vec![Image::new(1u64, "test.jpg", 640, 480)],
        categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
        annotations: vec![
            Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 80.0),
            )
            .with_confidence(0.7),
            Annotation::new(
                2u64,
                1u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 200.0, 210.0),
            ),
        ],
        ..Default::default()
    };

    let json = to_superannotate_string(&dataset).expect("serialize");
    let restored = from_superannotate_str(&json).expect("parse");

    assert_eq!(restored.images.len(), 1);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
    assert_bbox(restored.annotations[0].bbox, (10.0, 20.0, 100.0, 80.0));
    assert_eq!(restored.annotations[0].confidence, Some(0.7));
    assert_bbox(restored.annotations[1].bbox, (50.0, 60.0, 200.0, 210.0));
}

#[test]
fn direct_bbox_aliases_normalize_geometry_type_to_bbox() {
    let json = r#"{
        "metadata": {"name": "alias.jpg", "width": 100, "height": 100},
        "instances": [
            {
                "type": "bounding_box",
                "className": "cat",
                "points": {"x1": 1, "y1": 2, "x2": 30, "y2": 40}
            },
            {
                "type": "rectangle",
                "className": "dog",
                "points": {"x1": 5, "y1": 6, "x2": 50, "y2": 60}
            }
        ]
    }"#;

    let dataset = from_superannotate_str(json).expect("parse");
    assert_eq!(
        dataset.annotations[0].attributes.get(ATTR_GEOMETRY_TYPE),
        Some(&"bbox".to_string())
    );
    assert_eq!(
        dataset.annotations[1].attributes.get(ATTR_GEOMETRY_TYPE),
        Some(&"bbox".to_string())
    );
}

#[test]
fn rejects_unsupported_geometry() {
    let json = r#"{
        "metadata": {"name": "bad.jpg", "width": 100, "height": 100},
        "instances": [
            {"type": "cuboid", "className": "box"}
        ]
    }"#;

    let err = from_superannotate_str(json).expect_err("unsupported geometry should fail");
    let message = err.to_string();
    assert!(message.contains("cuboid"), "{message}");
    assert!(
        message.contains("unsupported SuperAnnotate geometry"),
        "{message}"
    );
}

#[test]
fn writer_rejects_unsafe_image_file_name() {
    let dataset = Dataset {
        images: vec![Image::new(1u64, "../escape.jpg", 100, 100)],
        categories: vec![Category::new(1u64, "cat")],
        annotations: vec![],
        ..Default::default()
    };

    let temp = tempfile::tempdir().expect("create temp dir");
    let err = write_superannotate_json(temp.path(), &dataset).expect_err("unsafe path should fail");
    let message = err.to_string();
    assert!(message.contains("unsafe image file_name"), "{message}");
}

#[test]
fn to_string_rejects_annotation_with_missing_image() {
    let dataset = Dataset {
        images: vec![Image::new(1u64, "ok.jpg", 100, 100)],
        categories: vec![Category::new(1u64, "cat")],
        annotations: vec![Annotation::new(
            1u64,
            999u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
        )],
        ..Default::default()
    };

    let err = to_superannotate_string(&dataset).expect_err("missing image should fail");
    let message = err.to_string();
    assert!(message.contains("references missing image"), "{message}");
}

#[test]
fn writer_rejects_annotation_with_missing_image() {
    let dataset = Dataset {
        images: vec![Image::new(1u64, "ok.jpg", 100, 100)],
        categories: vec![Category::new(1u64, "cat")],
        annotations: vec![Annotation::new(
            1u64,
            999u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
        )],
        ..Default::default()
    };

    let temp = tempfile::tempdir().expect("create temp dir");
    let err =
        write_superannotate_json(temp.path(), &dataset).expect_err("missing image should fail");
    let message = err.to_string();
    assert!(message.contains("references missing image"), "{message}");
}

#[test]
fn reads_directory_with_class_metadata() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("classes")).expect("create classes dir");
    fs::write(
        temp.path().join("classes/classes.json"),
        r#"[{"name": "cat"}, {"name": "unused"}]"#,
    )
    .expect("write classes");
    fs::write(
        temp.path().join("example.json"),
        r#"{
            "metadata": {"name": "example.jpg", "width": 320, "height": 240},
            "instances": [
                {
                    "type": "bbox",
                    "className": "cat",
                    "points": {"x1": 1, "y1": 2, "x2": 30, "y2": 40}
                }
            ]
        }"#,
    )
    .expect("write annotation");

    let dataset = read_superannotate_json(temp.path()).expect("read directory");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "example.jpg");
    assert_eq!(category_names(&dataset), vec!["cat", "unused"]);
    assert_eq!(dataset.annotations.len(), 1);
    assert_bbox(dataset.annotations[0].bbox, (1.0, 2.0, 30.0, 40.0));
}

#[test]
fn writes_and_reads_canonical_directory() {
    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "b.jpg", 800, 600),
            Image::new(2u64, "a.jpg", 640, 480),
        ],
        categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "unused")],
        annotations: vec![Annotation::new(
            1u64,
            2u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(5.0, 6.0, 50.0, 60.0),
        )],
        ..Default::default()
    };

    let temp = tempfile::tempdir().expect("create temp dir");
    write_superannotate_json(temp.path(), &dataset).expect("write directory");

    assert!(temp.path().join("annotations/a.json").is_file());
    assert!(temp.path().join("annotations/b.json").is_file());
    assert!(temp.path().join("classes/classes.json").is_file());
    assert!(temp.path().join("images/README.txt").is_file());

    let restored = read_superannotate_json(temp.path()).expect("read directory");
    assert_eq!(restored.images[0].file_name, "a.jpg");
    assert_eq!(restored.images[1].file_name, "b.jpg");
    assert_eq!(category_names(&restored), vec!["cat", "unused"]);
    assert_eq!(restored.annotations.len(), 1);
    assert_bbox(restored.annotations[0].bbox, (5.0, 6.0, 50.0, 60.0));
}
