use std::fs;

use panlabel::ir::io_cityscapes_json::{
    from_cityscapes_str, read_cityscapes_json, to_cityscapes_string, write_cityscapes_json,
    ATTR_BBOX_SOURCE, ATTR_CITY, ATTR_IS_GROUP, ATTR_LABEL_STATUS, ATTR_ORIGINAL_LABEL, ATTR_SPLIT,
    BBOX_SOURCE_POLYGON_ENVELOPE,
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
fn parses_fixture_with_polygon_envelopes_and_skip_policies() {
    let dataset = read_cityscapes_json("tests/fixtures/sample_valid.cityscapes.json".as_ref())
        .expect("parse fixture");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(
        dataset.images[0].file_name,
        "sample_valid.cityscapes_leftImg8bit.png"
    );
    assert_eq!(dataset.images[0].width, 640);
    assert_eq!(dataset.images[0].height, 480);
    assert_eq!(
        category_names(&dataset),
        vec!["car", "mystery-object", "person"]
    );
    assert_eq!(dataset.annotations.len(), 3);

    assert_bbox(dataset.annotations[0].bbox, (10.0, 20.0, 100.0, 90.0));
    assert_eq!(
        dataset.annotations[0].attributes.get(ATTR_ORIGINAL_LABEL),
        Some(&"car".to_string())
    );
    assert_eq!(
        dataset.annotations[0].attributes.get(ATTR_BBOX_SOURCE),
        Some(&BBOX_SOURCE_POLYGON_ENVELOPE.to_string())
    );

    assert_bbox(dataset.annotations[1].bbox, (200.0, 100.0, 250.0, 180.0));
    assert_eq!(
        dataset.annotations[1].attributes.get(ATTR_ORIGINAL_LABEL),
        Some(&"persongroup".to_string())
    );
    assert_eq!(
        dataset.annotations[1].attributes.get(ATTR_IS_GROUP),
        Some(&"true".to_string())
    );

    assert_bbox(dataset.annotations[2].bbox, (300.0, 200.0, 360.0, 260.0));
    assert_eq!(
        dataset.annotations[2].attributes.get(ATTR_LABEL_STATUS),
        Some(&"unknown".to_string())
    );
}

#[test]
fn write_read_single_file_semantic_roundtrip() {
    let dataset = Dataset {
        images: vec![Image::new(
            1u64,
            "aachen_000001_000019_leftImg8bit.png",
            640,
            480,
        )],
        categories: vec![Category::new(1u64, "car"), Category::new(2u64, "person")],
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
                BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 200.0, 210.0),
            ),
        ],
        ..Default::default()
    };

    let json = to_cityscapes_string(&dataset).expect("serialize");
    let value: serde_json::Value = serde_json::from_str(&json).expect("parse output json");
    let objects = value["objects"].as_array().expect("objects array");
    assert_eq!(
        objects[0]["polygon"],
        serde_json::json!([[10.0, 20.0], [100.0, 20.0], [100.0, 80.0], [10.0, 80.0]])
    );

    let restored = from_cityscapes_str(&json).expect("parse");
    assert_eq!(restored.images.len(), 1);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
    assert_bbox(restored.annotations[0].bbox, (10.0, 20.0, 100.0, 80.0));
    assert_bbox(restored.annotations[1].bbox, (50.0, 60.0, 200.0, 210.0));
}

#[test]
fn reads_dataset_root_gtfine_layout() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ann_dir = temp.path().join("gtFine/train/aachen");
    fs::create_dir_all(&ann_dir).expect("create ann dir");
    fs::write(
        ann_dir.join("aachen_000001_000019_gtFine_polygons.json"),
        r#"{
            "imgHeight": 256,
            "imgWidth": 512,
            "objects": [
                {"label": "car", "polygon": [[1, 2], [30, 4], [10, 40]]}
            ]
        }"#,
    )
    .expect("write annotation");

    let dataset = read_cityscapes_json(temp.path()).expect("read dataset root");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(
        dataset.images[0].file_name,
        "leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png"
    );
    assert_eq!(
        dataset.images[0].attributes.get(ATTR_SPLIT),
        Some(&"train".to_string())
    );
    assert_eq!(
        dataset.images[0].attributes.get(ATTR_CITY),
        Some(&"aachen".to_string())
    );
    assert_bbox(dataset.annotations[0].bbox, (1.0, 2.0, 30.0, 40.0));
}

#[test]
fn writes_canonical_gtfine_directory_without_images() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let mut image = Image::new(
        1u64,
        "leftImg8bit/val/bochum/bochum_000001_000019_leftImg8bit.png",
        320,
        240,
    );
    image
        .attributes
        .insert(ATTR_SPLIT.to_string(), "val".to_string());
    image
        .attributes
        .insert(ATTR_CITY.to_string(), "bochum".to_string());
    let dataset = Dataset {
        images: vec![image],
        categories: vec![Category::new(1u64, "car")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 80.0),
        )],
        ..Default::default()
    };

    write_cityscapes_json(temp.path(), &dataset).expect("write directory");

    let ann_path = temp
        .path()
        .join("gtFine/val/bochum/bochum_000001_000019_gtFine_polygons.json");
    assert!(ann_path.is_file());
    assert!(temp.path().join("leftImg8bit/README.txt").is_file());

    let restored = read_cityscapes_json(temp.path()).expect("read written directory");
    assert_eq!(restored.annotations.len(), 1);
    assert_bbox(restored.annotations[0].bbox, (10.0, 20.0, 100.0, 80.0));
}

#[test]
fn writer_rejects_unsafe_split_or_city_attributes() {
    let mut image = Image::new(1u64, "safe_leftImg8bit.png", 320, 240);
    image
        .attributes
        .insert(ATTR_SPLIT.to_string(), "../../escape".to_string());
    image
        .attributes
        .insert(ATTR_CITY.to_string(), "aachen".to_string());
    let dataset = Dataset {
        images: vec![image],
        categories: vec![Category::new(1u64, "car")],
        annotations: vec![],
        ..Default::default()
    };
    let temp = tempfile::tempdir().expect("create temp dir");

    let err = write_cityscapes_json(temp.path(), &dataset).expect_err("unsafe split should fail");
    let message = err.to_string();
    assert!(message.contains("unsafe Cityscapes"), "{message}");
}

#[test]
fn rejects_polygon_with_too_few_points() {
    let json = r#"{
        "imgHeight": 100,
        "imgWidth": 100,
        "objects": [
            {"label": "car", "polygon": [[1, 1], [5, 5]]}
        ]
    }"#;

    let err = from_cityscapes_str(json).expect_err("two-point polygon should fail");
    let message = err.to_string();
    assert!(
        message.contains("polygon must contain at least 3 points"),
        "{message}"
    );
}
