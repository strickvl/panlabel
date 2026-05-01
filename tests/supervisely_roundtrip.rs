use std::fs;

use panlabel::ir::io_supervisely_json::{
    from_supervisely_str, read_supervisely_json, to_supervisely_string, write_supervisely_json,
    ATTR_ANN_PATH, ATTR_DATASET, ATTR_GEOMETRY_TYPE, ATTR_OBJECT_ID,
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
fn parses_fixture_with_rectangle_and_polygon_envelopes() {
    let dataset = read_supervisely_json("tests/fixtures/sample_valid.supervisely.json".as_ref())
        .expect("parse fixture");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "sample_valid.supervisely");
    assert_eq!(dataset.images[0].width, 640);
    assert_eq!(dataset.images[0].height, 480);
    assert_eq!(category_names(&dataset), vec!["car", "cat", "dog"]);
    assert_eq!(dataset.annotations.len(), 3);

    assert_bbox(dataset.annotations[0].bbox, (10.0, 20.0, 100.0, 80.0));
    assert_eq!(
        dataset.annotations[0].attributes.get(ATTR_GEOMETRY_TYPE),
        Some(&"rectangle".to_string())
    );
    assert_eq!(
        dataset.annotations[0].attributes.get(ATTR_OBJECT_ID),
        Some(&"201".to_string())
    );

    assert_bbox(dataset.annotations[1].bbox, (50.0, 60.0, 200.0, 210.0));
    assert_eq!(
        dataset.annotations[1].attributes.get(ATTR_GEOMETRY_TYPE),
        Some(&"polygon".to_string())
    );

    assert_bbox(dataset.annotations[2].bbox, (280.0, 100.0, 360.0, 180.0));
    assert_eq!(
        dataset.annotations[2].attributes.get(ATTR_GEOMETRY_TYPE),
        Some(&"polygon".to_string())
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

    let json = to_supervisely_string(&dataset).expect("serialize");
    let restored = from_supervisely_str(&json).expect("parse");

    assert_eq!(restored.images.len(), 1);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
    assert_bbox(restored.annotations[0].bbox, (10.0, 20.0, 100.0, 80.0));
    assert_bbox(restored.annotations[1].bbox, (50.0, 60.0, 200.0, 210.0));
}

#[test]
fn rejects_unsupported_geometry() {
    let json = r#"{
        "size": {"width": 100, "height": 100},
        "objects": [
            {
                "classTitle": "mask",
                "geometryType": "bitmap"
            }
        ]
    }"#;

    let err = from_supervisely_str(json).expect_err("unsupported geometry should fail");
    let message = err.to_string();
    assert!(message.contains("bitmap"), "{message}");
    assert!(
        message.contains("unsupported Supervisely geometryType"),
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
    let err = write_supervisely_json(temp.path(), &dataset).expect_err("unsafe path should fail");
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

    let err = to_supervisely_string(&dataset).expect_err("missing image should fail");
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
    let err = write_supervisely_json(temp.path(), &dataset).expect_err("missing image should fail");
    let message = err.to_string();
    assert!(message.contains("references missing image"), "{message}");
}

#[test]
fn rejects_rectangle_with_more_than_two_points() {
    let json = r#"{
        "size": {"width": 100, "height": 100},
        "objects": [
            {
                "classTitle": "cat",
                "geometryType": "rectangle",
                "geometry": {
                    "points": {
                        "exterior": [[1, 1], [5, 1], [5, 5], [1, 5]],
                        "interior": []
                    }
                }
            }
        ]
    }"#;

    let err = from_supervisely_str(json).expect_err("rectangle with 4 points should fail");
    let message = err.to_string();
    assert!(
        message.contains("rectangle must contain exactly 2 exterior points"),
        "{message}"
    );
}

#[test]
fn reads_dataset_ann_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("ann")).expect("create ann dir");
    fs::write(
        temp.path().join("ann/example.jpg.json"),
        r#"{
            "size": {"width": 320, "height": 240},
            "objects": [
                {
                    "classTitle": "cat",
                    "geometryType": "rectangle",
                    "geometry": {"points": {"exterior": [[1, 2], [30, 40]], "interior": []}}
                }
            ]
        }"#,
    )
    .expect("write annotation");

    let dataset = read_supervisely_json(temp.path()).expect("read dataset dir");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "example.jpg");
    assert_eq!(
        dataset.images[0].attributes.get(ATTR_DATASET),
        temp.path()
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .as_ref()
    );
    assert_eq!(category_names(&dataset), vec!["cat"]);
    assert_bbox(dataset.annotations[0].bbox, (1.0, 2.0, 30.0, 40.0));
}

#[test]
fn reads_project_root_with_meta_classes() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("dataset_01/ann")).expect("create ann dir");
    fs::write(
        temp.path().join("meta.json"),
        r#"{"classes": [{"title": "cat"}, {"title": "unused"}], "tags": []}"#,
    )
    .expect("write meta");
    fs::write(
        temp.path().join("dataset_01/ann/project_image.png.json"),
        r#"{
            "size": {"width": 320, "height": 240},
            "objects": [
                {
                    "classTitle": "cat",
                    "geometryType": "polygon",
                    "geometry": {"points": {"exterior": [[1, 2], [30, 4], [10, 40]], "interior": []}}
                }
            ]
        }"#,
    )
    .expect("write annotation");

    let dataset = read_supervisely_json(temp.path()).expect("read project");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "dataset_01/project_image.png");
    assert_eq!(
        dataset.images[0].attributes.get(ATTR_DATASET),
        Some(&"dataset_01".to_string())
    );
    assert_eq!(
        dataset.images[0].attributes.get(ATTR_ANN_PATH),
        Some(&"dataset_01/ann/project_image.png.json".to_string())
    );
    assert_eq!(category_names(&dataset), vec!["cat", "unused"]);
    assert_bbox(dataset.annotations[0].bbox, (1.0, 2.0, 30.0, 40.0));
    assert_eq!(
        dataset.annotations[0].attributes.get(ATTR_GEOMETRY_TYPE),
        Some(&"polygon".to_string())
    );
}

#[test]
fn project_root_allows_duplicate_basenames_in_different_datasets() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("dataset_a/ann")).expect("create dataset_a ann");
    fs::create_dir_all(temp.path().join("dataset_b/ann")).expect("create dataset_b ann");
    fs::write(
        temp.path().join("meta.json"),
        r#"{"classes": [{"title": "cat"}]}"#,
    )
    .expect("write meta");

    let ann = r#"{
        "size": {"width": 100, "height": 100},
        "objects": [
            {
                "classTitle": "cat",
                "geometryType": "rectangle",
                "geometry": {"points": {"exterior": [[1, 2], [30, 40]], "interior": []}}
            }
        ]
    }"#;
    fs::write(temp.path().join("dataset_a/ann/same.jpg.json"), ann).expect("write ann a");
    fs::write(temp.path().join("dataset_b/ann/same.jpg.json"), ann).expect("write ann b");

    let dataset = read_supervisely_json(temp.path()).expect("read project");

    let image_names: Vec<&str> = dataset
        .images
        .iter()
        .map(|image| image.file_name.as_str())
        .collect();
    assert_eq!(
        image_names,
        vec!["dataset_a/same.jpg", "dataset_b/same.jpg"]
    );
    assert_eq!(dataset.annotations.len(), 2);
}

#[test]
fn dataset_ann_directory_preserves_nested_image_path() {
    let temp = tempfile::tempdir().expect("create temp dir");
    fs::create_dir_all(temp.path().join("ann/nested")).expect("create nested ann dir");
    fs::write(
        temp.path().join("ann/nested/example.jpg.json"),
        r#"{
            "size": {"width": 320, "height": 240},
            "objects": [
                {
                    "classTitle": "cat",
                    "geometryType": "rectangle",
                    "geometry": {"points": {"exterior": [[1, 2], [30, 40]], "interior": []}}
                }
            ]
        }"#,
    )
    .expect("write annotation");

    let dataset = read_supervisely_json(temp.path()).expect("read dataset dir");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "nested/example.jpg");
}

#[test]
fn writes_and_reads_canonical_project_directory() {
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
    write_supervisely_json(temp.path(), &dataset).expect("write project");

    assert!(temp.path().join("meta.json").is_file());
    assert!(temp.path().join("dataset/ann/a.jpg.json").is_file());
    assert!(temp.path().join("dataset/ann/b.jpg.json").is_file());
    assert!(temp.path().join("dataset/img/README.txt").is_file());

    let restored = read_supervisely_json(temp.path()).expect("read project");
    assert_eq!(restored.images[0].file_name, "dataset/a.jpg");
    assert_eq!(restored.images[1].file_name, "dataset/b.jpg");
    assert_eq!(category_names(&restored), vec!["cat", "unused"]);
    assert_eq!(restored.annotations.len(), 1);
    assert_bbox(restored.annotations[0].bbox, (5.0, 6.0, 50.0, 60.0));
}
