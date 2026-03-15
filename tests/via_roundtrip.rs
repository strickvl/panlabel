//! Integration tests for VIA JSON format support (write-then-read roundtrip).

mod common;

use common::write_bmp;
use panlabel::ir::io_via_json::{read_via_json, write_via_json};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

#[test]
fn write_then_read_roundtrip() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("annotations.json");

    let mut img1 = Image::new(1u64, "img1.bmp", 640, 480);
    img1.attributes
        .insert("via_size_bytes".to_string(), "5432".to_string());
    img1.attributes
        .insert("via_file_attr_source".to_string(), "camera".to_string());

    let mut img2 = Image::new(2u64, "img2.bmp", 800, 600);
    img2.attributes
        .insert("via_size_bytes".to_string(), "7890".to_string());
    img2.attributes
        .insert("via_file_attr_location".to_string(), "outdoor".to_string());

    let ann1 = Annotation::new(
        1u64,
        1u64,
        1u64,
        BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 200.0, 180.0),
    )
    .with_attribute("via_region_attr_difficult", "true");

    let ann2 = Annotation::new(
        2u64,
        1u64,
        2u64,
        BBoxXYXY::<Pixel>::from_xyxy(300.0, 100.0, 500.0, 400.0),
    )
    .with_attribute("via_region_attr_quality", "good");

    let ann3 = Annotation::new(
        3u64,
        2u64,
        1u64,
        BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 350.0, 280.0),
    );

    let dataset = Dataset {
        images: vec![img1, img2],
        categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
        annotations: vec![ann1, ann2, ann3],
        ..Default::default()
    };

    // Write the VIA JSON
    write_via_json(&json_path, &dataset).expect("write via json");

    // Create tiny BMP files next to the JSON so the reader can resolve dimensions
    write_bmp(&temp.path().join("img1.bmp"), 640, 480);
    write_bmp(&temp.path().join("img2.bmp"), 800, 600);

    // Read back
    let restored = read_via_json(&json_path).expect("read via json");

    // Basic counts
    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.categories.len(), 2);
    assert_eq!(restored.annotations.len(), 3);

    // Image dimensions resolved from BMP files
    let img1_r = restored
        .images
        .iter()
        .find(|i| i.file_name == "img1.bmp")
        .expect("img1 present");
    assert_eq!(img1_r.width, 640);
    assert_eq!(img1_r.height, 480);

    let img2_r = restored
        .images
        .iter()
        .find(|i| i.file_name == "img2.bmp")
        .expect("img2 present");
    assert_eq!(img2_r.width, 800);
    assert_eq!(img2_r.height, 600);

    // Bbox roundtrip: writer converts XYXY -> XYWH, reader converts XYWH -> XYXY
    // xyxy(10, 20, 200, 180) -> xywh(10, 20, 190, 160) -> xyxy(10, 20, 200, 180)
    let anns_for_img1: Vec<_> = restored
        .annotations
        .iter()
        .filter(|a| a.image_id == img1_r.id)
        .collect();
    assert_eq!(anns_for_img1.len(), 2);

    // Sort by xmin for deterministic comparison
    let mut sorted_anns: Vec<_> = anns_for_img1.iter().collect();
    sorted_anns.sort_by(|a, b| a.bbox.xmin().partial_cmp(&b.bbox.xmin()).unwrap());

    assert!((sorted_anns[0].bbox.xmin() - 10.0).abs() < 1e-9);
    assert!((sorted_anns[0].bbox.ymin() - 20.0).abs() < 1e-9);
    assert!((sorted_anns[0].bbox.xmax() - 200.0).abs() < 1e-9);
    assert!((sorted_anns[0].bbox.ymax() - 180.0).abs() < 1e-9);

    assert!((sorted_anns[1].bbox.xmin() - 300.0).abs() < 1e-9);
    assert!((sorted_anns[1].bbox.ymin() - 100.0).abs() < 1e-9);
    assert!((sorted_anns[1].bbox.xmax() - 500.0).abs() < 1e-9);
    assert!((sorted_anns[1].bbox.ymax() - 400.0).abs() < 1e-9);
}

#[test]
fn via_specific_attributes_roundtrip() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("annotations.json");

    let mut img = Image::new(1u64, "test.bmp", 640, 480);
    img.attributes
        .insert("via_size_bytes".to_string(), "12345".to_string());
    img.attributes
        .insert("via_file_attr_source".to_string(), "web".to_string());
    img.attributes
        .insert("via_file_attr_license".to_string(), "cc-by".to_string());

    let ann = Annotation::new(
        1u64,
        1u64,
        1u64,
        BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 80.0),
    )
    .with_attribute("via_region_attr_difficult", "true")
    .with_attribute("via_region_attr_color", "red");

    let dataset = Dataset {
        images: vec![img],
        categories: vec![Category::new(1u64, "cat")],
        annotations: vec![ann],
        ..Default::default()
    };

    write_via_json(&json_path, &dataset).expect("write");
    write_bmp(&temp.path().join("test.bmp"), 640, 480);

    let restored = read_via_json(&json_path).expect("read");

    // Check via_size_bytes roundtrips on image
    let rimg = &restored.images[0];
    assert_eq!(
        rimg.attributes.get("via_size_bytes"),
        Some(&"12345".to_string())
    );

    // Check via_file_attr_* roundtrips on image
    assert_eq!(
        rimg.attributes.get("via_file_attr_source"),
        Some(&"web".to_string())
    );
    assert_eq!(
        rimg.attributes.get("via_file_attr_license"),
        Some(&"cc-by".to_string())
    );

    // Check via_region_attr_* roundtrips on annotation
    let rann = &restored.annotations[0];
    assert_eq!(
        rann.attributes.get("via_region_attr_difficult"),
        Some(&"true".to_string())
    );
    assert_eq!(
        rann.attributes.get("via_region_attr_color"),
        Some(&"red".to_string())
    );
}

#[test]
fn unannotated_images_roundtrip_with_empty_regions() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("annotations.json");

    let dataset = Dataset {
        images: vec![
            Image::new(1u64, "annotated.bmp", 640, 480),
            Image::new(2u64, "empty.bmp", 800, 600),
        ],
        categories: vec![Category::new(1u64, "car")],
        annotations: vec![Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(20.0, 30.0, 200.0, 180.0),
        )],
        ..Default::default()
    };

    write_via_json(&json_path, &dataset).expect("write");
    write_bmp(&temp.path().join("annotated.bmp"), 640, 480);
    write_bmp(&temp.path().join("empty.bmp"), 800, 600);

    let restored = read_via_json(&json_path).expect("read");

    // Both images should be present
    assert_eq!(restored.images.len(), 2);
    // Only one annotation (the unannotated image has empty regions)
    assert_eq!(restored.annotations.len(), 1);

    // Verify both images are found
    let names: Vec<&str> = restored
        .images
        .iter()
        .map(|i| i.file_name.as_str())
        .collect();
    assert!(names.contains(&"annotated.bmp"));
    assert!(names.contains(&"empty.bmp"));
}

#[test]
fn deterministic_ids_sorted_by_filename() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let json_path = temp.path().join("annotations.json");

    // Write images out of alphabetical order
    let dataset = Dataset {
        images: vec![
            Image::new(2u64, "z_img.bmp", 640, 480),
            Image::new(1u64, "a_img.bmp", 800, 600),
        ],
        categories: vec![Category::new(1u64, "obj")],
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

    write_via_json(&json_path, &dataset).expect("write");
    write_bmp(&temp.path().join("a_img.bmp"), 800, 600);
    write_bmp(&temp.path().join("z_img.bmp"), 640, 480);

    let restored = read_via_json(&json_path).expect("read");

    // Reader sorts by filename: a_img gets ID 1, z_img gets ID 2
    assert_eq!(restored.images[0].file_name, "a_img.bmp");
    assert_eq!(restored.images[0].id.as_u64(), 1);
    assert_eq!(restored.images[1].file_name, "z_img.bmp");
    assert_eq!(restored.images[1].id.as_u64(), 2);
}
