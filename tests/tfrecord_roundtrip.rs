use std::collections::BTreeSet;
use std::path::Path;

use panlabel::ir::io_tfrecord::{
    from_tfrecord_slice, read_tfrecord, to_tfrecord_vec, write_tfrecord, ATTR_AREA,
    ATTR_CLASS_LABEL, ATTR_DIFFICULT, ATTR_FORMAT, ATTR_GROUP_OF, ATTR_IS_CROWD, ATTR_KEY_SHA256,
    ATTR_SOURCE_ID,
};
use panlabel::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

fn sample_dataset() -> Dataset {
    let mut image_with_objects = Image::new(2u64, "b.jpg", 200, 100);
    image_with_objects
        .attributes
        .insert(ATTR_SOURCE_ID.to_string(), "source-b".to_string());
    image_with_objects
        .attributes
        .insert(ATTR_KEY_SHA256.to_string(), "abc123".to_string());
    image_with_objects
        .attributes
        .insert(ATTR_FORMAT.to_string(), "jpeg".to_string());

    let image_without_objects = Image::new(1u64, "a.jpg", 640, 480);

    Dataset {
        images: vec![image_with_objects, image_without_objects],
        categories: vec![Category::new(2u64, "dog"), Category::new(1u64, "cat")],
        annotations: vec![
            Annotation::new(
                2u64,
                2u64,
                2u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 5.0, 60.0, 45.0),
            )
            .with_attribute(ATTR_CLASS_LABEL, "7")
            .with_attribute(ATTR_AREA, "2000")
            .with_attribute(ATTR_IS_CROWD, "0")
            .with_attribute(ATTR_DIFFICULT, "1")
            .with_attribute(ATTR_GROUP_OF, "0")
            .with_attribute("tfrecord_weight", "0.5"),
            Annotation::new(
                1u64,
                2u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(100.0, 20.0, 180.0, 80.0),
            )
            .with_attribute(ATTR_CLASS_LABEL, "3")
            .with_attribute(ATTR_AREA, "4800")
            .with_attribute(ATTR_IS_CROWD, "0")
            .with_attribute(ATTR_DIFFICULT, "0")
            .with_attribute(ATTR_GROUP_OF, "0")
            .with_attribute("tfrecord_weight", "1.25"),
        ],
        ..Default::default()
    }
}

#[test]
fn tfrecord_slice_roundtrip_preserves_detection_semantics_and_empty_images() {
    let original = sample_dataset();
    let bytes = to_tfrecord_vec(&original).expect("serialize tfrecord");
    assert!(!bytes.is_empty());

    let restored = from_tfrecord_slice(&bytes).expect("parse tfrecord");

    let image_files: BTreeSet<_> = restored
        .images
        .iter()
        .map(|image| image.file_name.as_str())
        .collect();
    assert_eq!(image_files, BTreeSet::from(["a.jpg", "b.jpg"]));
    assert_eq!(restored.annotations.len(), 2);

    let category_names: BTreeSet<_> = restored
        .categories
        .iter()
        .map(|category| category.name.as_str())
        .collect();
    assert_eq!(category_names, BTreeSet::from(["cat", "dog"]));

    let b_image = restored
        .images
        .iter()
        .find(|image| image.file_name == "b.jpg")
        .expect("b.jpg image");
    assert_eq!(b_image.attributes[ATTR_SOURCE_ID], "source-b");
    assert_eq!(b_image.attributes[ATTR_KEY_SHA256], "abc123");
    assert_eq!(b_image.attributes[ATTR_FORMAT], "jpeg");

    let dog_ann = restored
        .annotations
        .iter()
        .find(|ann| ann.attributes.get(ATTR_CLASS_LABEL).map(String::as_str) == Some("7"))
        .expect("dog annotation with numeric label");
    assert!((dog_ann.bbox.xmin() - 10.0).abs() < 1e-4);
    assert!((dog_ann.bbox.ymin() - 5.0).abs() < 1e-4);
    assert!((dog_ann.bbox.xmax() - 60.0).abs() < 1e-4);
    assert!((dog_ann.bbox.ymax() - 45.0).abs() < 1e-4);
    assert_eq!(dog_ann.attributes[ATTR_AREA], "2000");
    assert_eq!(dog_ann.attributes[ATTR_IS_CROWD], "0");
    assert_eq!(dog_ann.attributes[ATTR_DIFFICULT], "1");
}

#[test]
fn tfrecord_file_roundtrip_uses_public_read_write_api() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("sample.tfrecord");
    let original = sample_dataset();

    write_tfrecord(&path, &original).expect("write tfrecord");
    let restored = read_tfrecord(Path::new(&path)).expect("read tfrecord");

    assert_eq!(restored.images.len(), 2);
    assert_eq!(restored.annotations.len(), 2);
}
