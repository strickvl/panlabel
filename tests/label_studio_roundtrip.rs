//! Integration tests for Label Studio JSON format support.

use panlabel::ir::io_label_studio_json::{
    from_label_studio_str, read_label_studio_json, to_label_studio_string,
};

#[test]
fn label_studio_write_then_read_roundtrip_semantic() {
    let dataset = read_label_studio_json(std::path::Path::new(
        "tests/fixtures/sample_valid.label_studio.json",
    ))
    .expect("read label-studio fixture");

    let json = to_label_studio_string(&dataset).expect("serialize label-studio");
    let restored = from_label_studio_str(&json).expect("parse serialized label-studio");

    assert_eq!(dataset.images.len(), restored.images.len());
    assert_eq!(dataset.categories.len(), restored.categories.len());
    assert_eq!(dataset.annotations.len(), restored.annotations.len());

    let mut left_categories: Vec<_> = dataset
        .categories
        .iter()
        .map(|cat| cat.name.clone())
        .collect();
    let mut right_categories: Vec<_> = restored
        .categories
        .iter()
        .map(|cat| cat.name.clone())
        .collect();
    left_categories.sort();
    right_categories.sort();
    assert_eq!(left_categories, right_categories);

    let mut left_boxes: Vec<_> = dataset
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
                ann.confidence,
            )
        })
        .collect();
    let mut right_boxes: Vec<_> = restored
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
                ann.confidence,
            )
        })
        .collect();

    left_boxes.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));
    right_boxes.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));

    assert_eq!(left_boxes.len(), right_boxes.len());
    for (left, right) in left_boxes.iter().zip(right_boxes.iter()) {
        assert_eq!(left.0, right.0);
        assert_eq!(left.1, right.1);
        assert!((left.2 - right.2).abs() < 1e-6);
        assert!((left.3 - right.3).abs() < 1e-6);
        assert!((left.4 - right.4).abs() < 1e-6);
        assert!((left.5 - right.5).abs() < 1e-6);
        match (left.6, right.6) {
            (Some(a), Some(b)) => assert!((a - b).abs() < 1e-12),
            (None, None) => {}
            _ => panic!("confidence mismatch: left={:?} right={:?}", left.6, right.6),
        }
    }
}

#[test]
fn label_studio_legacy_completions_fixture_is_supported() {
    let dataset = read_label_studio_json(std::path::Path::new(
        "tests/fixtures/sample_legacy_completions.label_studio.json",
    ))
    .expect("read legacy completions fixture");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.categories.len(), 1);
    assert_eq!(dataset.annotations.len(), 1);
    assert_eq!(dataset.images[0].file_name, "legacy_img.jpg");
    assert_eq!(dataset.categories[0].name, "legacy_cat");
}
