use std::collections::BTreeMap;

use panlabel::ir::io_yolo::{read_yolo_dir, write_yolo_dir};
use panlabel::ir::{CategoryId, ImageId};
use proptest::prelude::*;

mod common;
mod proptest_helpers;

proptest! {
    #![proptest_config(proptest_helpers::proptest_config())]

    #[test]
    fn yolo_roundtrip_preserves_annotation_semantics(
        dataset in proptest_helpers::arb_dataset_annotated(3, 3, 10)
    ) {
        let temp = tempfile::tempdir().expect("create temp dir");
        let output_root = temp.path().join("yolo_output");

        write_yolo_dir(&output_root, &dataset).expect("write yolo");
        materialize_images_for_reread(&output_root, &dataset);
        let restored = read_yolo_dir(&output_root).expect("read yolo");

        let eps = proptest_helpers::eps_yolo_for_dataset(&dataset);
        let res = proptest_helpers::assert_annotations_equivalent(&dataset, &restored, eps);
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn yolo_roundtrip_is_semantically_idempotent(
        dataset in proptest_helpers::arb_dataset_annotated(3, 3, 10)
    ) {
        let temp = tempfile::tempdir().expect("create temp dir");
        let pass1_root = temp.path().join("pass1");
        let pass2_root = temp.path().join("pass2");

        write_yolo_dir(&pass1_root, &dataset).expect("write first pass");
        materialize_images_for_reread(&pass1_root, &dataset);
        let first = read_yolo_dir(&pass1_root).expect("read first pass");

        write_yolo_dir(&pass2_root, &first).expect("write second pass");
        materialize_images_for_reread(&pass2_root, &first);
        let second = read_yolo_dir(&pass2_root).expect("read second pass");

        let eps = proptest_helpers::eps_yolo_for_dataset(&first);
        let res = proptest_helpers::assert_annotations_equivalent(&first, &second, eps);
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn yolo_confidence_roundtrip_preserves_values(
        dataset in proptest_helpers::arb_dataset_with_confidence(3, 3, 10)
    ) {
        let temp = tempfile::tempdir().expect("create temp dir");
        let output_root = temp.path().join("yolo_conf");

        write_yolo_dir(&output_root, &dataset).expect("write yolo");
        materialize_images_for_reread(&output_root, &dataset);
        let restored = read_yolo_dir(&output_root).expect("read yolo");

        let eps = proptest_helpers::eps_yolo_for_dataset(&dataset);
        let res = proptest_helpers::assert_annotations_equivalent(&dataset, &restored, eps);
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());

        // Build semantic lookup for confidence comparison
        let left = semantics_with_confidence(&dataset);
        let right = semantics_with_confidence(&restored);

        for (wanted, confidence) in left.iter().filter(|(_, conf)| conf.is_some()) {
            let expected = confidence.expect("filtered Some confidence");
            let found = right.iter().find(|(candidate, candidate_conf)| {
                approx_semantic(wanted, candidate, eps)
                    && candidate_conf
                        .map(|value| (value - expected).abs() < 1e-6)
                        .unwrap_or(false)
            });

            prop_assert!(
                found.is_some(),
                "missing confidence-preserving match for {:?} (confidence={})",
                wanted,
                expected
            );
        }
    }
}

fn materialize_images_for_reread(root: &std::path::Path, dataset: &panlabel::ir::Dataset) {
    for image in &dataset.images {
        let image_path = root.join("images").join(&image.file_name);
        common::write_bmp(&image_path, image.width, image.height);
    }
}

fn semantics_with_confidence(
    dataset: &panlabel::ir::Dataset,
) -> Vec<(proptest_helpers::AnnSem, Option<f64>)> {
    let image_by_id: BTreeMap<ImageId, String> = dataset
        .images
        .iter()
        .map(|img| (img.id, img.file_name.clone()))
        .collect();
    let category_by_id: BTreeMap<CategoryId, String> = dataset
        .categories
        .iter()
        .map(|cat| (cat.id, cat.name.clone()))
        .collect();

    let mut rows: Vec<(proptest_helpers::AnnSem, Option<f64>)> = dataset
        .annotations
        .iter()
        .map(|ann| {
            (
                proptest_helpers::AnnSem {
                    image_file: image_by_id[&ann.image_id].clone(),
                    category: category_by_id[&ann.category_id].clone(),
                    xmin: ann.bbox.xmin(),
                    ymin: ann.bbox.ymin(),
                    xmax: ann.bbox.xmax(),
                    ymax: ann.bbox.ymax(),
                },
                ann.confidence,
            )
        })
        .collect();

    rows.sort_by(|a, b| {
        a.0.image_file
            .cmp(&b.0.image_file)
            .then_with(|| a.0.category.cmp(&b.0.category))
            .then_with(|| a.0.xmin.total_cmp(&b.0.xmin))
            .then_with(|| a.0.ymin.total_cmp(&b.0.ymin))
    });

    rows
}

fn approx_semantic(
    left: &proptest_helpers::AnnSem,
    right: &proptest_helpers::AnnSem,
    eps: f64,
) -> bool {
    left.image_file == right.image_file
        && left.category == right.category
        && (left.xmin - right.xmin).abs() <= eps
        && (left.ymin - right.ymin).abs() <= eps
        && (left.xmax - right.xmax).abs() <= eps
        && (left.ymax - right.ymax).abs() <= eps
}
