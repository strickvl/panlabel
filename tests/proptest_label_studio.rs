use std::collections::{BTreeMap, BTreeSet};

use panlabel::ir::io_label_studio_json::{from_label_studio_str, to_label_studio_string};
use proptest::prelude::*;

mod proptest_helpers;

proptest! {
    #![proptest_config(proptest_helpers::proptest_config())]

    #[test]
    fn label_studio_roundtrip_preserves_annotation_semantics(
        dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20)
    ) {
        let json = to_label_studio_string(&dataset).expect("serialize label-studio");
        let restored = from_label_studio_str(&json).expect("parse label-studio");

        let res = proptest_helpers::assert_annotations_equivalent(
            &dataset,
            &restored,
            proptest_helpers::EPS_LABEL_STUDIO,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn label_studio_roundtrip_is_semantically_idempotent(
        dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20)
    ) {
        let first = from_label_studio_str(&to_label_studio_string(&dataset).expect("serialize first pass"))
            .expect("parse first pass");
        let second = from_label_studio_str(&to_label_studio_string(&first).expect("serialize second pass"))
            .expect("parse second pass");

        let res = proptest_helpers::assert_annotations_equivalent(
            &first,
            &second,
            proptest_helpers::EPS_LABEL_STUDIO,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn label_studio_roundtrip_drops_unused_categories(
        dataset in proptest_helpers::arb_dataset_full(5, 5, 20)
    ) {
        let json = to_label_studio_string(&dataset).expect("serialize label-studio");
        let restored = from_label_studio_str(&json).expect("parse label-studio");

        let expected_categories = proptest_helpers::used_category_names(&dataset)
            .expect("original dataset should have valid references");
        let restored_categories: BTreeSet<_> = restored
            .categories
            .iter()
            .map(|cat| cat.name.clone())
            .collect();
        prop_assert_eq!(restored_categories, expected_categories);

        let original_images: BTreeSet<_> = dataset
            .images
            .iter()
            .map(|img| img.file_name.clone())
            .collect();
        let restored_images: BTreeSet<_> = restored
            .images
            .iter()
            .map(|img| img.file_name.clone())
            .collect();
        prop_assert_eq!(restored_images, original_images);
    }

    #[test]
    fn label_studio_confidence_roundtrip_preserves_values(
        dataset in proptest_helpers::arb_dataset_with_confidence(5, 5, 20)
    ) {
        let restored = from_label_studio_str(&to_label_studio_string(&dataset).expect("serialize"))
            .expect("parse");

        let left = semantics_with_confidence(&dataset).expect("build semantics for original");
        let right = semantics_with_confidence(&restored).expect("build semantics for restored");

        for (wanted, confidence) in left.iter().filter(|(_, conf)| conf.is_some()) {
            let expected = confidence.expect("filtered Some confidence");
            let found = right.iter().find(|(candidate, candidate_conf)| {
                approx_semantic(wanted, candidate, proptest_helpers::EPS_LABEL_STUDIO)
                    && candidate_conf
                        .map(|value| (value - expected).abs() < 1e-12)
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

    #[test]
    fn label_studio_rotation_is_envelope_lossy(
        dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20),
        rotations in proptest::collection::vec(-89i16..=89i16, 1..=20)
    ) {
        let mut rotated = dataset.clone();

        for (idx, ann) in rotated.annotations.iter_mut().enumerate() {
            let mut degrees = rotations[idx % rotations.len()] as f64;
            if degrees.abs() < 1.0 {
                degrees = 17.0;
            }
            ann.attributes
                .insert("ls_rotation_deg".to_string(), format!("{degrees:.3}"));
        }

        let mut expected = rotated.clone();
        for ann in &mut expected.annotations {
            let rotation = ann
                .attributes
                .get("ls_rotation_deg")
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.0);
            ann.bbox = rotated_envelope_bbox_for_test(ann.bbox, rotation);
        }

        let restored = from_label_studio_str(&to_label_studio_string(&rotated).expect("serialize"))
            .expect("parse");

        let sem_res = proptest_helpers::assert_annotations_equivalent(&expected, &restored, 1e-3);
        prop_assert!(sem_res.is_ok(), "{}", sem_res.unwrap_err());

        let mut expected_rotations: Vec<f64> = rotated
            .annotations
            .iter()
            .filter_map(|ann| ann.attributes.get("ls_rotation_deg"))
            .filter_map(|value| value.parse::<f64>().ok())
            .collect();
        let mut restored_rotations: Vec<f64> = restored
            .annotations
            .iter()
            .filter_map(|ann| ann.attributes.get("ls_rotation_deg"))
            .filter_map(|value| value.parse::<f64>().ok())
            .collect();

        expected_rotations.sort_by(|a, b| a.total_cmp(b));
        restored_rotations.sort_by(|a, b| a.total_cmp(b));
        prop_assert_eq!(expected_rotations.len(), restored_rotations.len());
        for (left, right) in expected_rotations.iter().zip(restored_rotations.iter()) {
            prop_assert!((left - right).abs() < 1e-6);
        }
    }
}

fn rotated_envelope_bbox_for_test(
    bbox: panlabel::ir::BBoxXYXY<panlabel::ir::Pixel>,
    rotation_deg: f64,
) -> panlabel::ir::BBoxXYXY<panlabel::ir::Pixel> {
    let xmin = bbox.xmin();
    let ymin = bbox.ymin();
    let xmax = bbox.xmax();
    let ymax = bbox.ymax();

    let theta = rotation_deg.to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let cx = (xmin + xmax) / 2.0;
    let cy = (ymin + ymax) / 2.0;

    let corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)];

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (x, y) in corners {
        let dx = x - cx;
        let dy = y - cy;
        let rx = cx + (dx * cos_t) - (dy * sin_t);
        let ry = cy + (dx * sin_t) + (dy * cos_t);

        min_x = min_x.min(rx);
        min_y = min_y.min(ry);
        max_x = max_x.max(rx);
        max_y = max_y.max(ry);
    }

    panlabel::ir::BBoxXYXY::from_xyxy(min_x, min_y, max_x, max_y)
}

fn semantics_with_confidence(
    dataset: &panlabel::ir::Dataset,
) -> Result<Vec<(proptest_helpers::AnnSem, Option<f64>)>, String> {
    let image_by_id: BTreeMap<_, _> = dataset
        .images
        .iter()
        .map(|img| (img.id, img.file_name.clone()))
        .collect();
    let category_by_id: BTreeMap<_, _> = dataset
        .categories
        .iter()
        .map(|cat| (cat.id, cat.name.clone()))
        .collect();

    let mut rows = Vec::with_capacity(dataset.annotations.len());
    for ann in &dataset.annotations {
        let image_file = image_by_id.get(&ann.image_id).ok_or_else(|| {
            format!(
                "annotation {} references missing image {}",
                ann.id.as_u64(),
                ann.image_id.as_u64()
            )
        })?;
        let category_name = category_by_id.get(&ann.category_id).ok_or_else(|| {
            format!(
                "annotation {} references missing category {}",
                ann.id.as_u64(),
                ann.category_id.as_u64()
            )
        })?;

        rows.push((
            proptest_helpers::AnnSem {
                image_file: image_file.clone(),
                category: category_name.clone(),
                xmin: ann.bbox.xmin(),
                ymin: ann.bbox.ymin(),
                xmax: ann.bbox.xmax(),
                ymax: ann.bbox.ymax(),
            },
            ann.confidence,
        ));
    }

    rows.sort_by(|a, b| {
        a.0.image_file
            .cmp(&b.0.image_file)
            .then_with(|| a.0.category.cmp(&b.0.category))
            .then_with(|| a.0.xmin.total_cmp(&b.0.xmin))
            .then_with(|| a.0.ymin.total_cmp(&b.0.ymin))
            .then_with(|| a.0.xmax.total_cmp(&b.0.xmax))
            .then_with(|| a.0.ymax.total_cmp(&b.0.ymax))
            .then_with(|| a.1.unwrap_or(-1.0).total_cmp(&b.1.unwrap_or(-1.0)))
    });

    Ok(rows)
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
