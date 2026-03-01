use std::collections::BTreeMap;

use panlabel::ir::io_coco_json::{from_coco_str, to_coco_string};
use proptest::prelude::*;

mod proptest_helpers;

proptest! {
    #![proptest_config(proptest_helpers::proptest_config())]

    #[test]
    fn coco_roundtrip_preserves_annotation_semantics(dataset in proptest_helpers::arb_dataset_full(5, 5, 20)) {
        let json = to_coco_string(&dataset).expect("serialize coco");
        let restored = from_coco_str(&json).expect("parse coco");

        let res = proptest_helpers::assert_annotations_equivalent(
            &dataset,
            &restored,
            proptest_helpers::EPS_COCO,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn coco_roundtrip_is_semantically_idempotent(dataset in proptest_helpers::arb_dataset_full(5, 5, 20)) {
        let first = from_coco_str(&to_coco_string(&dataset).expect("serialize first pass"))
            .expect("parse first pass");
        let second = from_coco_str(&to_coco_string(&first).expect("serialize second pass"))
            .expect("parse second pass");

        let res = proptest_helpers::assert_annotations_equivalent(
            &first,
            &second,
            proptest_helpers::EPS_COCO,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn coco_confidence_roundtrip_preserves_values(dataset in proptest_helpers::arb_dataset_with_confidence(5, 5, 20)) {
        let restored = from_coco_str(&to_coco_string(&dataset).expect("serialize coco"))
            .expect("parse coco");

        let left = semantics_with_confidence(&dataset).expect("build semantics for original");
        let right = semantics_with_confidence(&restored).expect("build semantics for restored");

        for (wanted, confidence) in left.iter().filter(|(_, conf)| conf.is_some()) {
            let expected = confidence.expect("filtered Some confidence");
            let found = right.iter().find(|(candidate, candidate_conf)| {
                approx_semantic(wanted, candidate, proptest_helpers::EPS_COCO)
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
    fn coco_iscrowd_area_attributes_roundtrip(
        dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20),
        attrs in proptest::collection::vec((0u8..=1u8, 0u32..=100_000u32), 1..=20)
    ) {
        let mut dataset = dataset;
        for (idx, ann) in dataset.annotations.iter_mut().enumerate() {
            let (iscrowd, area_raw) = attrs[idx % attrs.len()];
            let area = area_raw as f64 / 100.0;
            ann.attributes.insert("iscrowd".to_string(), iscrowd.to_string());
            ann.attributes.insert("area".to_string(), format!("{area:.6}"));
        }

        let restored = from_coco_str(&to_coco_string(&dataset).expect("serialize coco"))
            .expect("parse coco");

        let restored_by_id: BTreeMap<_, _> = restored.annotations.iter().map(|ann| (ann.id, ann)).collect();

        for ann in &dataset.annotations {
            let restored_ann = restored_by_id.get(&ann.id).expect("annotation id should be preserved in COCO roundtrip");

            let expected_iscrowd = ann.attributes.get("iscrowd").and_then(|v| v.parse::<u8>().ok());
            let actual_iscrowd = restored_ann.attributes.get("iscrowd").and_then(|v| v.parse::<u8>().ok());
            prop_assert_eq!(expected_iscrowd, actual_iscrowd);

            let expected_area = ann.attributes.get("area").and_then(|v| v.parse::<f64>().ok());
            let actual_area = restored_ann.attributes.get("area").and_then(|v| v.parse::<f64>().ok());
            match (expected_area, actual_area) {
                (Some(left), Some(right)) => {
                    prop_assert!((left - right).abs() < 1e-6, "area mismatch: left={left} right={right}");
                }
                _ => prop_assert!(false, "expected area to roundtrip for annotation id {}", ann.id.as_u64()),
            }
        }
    }
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
