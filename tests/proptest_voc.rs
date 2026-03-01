use std::collections::{BTreeMap, BTreeSet};

use panlabel::ir::io_voc_xml::{read_voc_dir, write_voc_dir};
use proptest::prelude::*;

mod proptest_helpers;

proptest! {
    #![proptest_config(proptest_helpers::proptest_config())]

    #[test]
    fn voc_roundtrip_preserves_annotation_semantics(dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20)) {
        let temp = tempfile::tempdir().expect("create temp dir");
        write_voc_dir(temp.path(), &dataset).expect("write voc");
        let restored = read_voc_dir(temp.path()).expect("read voc");

        let res = proptest_helpers::assert_annotations_equivalent(
            &dataset,
            &restored,
            proptest_helpers::EPS_VOC,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn voc_roundtrip_is_semantically_idempotent(dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20)) {
        let temp = tempfile::tempdir().expect("create temp dir");
        let pass1 = temp.path().join("pass1");
        let pass2 = temp.path().join("pass2");

        write_voc_dir(&pass1, &dataset).expect("write first pass");
        let first = read_voc_dir(&pass1).expect("read first pass");

        write_voc_dir(&pass2, &first).expect("write second pass");
        let second = read_voc_dir(&pass2).expect("read second pass");

        let res = proptest_helpers::assert_annotations_equivalent(
            &first,
            &second,
            proptest_helpers::EPS_VOC,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn voc_roundtrip_drops_unused_categories(dataset in proptest_helpers::arb_dataset_full(5, 5, 20)) {
        let temp = tempfile::tempdir().expect("create temp dir");
        write_voc_dir(temp.path(), &dataset).expect("write voc");
        let restored = read_voc_dir(temp.path()).expect("read voc");

        let expected_categories = proptest_helpers::used_category_names(&dataset)
            .expect("original dataset should have valid references");
        let restored_categories: BTreeSet<_> = restored
            .categories
            .iter()
            .map(|cat| cat.name.clone())
            .collect();

        prop_assert_eq!(restored_categories, expected_categories);

        // VOC writes one XML per image, so unannotated images remain present.
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
    fn voc_pose_and_bool_attrs_roundtrip(
        dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20),
        attrs in proptest::collection::vec((
            proptest::string::string_regex("[A-Za-z]{1,8}").expect("valid pose regex"),
            any::<bool>(),
            any::<bool>(),
            any::<bool>()
        ), 1..=20)
    ) {
        let mut dataset = dataset;

        for (idx, ann) in dataset.annotations.iter_mut().enumerate() {
            let (pose, truncated, difficult, occluded) = &attrs[idx % attrs.len()];
            ann.attributes.insert("pose".to_string(), pose.clone());
            ann.attributes.insert(
                "truncated".to_string(),
                if *truncated { "yes".to_string() } else { "no".to_string() },
            );
            ann.attributes.insert(
                "difficult".to_string(),
                if *difficult { "true".to_string() } else { "false".to_string() },
            );
            ann.attributes.insert(
                "occluded".to_string(),
                if *occluded { "1".to_string() } else { "0".to_string() },
            );
        }

        let temp = tempfile::tempdir().expect("create temp dir");
        write_voc_dir(temp.path(), &dataset).expect("write voc");
        let restored = read_voc_dir(temp.path()).expect("read voc");

        let expected = voc_attr_signatures(&dataset, true).expect("build expected signatures");
        let actual = voc_attr_signatures(&restored, false).expect("build restored signatures");
        prop_assert_eq!(expected, actual);
    }
}

fn voc_attr_signatures(
    dataset: &panlabel::ir::Dataset,
    normalize_bools: bool,
) -> Result<
    Vec<(
        proptest_helpers::AnnSem,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    )>,
    String,
> {
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

        let normalize = |key: &str| {
            ann.attributes.get(key).and_then(|value| {
                if normalize_bools {
                    normalize_bool_attr(value).map(|v| v.to_string())
                } else {
                    Some(value.clone())
                }
            })
        };

        rows.push((
            proptest_helpers::AnnSem {
                image_file: image_file.clone(),
                category: category_name.clone(),
                xmin: ann.bbox.xmin(),
                ymin: ann.bbox.ymin(),
                xmax: ann.bbox.xmax(),
                ymax: ann.bbox.ymax(),
            },
            ann.attributes.get("pose").cloned(),
            normalize("truncated"),
            normalize("difficult"),
            normalize("occluded"),
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
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.4.cmp(&b.4))
    });

    Ok(rows)
}

fn normalize_bool_attr(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" => Some("1"),
        "false" | "no" | "0" => Some("0"),
        _ => None,
    }
}
