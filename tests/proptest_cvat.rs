use std::collections::BTreeSet;

use panlabel::ir::io_cvat_xml::{from_cvat_xml_str, to_cvat_xml_string};
use proptest::prelude::*;

mod proptest_helpers;

proptest! {
    #![proptest_config(proptest_helpers::proptest_config())]

    #[test]
    fn cvat_roundtrip_preserves_annotation_semantics(dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20)) {
        let xml = to_cvat_xml_string(&dataset).expect("serialize cvat");
        let restored = from_cvat_xml_str(&xml).expect("parse cvat");

        let res = proptest_helpers::assert_annotations_equivalent(
            &dataset,
            &restored,
            proptest_helpers::EPS_CVAT,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn cvat_roundtrip_is_semantically_idempotent(dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20)) {
        let first = from_cvat_xml_str(&to_cvat_xml_string(&dataset).expect("serialize first"))
            .expect("parse first");
        let second = from_cvat_xml_str(&to_cvat_xml_string(&first).expect("serialize second"))
            .expect("parse second");

        let res = proptest_helpers::assert_annotations_equivalent(
            &first,
            &second,
            proptest_helpers::EPS_CVAT,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn cvat_roundtrip_drops_unused_categories(dataset in proptest_helpers::arb_dataset_full(5, 5, 20)) {
        let xml = to_cvat_xml_string(&dataset).expect("serialize cvat");
        let restored = from_cvat_xml_str(&xml).expect("parse cvat");

        let expected_categories = proptest_helpers::used_category_names(&dataset)
            .expect("original dataset should have valid references");
        let restored_categories: BTreeSet<_> = restored
            .categories
            .iter()
            .map(|cat| cat.name.clone())
            .collect();
        prop_assert_eq!(restored_categories, expected_categories);

        // CVAT preserves images without annotations via empty <image> elements.
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
    fn cvat_occluded_z_order_source_and_custom_attrs_roundtrip(
        dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20),
        attrs in proptest::collection::vec((
            any::<bool>(),
            0i16..=5i16,
            proptest::string::string_regex("[a-z]{1,8}").expect("source regex"),
            proptest::string::string_regex("[a-z]{1,8}").expect("attr name regex"),
            proptest::string::string_regex("[A-Za-z0-9_\\- ]{0,12}").expect("attr value regex"),
        ), 1..=20)
    ) {
        let mut dataset = dataset;

        for (idx, ann) in dataset.annotations.iter_mut().enumerate() {
            let (occ, z, source, aname, aval) = &attrs[idx % attrs.len()];
            if *occ {
                ann.attributes.insert("occluded".to_string(), "1".to_string());
            } else {
                ann.attributes.insert("occluded".to_string(), "0".to_string());
            }
            if *z != 0 {
                ann.attributes.insert("z_order".to_string(), z.to_string());
            }
            ann.attributes.insert("source".to_string(), source.clone());
            ann.attributes.insert(format!("cvat_attr_{aname}"), aval.clone());
        }

        let xml = to_cvat_xml_string(&dataset).expect("serialize");
        let restored = from_cvat_xml_str(&xml).expect("parse");

        let res = proptest_helpers::assert_annotations_equivalent(
            &dataset,
            &restored,
            proptest_helpers::EPS_CVAT,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());

        prop_assert!(restored.annotations.iter().any(|a| a.attributes.contains_key("source")));
        prop_assert!(restored
            .annotations
            .iter()
            .any(|a| a.attributes.keys().any(|k| k.starts_with("cvat_attr_"))));
    }
}
