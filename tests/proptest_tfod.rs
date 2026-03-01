use std::collections::BTreeSet;

use panlabel::ir::io_tfod_csv::{from_tfod_csv_str, to_tfod_csv_string};
use proptest::prelude::*;

mod proptest_helpers;

proptest! {
    #![proptest_config(proptest_helpers::proptest_config())]

    #[test]
    fn tfod_roundtrip_preserves_annotation_semantics(dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20)) {
        let csv = to_tfod_csv_string(&dataset).expect("serialize tfod");
        let restored = from_tfod_csv_str(&csv).expect("parse tfod");

        let res = proptest_helpers::assert_annotations_equivalent(
            &dataset,
            &restored,
            proptest_helpers::EPS_TFOD,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn tfod_roundtrip_is_semantically_idempotent(dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20)) {
        let first = from_tfod_csv_str(&to_tfod_csv_string(&dataset).expect("serialize first pass"))
            .expect("parse first pass");
        let second = from_tfod_csv_str(&to_tfod_csv_string(&first).expect("serialize second pass"))
            .expect("parse second pass");

        let res = proptest_helpers::assert_annotations_equivalent(
            &first,
            &second,
            proptest_helpers::EPS_TFOD,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn tfod_roundtrip_drops_only_unannotated_images(dataset in proptest_helpers::arb_dataset_full(5, 5, 20)) {
        let csv = to_tfod_csv_string(&dataset).expect("serialize tfod");
        let restored = from_tfod_csv_str(&csv).expect("parse tfod");

        let expected_images = proptest_helpers::used_image_file_names(&dataset)
            .expect("original dataset should have valid references");
        let restored_images: BTreeSet<_> = restored
            .images
            .iter()
            .map(|img| img.file_name.clone())
            .collect();

        prop_assert_eq!(restored_images, expected_images);

        let refs_ok = proptest_helpers::assert_valid_references(&restored);
        prop_assert!(refs_ok.is_ok(), "{}", refs_ok.unwrap_err());
    }
}
