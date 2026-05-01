use std::collections::BTreeSet;

use panlabel::ir::io_tfrecord::{from_tfrecord_slice, to_tfrecord_vec};
use proptest::prelude::*;

mod proptest_helpers;

proptest! {
    #![proptest_config(proptest_helpers::proptest_config())]

    #[test]
    fn tfrecord_roundtrip_preserves_annotation_semantics(dataset in proptest_helpers::arb_dataset_annotated(5, 5, 20)) {
        let bytes = to_tfrecord_vec(&dataset).expect("serialize tfrecord");
        let restored = from_tfrecord_slice(&bytes).expect("parse tfrecord");

        let res = proptest_helpers::assert_annotations_equivalent(
            &dataset,
            &restored,
            proptest_helpers::EPS_TFOD,
        );
        prop_assert!(res.is_ok(), "{}", res.unwrap_err());
    }

    #[test]
    fn tfrecord_roundtrip_preserves_unannotated_images(dataset in proptest_helpers::arb_dataset_full(5, 5, 20)) {
        let bytes = to_tfrecord_vec(&dataset).expect("serialize tfrecord");
        let restored = from_tfrecord_slice(&bytes).expect("parse tfrecord");

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

        let refs_ok = proptest_helpers::assert_valid_references(&restored);
        prop_assert!(refs_ok.is_ok(), "{}", refs_ok.unwrap_err());
    }
}
