use panlabel::ir::io_yolo::{read_yolo_dir, write_yolo_dir};
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
}

fn materialize_images_for_reread(root: &std::path::Path, dataset: &panlabel::ir::Dataset) {
    for image in &dataset.images {
        let image_path = root.join("images").join(&image.file_name);
        common::write_bmp(&image_path, image.width, image.height);
    }
}
