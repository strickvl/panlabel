use panlabel::ir::io_coco_json::{from_coco_str, to_coco_string};
use panlabel::ir::io_json::{from_json_str, to_json_string};
use panlabel::ir::io_label_studio_json::{from_label_studio_str, to_label_studio_string};
use panlabel::ir::io_tfod_csv::{from_tfod_csv_str, to_tfod_csv_string};
use panlabel::ir::io_voc_xml::{read_voc_dir, write_voc_dir};
use panlabel::ir::io_yolo::{read_yolo_dir, write_yolo_dir};
use proptest::prelude::*;

mod common;
mod proptest_helpers;

proptest! {
    #![proptest_config(proptest_helpers::proptest_config())]

    #[test]
    fn each_format_annotations_are_subset_of_ir_reference(dataset in proptest_helpers::arb_dataset_full(3, 3, 8)) {
        let ir_reference = from_json_str(&to_json_string(&dataset).expect("serialize ir"))
            .expect("parse ir");

        let coco = from_coco_str(&to_coco_string(&dataset).expect("serialize coco")).expect("parse coco");
        let coco_res = proptest_helpers::assert_annotations_subset(&coco, &ir_reference, proptest_helpers::EPS_COCO);
        prop_assert!(coco_res.is_ok(), "COCO subset failure: {}", coco_res.unwrap_err());

        let tfod = from_tfod_csv_str(&to_tfod_csv_string(&dataset).expect("serialize tfod")).expect("parse tfod");
        let tfod_res = proptest_helpers::assert_annotations_subset(&tfod, &ir_reference, proptest_helpers::EPS_TFOD);
        prop_assert!(tfod_res.is_ok(), "TFOD subset failure: {}", tfod_res.unwrap_err());

        let ls = from_label_studio_str(&to_label_studio_string(&dataset).expect("serialize label-studio")).expect("parse label-studio");
        let ls_res = proptest_helpers::assert_annotations_subset(&ls, &ir_reference, proptest_helpers::EPS_LABEL_STUDIO);
        prop_assert!(ls_res.is_ok(), "Label Studio subset failure: {}", ls_res.unwrap_err());

        let voc_temp = tempfile::tempdir().expect("create voc tempdir");
        write_voc_dir(voc_temp.path(), &dataset).expect("write voc");
        let voc = read_voc_dir(voc_temp.path()).expect("read voc");
        let voc_res = proptest_helpers::assert_annotations_subset(&voc, &ir_reference, proptest_helpers::EPS_VOC);
        prop_assert!(voc_res.is_ok(), "VOC subset failure: {}", voc_res.unwrap_err());

        let yolo_temp = tempfile::tempdir().expect("create yolo tempdir");
        write_yolo_dir(yolo_temp.path(), &dataset).expect("write yolo");
        materialize_images_for_reread(yolo_temp.path(), &dataset);
        let yolo = read_yolo_dir(yolo_temp.path()).expect("read yolo");
        let yolo_res = proptest_helpers::assert_annotations_subset(
            &yolo,
            &ir_reference,
            proptest_helpers::eps_yolo_for_dataset(&ir_reference),
        );
        prop_assert!(yolo_res.is_ok(), "YOLO subset failure: {}", yolo_res.unwrap_err());
    }
}

fn materialize_images_for_reread(root: &std::path::Path, dataset: &panlabel::ir::Dataset) {
    for image in &dataset.images {
        let image_path = root.join("images").join(&image.file_name);
        common::write_bmp(&image_path, image.width, image.height);
    }
}
