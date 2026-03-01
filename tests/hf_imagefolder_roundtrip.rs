//! Integration tests for HF ImageFolder metadata support.

use std::fs;
use std::path::Path;

use panlabel::ir::io_hf_imagefolder::{
    read_hf_imagefolder, read_hf_imagefolder_with_options, write_hf_imagefolder, HfBboxFormat,
    HfReadOptions,
};
use panlabel::PanlabelError;

mod common;
use common::write_bmp;

fn create_hf_dataset(root: &Path, xyxy: bool) {
    fs::create_dir_all(root).expect("create root");
    write_bmp(&root.join("img_a.bmp"), 100, 80);
    write_bmp(&root.join("img_b.bmp"), 50, 40);

    let row_a = if xyxy {
        "[[10,20,50,70]]"
    } else {
        "[[10,20,40,50]]"
    };
    let row_b = if xyxy {
        "[[2,3,12,18]]"
    } else {
        "[[2,3,10,15]]"
    };

    let jsonl = format!(
        "{{\"file_name\":\"img_a.bmp\",\"width\":100,\"height\":80,\"objects\":{{\"bbox\":{},\"categories\":[\"person\"]}}}}\n{{\"file_name\":\"img_b.bmp\",\"width\":50,\"height\":40,\"objects\":{{\"bbox\":{},\"categories\":[\"car\"]}}}}\n",
        row_a, row_b
    );

    fs::write(root.join("metadata.jsonl"), jsonl).expect("write metadata");
}

#[test]
fn read_hf_xywh_dataset() {
    let temp = tempfile::tempdir().expect("tempdir");
    create_hf_dataset(temp.path(), false);

    let dataset = read_hf_imagefolder(temp.path()).expect("read dataset");
    assert_eq!(dataset.images.len(), 2);
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.annotations.len(), 2);

    let bbox = &dataset.annotations[0].bbox;
    assert!((bbox.xmin() - 10.0).abs() < 1e-9);
    assert!((bbox.ymin() - 20.0).abs() < 1e-9);
    assert!((bbox.xmax() - 50.0).abs() < 1e-9);
    assert!((bbox.ymax() - 70.0).abs() < 1e-9);
}

#[test]
fn read_hf_xyxy_dataset_with_option() {
    let temp = tempfile::tempdir().expect("tempdir");
    create_hf_dataset(temp.path(), true);

    let options = HfReadOptions {
        bbox_format: HfBboxFormat::Xyxy,
        ..Default::default()
    };
    let dataset = read_hf_imagefolder_with_options(temp.path(), &options).expect("read dataset");

    let bbox = &dataset.annotations[0].bbox;
    assert!((bbox.xmin() - 10.0).abs() < 1e-9);
    assert!((bbox.ymin() - 20.0).abs() < 1e-9);
    assert!((bbox.xmax() - 50.0).abs() < 1e-9);
    assert!((bbox.ymax() - 70.0).abs() < 1e-9);
}

#[test]
fn hf_writer_then_reader_roundtrip_counts() {
    let temp = tempfile::tempdir().expect("tempdir");
    let input = temp.path().join("input");
    let output = temp.path().join("output");

    create_hf_dataset(&input, false);

    let dataset = read_hf_imagefolder(&input).expect("read input dataset");
    write_hf_imagefolder(&output, &dataset).expect("write hf metadata");

    let restored = read_hf_imagefolder(&output).expect("read output dataset");
    assert_eq!(restored.images.len(), dataset.images.len());
    assert_eq!(restored.categories.len(), dataset.categories.len());
    assert_eq!(restored.annotations.len(), dataset.annotations.len());
}

#[test]
fn hf_reader_rejects_duplicate_file_names() {
    let temp = tempfile::tempdir().expect("tempdir");
    let jsonl = r#"{"file_name":"img.bmp","width":10,"height":10,"objects":{"bbox":[],"categories":[]}}
{"file_name":"img.bmp","width":10,"height":10,"objects":{"bbox":[],"categories":[]}}
"#;
    fs::write(temp.path().join("metadata.jsonl"), jsonl).expect("write metadata");

    let err = read_hf_imagefolder(temp.path()).expect_err("expected duplicate-name failure");
    match err {
        PanlabelError::HfLayoutInvalid { message, .. } => {
            assert!(message.contains("duplicate file_name"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
