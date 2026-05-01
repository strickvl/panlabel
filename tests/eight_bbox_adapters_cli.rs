mod common;

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use std::fs;
use tempfile::tempdir;

#[test]
fn list_formats_includes_eight_new_bbox_adapters() {
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args(["list-formats", "--output-format", "json"]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("datumaro"))
        .stdout(predicate::str::contains("wider-face"))
        .stdout(predicate::str::contains("oidv4"))
        .stdout(predicate::str::contains("bdd100k"))
        .stdout(predicate::str::contains("v7-darwin"))
        .stdout(predicate::str::contains("edge-impulse"))
        .stdout(predicate::str::contains("openlabel"))
        .stdout(predicate::str::contains("via-csv"));
}

#[test]
fn via_csv_alias_converts_but_via_remains_json_name() {
    let dir = tempdir().unwrap();
    let out = dir.path().join("via.csv");
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "ir-json",
        "-t",
        "vgg-via-csv",
        "-i",
        "tests/fixtures/sample_valid.ir.json",
        "-o",
        out.to_str().unwrap(),
        "--allow-lossy",
    ]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Converted"));
    let csv = fs::read_to_string(out).unwrap();
    assert!(csv.starts_with("filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes"));

    let mut list = cargo_bin_cmd!("panlabel");
    list.args(["list-formats", "--output-format", "json"]);
    list.assert()
        .success()
        .stdout(predicate::str::contains("\"name\":\"via\""))
        .stdout(predicate::str::contains("\"name\":\"via-csv\""));
}

#[test]
fn auto_detects_datumaro_bdd_openlabel_edge_and_via_csv() {
    let dir = tempdir().unwrap();
    let datumaro = dir.path().join("datumaro.json");
    fs::write(&datumaro, r#"{"categories":{"label":{"labels":[{"name":"cat"}]}},"items":[{"id":"a.bmp","image":{"path":"a.bmp","size":[80,100]},"annotations":[{"type":"bbox","bbox":[1,2,3,4],"label_id":0}]}]}"#).unwrap();
    let bdd = dir.path().join("bdd.json");
    fs::write(&bdd, r#"[{"name":"a.bmp","width":100,"height":80,"labels":[{"category":"cat","box2d":{"x1":1,"y1":2,"x2":4,"y2":6}}]}]"#).unwrap();
    let openlabel = dir.path().join("openlabel.json");
    fs::write(&openlabel, r#"{"openlabel":{"objects":{"1":{"type":"cat"}},"frames":{"0":{"frame_properties":{"file_name":"a.bmp","width":100,"height":80},"objects":{"1":{"object_data":{"bbox":[{"name":"cat","val":[2.5,4,3,4]}]}}}}}}}"#).unwrap();
    let edge_dir = dir.path().join("edge");
    fs::create_dir_all(&edge_dir).unwrap();
    fs::write(edge_dir.join("bounding_boxes.labels"), r#"{"version":1,"type":"bounding-box-labels","files":[{"path":"a.bmp","boundingBoxes":[{"label":"cat","x":1,"y":2,"width":3,"height":4}]}]}"#).unwrap();
    fs::write(dir.path().join("a.bmp"), common::bmp_bytes(100, 80)).unwrap();
    let via = dir.path().join("via.csv");
    fs::write(&via, "filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\na.bmp,,{},1,0,\"{\"\"name\"\":\"\"rect\"\",\"\"x\"\":1,\"\"y\"\":2,\"\"width\"\":3,\"\"height\"\":4}\",\"{\"\"label\"\":\"\"cat\"\"}\"\n").unwrap();

    for input in [&datumaro, &bdd, &openlabel, &via] {
        let out = dir.path().join(format!(
            "{}.ir.json",
            input.file_stem().unwrap().to_string_lossy()
        ));
        let mut cmd = cargo_bin_cmd!("panlabel");
        cmd.args([
            "convert",
            "-f",
            "auto",
            "-t",
            "ir-json",
            "-i",
            input.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ]);
        cmd.assert().success();
    }
    let edge_out = dir.path().join("edge.ir.json");
    let mut cmd = cargo_bin_cmd!("panlabel");
    cmd.args([
        "convert",
        "-f",
        "auto",
        "-t",
        "ir-json",
        "-i",
        edge_dir.to_str().unwrap(),
        "-o",
        edge_out.to_str().unwrap(),
    ]);
    cmd.assert().success();
}
