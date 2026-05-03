#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_cloud_annotations_json::parse_cloud_annotations_slice;

fuzz_target!(|data: &[u8]| {
    let _ = parse_cloud_annotations_slice(data);
});
