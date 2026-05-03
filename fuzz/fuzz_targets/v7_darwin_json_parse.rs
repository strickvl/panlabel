#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_v7_darwin_json::from_v7_darwin_json_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_v7_darwin_json_slice(data);
});
