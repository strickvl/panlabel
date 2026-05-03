#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_openlabel_json::from_openlabel_json_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_openlabel_json_slice(data);
});
