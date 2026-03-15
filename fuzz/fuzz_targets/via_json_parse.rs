#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_via_json::parse_via_json_slice;

fuzz_target!(|data: &[u8]| {
    let _ = parse_via_json_slice(data);
});
