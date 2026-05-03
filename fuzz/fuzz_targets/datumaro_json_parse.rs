#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_datumaro_json::from_datumaro_json_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_datumaro_json_slice(data);
});
