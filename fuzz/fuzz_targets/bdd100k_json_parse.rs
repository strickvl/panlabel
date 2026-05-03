#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_bdd100k_json::from_bdd100k_json_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_bdd100k_json_slice(data);
});
