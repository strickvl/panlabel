#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = panlabel::ir::io_labelme_json::from_labelme_slice(data);
});
