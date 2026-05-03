#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_supervisely_json::from_supervisely_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_supervisely_slice(data);
});
