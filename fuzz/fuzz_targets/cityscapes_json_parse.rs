#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_cityscapes_json::from_cityscapes_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_cityscapes_slice(data);
});
