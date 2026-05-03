#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_superannotate_json::from_superannotate_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_superannotate_slice(data);
});
