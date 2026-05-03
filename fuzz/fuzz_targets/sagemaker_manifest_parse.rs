#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_sagemaker_manifest::from_sagemaker_manifest_str;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = from_sagemaker_manifest_str(input);
    }
});
