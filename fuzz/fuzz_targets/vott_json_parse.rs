#![no_main]
use std::path::Path;

use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_vott_json::from_vott_json_str_with_base_dir;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = from_vott_json_str_with_base_dir(input, Path::new("."));
    }
});
