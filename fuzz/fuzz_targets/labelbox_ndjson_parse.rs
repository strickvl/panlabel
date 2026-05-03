#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_labelbox_json::from_labelbox_ndjson_str;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = from_labelbox_ndjson_str(input);
    }
});
