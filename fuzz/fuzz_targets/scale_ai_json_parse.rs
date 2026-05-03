#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_scale_ai_json::from_scale_ai_json_str;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = from_scale_ai_json_str(input);
    }
});
