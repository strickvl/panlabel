#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_unity_perception_json::from_unity_perception_json_str;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = from_unity_perception_json_str(input);
    }
});
