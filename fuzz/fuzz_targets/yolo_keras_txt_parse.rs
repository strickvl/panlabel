#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_yolo_keras_txt::parse_yolo_keras_txt_str;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = parse_yolo_keras_txt_str(input);
    }
});
