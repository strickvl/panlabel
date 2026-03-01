//! Fuzz target for YOLO single-line label parsing.
//!
//! This fuzzer feeds arbitrary UTF-8 lines to the YOLO line parser,
//! checking for panics, crashes, or hangs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_yolo::fuzz_parse_label_line;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10 * 1024 * 1024 {
        return;
    }

    let Ok(line) = std::str::from_utf8(data) else {
        return;
    };

    let _ = fuzz_parse_label_line(line);
});
