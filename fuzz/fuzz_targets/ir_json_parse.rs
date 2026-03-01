//! Fuzz target for IR JSON parsing.
//!
//! This fuzzer feeds arbitrary byte sequences to the IR JSON parser,
//! checking for panics, crashes, or hangs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_json::from_json_slice;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10 * 1024 * 1024 {
        return;
    }

    let _ = from_json_slice(data);
});
