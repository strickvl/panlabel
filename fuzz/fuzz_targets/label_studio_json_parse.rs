//! Fuzz target for Label Studio JSON parsing.
//!
//! This fuzzer feeds arbitrary byte sequences to the Label Studio parser,
//! checking for panics, crashes, or hangs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_label_studio_json::from_label_studio_slice;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10 * 1024 * 1024 {
        return;
    }

    let _ = from_label_studio_slice(data);
});
