//! Fuzz target for COCO JSON parsing.
//!
//! This fuzzer feeds arbitrary byte sequences to the COCO JSON parser,
//! checking for panics, buffer overflows, or other undefined behavior.
//!
//! Run with:
//!   cargo +nightly fuzz run coco_json_parse
//!
//! Or with a corpus:
//!   cargo +nightly fuzz run coco_json_parse fuzz/corpus/coco_json_parse/

#![no_main]

use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_coco_json::from_coco_slice;

fuzz_target!(|data: &[u8]| {
    // Cap input size to avoid OOM on very large inputs.
    // 10MB is generous for JSON annotation files.
    if data.len() > 10 * 1024 * 1024 {
        return;
    }

    // Try to parse the data. We don't care about errors—
    // we only care about panics, crashes, or hangs.
    let _ = from_coco_slice(data);
});
