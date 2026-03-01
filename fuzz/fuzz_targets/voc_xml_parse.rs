//! Fuzz target for VOC XML parsing.
//!
//! This fuzzer feeds arbitrary byte sequences to the VOC XML parser,
//! checking for panics, crashes, or hangs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_voc_xml::from_voc_xml_slice;

fuzz_target!(|data: &[u8]| {
    // Cap input size to avoid excessive memory usage.
    if data.len() > 10 * 1024 * 1024 {
        return;
    }

    let _ = from_voc_xml_slice(data);
});
