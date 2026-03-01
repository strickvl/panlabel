//! Fuzz target for CVAT XML parsing.
//!
//! This fuzzer feeds arbitrary byte sequences to the CVAT XML parser,
//! checking for panics, crashes, or hangs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_cvat_xml::from_cvat_xml_slice;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10 * 1024 * 1024 {
        return;
    }
    let _ = from_cvat_xml_slice(data);
});
