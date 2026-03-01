//! Fuzz target for TFOD CSV parsing.
//!
//! This fuzzer feeds arbitrary byte sequences to the TFOD CSV parser,
//! checking for panics, crashes, or hangs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_tfod_csv::from_tfod_csv_slice;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10 * 1024 * 1024 {
        return;
    }

    let _ = from_tfod_csv_slice(data);
});
