#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_via_csv::from_via_csv_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_via_csv_slice(data);
});
