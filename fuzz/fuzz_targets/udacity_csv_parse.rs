#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = panlabel::ir::io_udacity_csv::from_udacity_csv_slice(data);
});
