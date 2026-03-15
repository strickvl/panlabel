#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = panlabel::ir::io_openimages_csv::parse_openimages_csv_slice(data);
});
