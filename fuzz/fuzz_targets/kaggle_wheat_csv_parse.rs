#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = panlabel::ir::io_kaggle_wheat_csv::from_kaggle_wheat_csv_slice(data);
});
