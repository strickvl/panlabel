#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = panlabel::ir::io_automl_vision_csv::parse_automl_vision_csv_slice(data);
});
