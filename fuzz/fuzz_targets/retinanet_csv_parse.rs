#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_retinanet_csv::parse_retinanet_csv_slice;

fuzz_target!(|data: &[u8]| {
    let _ = parse_retinanet_csv_slice(data);
});
