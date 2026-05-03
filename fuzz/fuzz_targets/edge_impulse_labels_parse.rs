#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_edge_impulse_labels::from_edge_impulse_labels_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_edge_impulse_labels_slice(data);
});
