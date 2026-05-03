#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_tfrecord::from_tfrecord_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_tfrecord_slice(data);
});
