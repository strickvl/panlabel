#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_kitti::from_kitti_slice;

fuzz_target!(|data: &[u8]| {
    let _ = from_kitti_slice(data);
});
