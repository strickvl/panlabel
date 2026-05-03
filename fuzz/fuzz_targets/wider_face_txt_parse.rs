#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_wider_face_txt::parse_wider_face_txt_slice;

fuzz_target!(|data: &[u8]| {
    let _ = parse_wider_face_txt_slice(data);
});
