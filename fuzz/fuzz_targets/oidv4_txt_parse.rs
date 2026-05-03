#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_oidv4_txt::parse_oidv4_txt_slice;

fuzz_target!(|data: &[u8]| {
    let _ = parse_oidv4_txt_slice(data);
});
