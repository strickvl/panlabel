#![no_main]
use libfuzzer_sys::fuzz_target;
use panlabel::ir::io_marmot_xml::from_marmot_xml_str;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = from_marmot_xml_str(input, "fuzz.png", 100, 100);
    }
});
