use std::fs;
use std::path::Path;

pub fn bmp_bytes(width: u32, height: u32) -> Vec<u8> {
    let row_stride = (width * 3).div_ceil(4) * 4;
    let pixel_array_size = row_stride * height;
    let file_size = 54 + pixel_array_size;

    let mut bytes = Vec::with_capacity(file_size as usize);
    bytes.extend_from_slice(b"BM");
    bytes.extend_from_slice(&file_size.to_le_bytes());
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    bytes.extend_from_slice(&54u32.to_le_bytes());

    bytes.extend_from_slice(&40u32.to_le_bytes());
    bytes.extend_from_slice(&(width as i32).to_le_bytes());
    bytes.extend_from_slice(&(height as i32).to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&24u16.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&pixel_array_size.to_le_bytes());
    bytes.extend_from_slice(&2835u32.to_le_bytes());
    bytes.extend_from_slice(&2835u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());

    bytes.resize(file_size as usize, 0);
    bytes
}

pub fn write_bmp(path: &Path, width: u32, height: u32) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dir");
    }
    fs::write(path, bmp_bytes(width, height)).expect("write bmp file");
}
