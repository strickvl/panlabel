use std::fs;
use std::path::{Component, Path};

use crate::error::PanlabelError;

pub(crate) fn basename_from_uri_or_path(raw: &str) -> Option<String> {
    let without_fragment = raw.split('#').next().unwrap_or(raw);
    let clean = without_fragment
        .split('?')
        .next()
        .unwrap_or(without_fragment)
        .trim();
    if clean.is_empty() {
        return None;
    }
    let stripped = clean
        .strip_prefix("file://")
        .or_else(|| clean.strip_prefix("file:"))
        .unwrap_or(clean)
        .trim();
    if stripped.is_empty() {
        return None;
    }
    stripped
        .rsplit(['/', '\\'])
        .find(|part| !part.trim().is_empty())
        .map(str::to_string)
        .or_else(|| Some(stripped.to_string()))
}

pub(crate) fn has_extension(path: &Path, expected_extension: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(expected_extension))
        .unwrap_or(false)
}

pub(crate) fn has_json_extension(path: &Path) -> bool {
    has_extension(path, "json")
}

pub(crate) fn has_json_lines_extension(path: &Path) -> bool {
    has_extension(path, "jsonl") || has_extension(path, "ndjson")
}

pub(crate) fn is_safe_relative_image_ref(image_ref: &str) -> bool {
    let image_ref = image_ref.trim();
    if image_ref.is_empty() || image_ref.starts_with('/') || image_ref.starts_with('\\') {
        return false;
    }

    let normalized = image_ref.replace('\\', "/");
    let first_component = normalized.split('/').next().unwrap_or_default();
    if first_component.len() == 2 {
        let bytes = first_component.as_bytes();
        if bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            return false;
        }
    }

    if Path::new(image_ref).components().any(|component| {
        matches!(
            component,
            Component::Prefix(_) | Component::RootDir | Component::ParentDir
        )
    }) {
        return false;
    }
    !normalized.split('/').any(|part| part == "..")
}

pub(crate) fn normalize_path_separators(path: &str) -> String {
    path.replace('\\', "/")
}

pub(crate) fn write_images_readme(output_dir: &Path, contents: &str) -> Result<(), PanlabelError> {
    let images_dir = output_dir.join("images");
    fs::create_dir_all(&images_dir).map_err(PanlabelError::Io)?;
    fs::write(images_dir.join("README.txt"), contents).map_err(PanlabelError::Io)
}
