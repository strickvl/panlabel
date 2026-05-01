use std::path::{Component, Path};

use serde_json::Value;

use super::{BBoxXYXY, Pixel};
use crate::error::PanlabelError;

pub(super) fn has_json_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
}

pub(super) fn scalar_to_string(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

pub(super) fn required_u32(
    value: Option<&Value>,
    path: &Path,
    field: &str,
    invalid: fn(&Path, String) -> PanlabelError,
) -> Result<u32, PanlabelError> {
    let number = value.and_then(Value::as_u64).ok_or_else(|| {
        invalid(
            path,
            format!("missing or invalid unsigned integer field '{field}'"),
        )
    })?;
    u32::try_from(number)
        .map_err(|_| invalid(path, format!("field '{field}' is too large for u32")))
}

pub(super) fn required_f64(
    value: Option<&Value>,
    path: &Path,
    field: impl AsRef<str>,
    invalid: fn(&Path, String) -> PanlabelError,
) -> Result<f64, PanlabelError> {
    let field = field.as_ref();
    let number = value
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(path, format!("missing or invalid number field '{field}'")))?;
    if number.is_finite() {
        Ok(number)
    } else {
        Err(invalid(path, format!("field '{field}' must be finite")))
    }
}

pub(super) fn optional_finite_f64(
    value: Option<&Value>,
    path: &Path,
    field: impl AsRef<str>,
    invalid: fn(&Path, String) -> PanlabelError,
) -> Result<Option<f64>, PanlabelError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let number = value
        .as_f64()
        .ok_or_else(|| invalid(path, format!("field '{}' must be a number", field.as_ref())))?;
    if number.is_finite() {
        Ok(Some(number))
    } else {
        Err(invalid(
            path,
            format!("field '{}' must be finite", field.as_ref()),
        ))
    }
}

pub(super) fn parse_point_pair(
    point: &Value,
    path: &Path,
    field: String,
    invalid: fn(&Path, String) -> PanlabelError,
) -> Result<[f64; 2], PanlabelError> {
    let array = point
        .as_array()
        .ok_or_else(|| invalid(path, format!("{field} must be a [x, y] array")))?;
    if array.len() != 2 {
        return Err(invalid(
            path,
            format!("{field} must have exactly 2 numbers"),
        ));
    }
    Ok([
        required_f64(array.first(), path, format!("{field}[0]"), invalid)?,
        required_f64(array.get(1), path, format!("{field}[1]"), invalid)?,
    ])
}

pub(super) fn envelope(points: &[[f64; 2]]) -> BBoxXYXY<Pixel> {
    let mut xmin = f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for [x, y] in points {
        xmin = xmin.min(*x);
        ymin = ymin.min(*y);
        xmax = xmax.max(*x);
        ymax = ymax.max(*y);
    }
    BBoxXYXY::<Pixel>::from_xyxy(xmin, ymin, xmax, ymax)
}

pub(super) fn reject_unsafe_relative_path(
    file_name: &str,
    path: &Path,
    invalid: fn(&Path, String) -> PanlabelError,
) -> Result<(), PanlabelError> {
    let candidate = Path::new(file_name);
    if file_name.trim().is_empty() {
        return Err(invalid(
            path,
            "image file_name must not be empty".to_string(),
        ));
    }
    for component in candidate.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(invalid(
                    path,
                    format!(
                        "unsafe image file_name '{file_name}' cannot be used as an output annotation path"
                    ),
                ));
            }
        }
    }
    Ok(())
}
