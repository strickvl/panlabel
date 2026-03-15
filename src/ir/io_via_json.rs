//! VGG Image Annotator (VIA) JSON format reader and writer.
//!
//! VIA is a widely used browser-based annotation tool. Its JSON export is an
//! object keyed by arbitrary strings (typically `<filename><size>`), where each
//! value describes one image and its `regions` (annotations).
//!
//! # Format Reference
//!
//! ```json
//! {
//!   "img001.jpg1234": {
//!     "filename": "img001.jpg",
//!     "size": 1234,
//!     "regions": [
//!       {
//!         "shape_attributes": { "name": "rect", "x": 10, "y": 20, "width": 80, "height": 60 },
//!         "region_attributes": { "label": "cat" }
//!       }
//!     ],
//!     "file_attributes": {}
//!   }
//! }
//! ```
//!
//! VIA does **not** store image dimensions in the JSON — the reader resolves
//! them from image files on disk, relative to the JSON file's parent directory.
//!
//! # Deterministic Output
//!
//! The writer produces deterministic output: entries are sorted by filename,
//! regions within each entry are sorted by annotation ID.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::model::{Annotation, Category, Dataset, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

// ============================================================================
// VIA Schema Types (internal to this module)
// ============================================================================

/// Top-level VIA project: an object keyed by arbitrary strings.
#[derive(Debug, Deserialize)]
struct ViaProject(BTreeMap<String, ViaEntry>);

/// One image entry in a VIA JSON export.
#[derive(Debug, Deserialize)]
struct ViaEntry {
    filename: String,
    size: u64,
    #[serde(default)]
    regions: ViaRegions,
    #[serde(default)]
    file_attributes: BTreeMap<String, serde_json::Value>,
}

/// VIA exports regions as either an array or an object map keyed by region ID.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ViaRegions {
    Array(Vec<ViaRegion>),
    Map(BTreeMap<String, ViaRegion>),
}

impl Default for ViaRegions {
    fn default() -> Self {
        Self::Array(vec![])
    }
}

/// A single region (annotation) within a VIA entry.
#[derive(Debug, Deserialize)]
struct ViaRegion {
    shape_attributes: ViaShapeAttributes,
    #[serde(default)]
    region_attributes: BTreeMap<String, serde_json::Value>,
}

/// Shape attributes for a VIA region.
#[derive(Debug, Deserialize)]
struct ViaShapeAttributes {
    name: String,
    #[serde(default)]
    x: f64,
    #[serde(default)]
    y: f64,
    #[serde(default)]
    width: f64,
    #[serde(default)]
    height: f64,
}

// ============================================================================
// Writer Schema Types
// ============================================================================

#[derive(Debug, Serialize)]
struct ViaEntryOut {
    filename: String,
    size: u64,
    regions: Vec<ViaRegionOut>,
    file_attributes: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ViaRegionOut {
    shape_attributes: ViaShapeAttrsOut,
    region_attributes: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ViaShapeAttrsOut {
    name: String,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

// ============================================================================
// Public API
// ============================================================================

/// Reads a dataset from a VIA JSON file.
///
/// Image dimensions are resolved by probing local image files relative to
/// the JSON file's parent directory.
pub fn read_via_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    let project: ViaProject =
        serde_json::from_reader(reader).map_err(|source| PanlabelError::ViaJsonParse {
            path: path.to_path_buf(),
            source,
        })?;

    via_project_to_ir(project, base_dir, path)
}

/// Writes a dataset to a VIA JSON file.
pub fn write_via_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);

    let project = ir_to_via_project(dataset);

    serde_json::to_writer_pretty(writer, &project).map_err(|source| PanlabelError::ViaJsonWrite {
        path: path.to_path_buf(),
        source,
    })
}

/// Reads a dataset from a VIA JSON string, resolving images from `base_dir`.
pub fn from_via_json_str_with_base_dir(
    json: &str,
    base_dir: &Path,
) -> Result<Dataset, PanlabelError> {
    let project: ViaProject =
        serde_json::from_str(json).map_err(|source| PanlabelError::ViaJsonParse {
            path: base_dir.to_path_buf(),
            source,
        })?;

    via_project_to_ir(project, base_dir, base_dir)
}

/// Parses VIA JSON from a byte slice (schema-only, no image resolution).
///
/// Fuzz-only entrypoint: exercises JSON/schema parsing without requiring
/// image files on disk.
#[cfg(feature = "fuzzing")]
pub fn parse_via_json_slice(bytes: &[u8]) -> Result<(), serde_json::Error> {
    let _project: ViaProject = serde_json::from_slice(bytes)?;
    Ok(())
}

/// Writes a dataset to a VIA JSON string.
pub fn to_via_json_string(dataset: &Dataset) -> Result<String, serde_json::Error> {
    let project = ir_to_via_project(dataset);
    serde_json::to_string_pretty(&project)
}

// ============================================================================
// Conversion: VIA -> IR
// ============================================================================

fn via_project_to_ir(
    project: ViaProject,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    let entries = project.0;

    // Validate: no duplicate filenames across entries
    let mut seen_filenames: BTreeSet<String> = BTreeSet::new();
    for entry in entries.values() {
        if entry.filename.is_empty() {
            return Err(PanlabelError::ViaJsonInvalid {
                path: source_path.to_path_buf(),
                message: "empty 'filename' field".to_string(),
            });
        }
        if !seen_filenames.insert(entry.filename.clone()) {
            return Err(PanlabelError::ViaJsonInvalid {
                path: source_path.to_path_buf(),
                message: format!("duplicate filename: '{}'", entry.filename),
            });
        }
    }

    // Sort entries by filename for deterministic image ID assignment
    let mut sorted_entries: Vec<&ViaEntry> = entries.values().collect();
    sorted_entries.sort_by(|a, b| a.filename.cmp(&b.filename));

    // First pass: collect all labels for category assignment
    let mut label_set: BTreeSet<String> = BTreeSet::new();
    for entry in &sorted_entries {
        let regions = collect_rect_regions_with_attrs(entry, source_path);
        for (_, label, _) in &regions {
            label_set.insert(label.clone());
        }
    }

    // Build category map: label -> CategoryId (1-based, sorted)
    let label_to_cat: BTreeMap<String, CategoryId> = label_set
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), CategoryId::new((i + 1) as u64)))
        .collect();

    let categories: Vec<Category> = label_set
        .iter()
        .enumerate()
        .map(|(i, name)| Category::new((i + 1) as u64, name.clone()))
        .collect();

    // Second pass: build images and annotations
    let mut images = Vec::new();
    let mut annotations = Vec::new();
    let mut ann_id_counter: u64 = 1;

    for (img_idx, entry) in sorted_entries.iter().enumerate() {
        let image_id = ImageId::new((img_idx + 1) as u64);

        // Resolve image dimensions from disk
        let (width, height) = resolve_image_dimensions(base_dir, &entry.filename, source_path)?;

        let file_name = entry.filename.replace('\\', "/");

        let mut img = Image::new(image_id, file_name, width, height);

        // Store VIA-specific attributes
        img.attributes
            .insert("via_size_bytes".to_string(), entry.size.to_string());

        // Store scalar file_attributes
        for (key, val) in &entry.file_attributes {
            if let Some(s) = scalar_value_to_string(val) {
                img.attributes.insert(format!("via_file_attr_{key}"), s);
            }
        }

        images.push(img);

        // Process regions
        let rect_regions = collect_rect_regions_with_attrs(entry, source_path);
        for (region, label, extra_attrs) in rect_regions {
            let cat_id = label_to_cat[&label];
            let bbox = BBoxXYXY::<Pixel>::from_xywh(
                region.shape_attributes.x,
                region.shape_attributes.y,
                region.shape_attributes.width,
                region.shape_attributes.height,
            );

            let mut ann =
                Annotation::new(AnnotationId::new(ann_id_counter), image_id, cat_id, bbox);

            // Store non-label region_attributes as annotation attributes
            for (key, val) in &extra_attrs {
                ann.attributes
                    .insert(format!("via_region_attr_{key}"), val.clone());
            }

            annotations.push(ann);
            ann_id_counter += 1;
        }
    }

    Ok(Dataset {
        images,
        categories,
        annotations,
        ..Default::default()
    })
}

/// Attempt to resolve a label from `region_attributes`.
///
/// Precedence:
/// 1. `region_attributes["label"]` (scalar)
/// 2. `region_attributes["class"]` (scalar)
/// 3. If exactly one scalar attribute exists, use its value
/// 4. Otherwise `None`
fn resolve_label(region_attributes: &BTreeMap<String, serde_json::Value>) -> Option<String> {
    // Try "label" key
    if let Some(val) = region_attributes.get("label") {
        if let Some(s) = scalar_value_to_string(val) {
            if !s.is_empty() {
                return Some(s);
            }
        }
    }

    // Try "class" key
    if let Some(val) = region_attributes.get("class") {
        if let Some(s) = scalar_value_to_string(val) {
            if !s.is_empty() {
                return Some(s);
            }
        }
    }

    // Exactly one scalar attribute -> use its value
    let scalars: Vec<String> = region_attributes
        .values()
        .filter_map(scalar_value_to_string)
        .filter(|s| !s.is_empty())
        .collect();

    if scalars.len() == 1 {
        return Some(scalars.into_iter().next().unwrap());
    }

    None
}

/// Convert a `serde_json::Value` to a `String` if it is a scalar
/// (string, number, or bool). Returns `None` for null, arrays, and objects.
fn scalar_value_to_string(val: &serde_json::Value) -> Option<String> {
    match val {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Collect rect regions with resolvable labels and their non-label scalar
/// attributes. Non-rect shapes and unlabeled regions are skipped with warnings.
fn collect_rect_regions_with_attrs<'a>(
    entry: &'a ViaEntry,
    source_path: &Path,
) -> Vec<(&'a ViaRegion, String, BTreeMap<String, String>)> {
    let regions = match &entry.regions {
        ViaRegions::Array(v) => v.iter().collect::<Vec<_>>(),
        ViaRegions::Map(m) => m.values().collect::<Vec<_>>(),
    };

    let mut result = Vec::new();
    for region in regions {
        if region.shape_attributes.name != "rect" {
            eprintln!(
                "panlabel: warning: skipping non-rect shape '{}' in '{}' ({})",
                region.shape_attributes.name,
                entry.filename,
                source_path.display(),
            );
            continue;
        }

        let label = match resolve_label(&region.region_attributes) {
            Some(l) => l,
            None => {
                eprintln!(
                    "panlabel: warning: skipping rect region with no resolvable label in '{}' ({})",
                    entry.filename,
                    source_path.display(),
                );
                continue;
            }
        };

        // Determine which key was used for the label so we can exclude it
        let label_key = determine_label_key(&region.region_attributes);

        let mut extra_attrs = BTreeMap::new();
        for (key, val) in &region.region_attributes {
            if Some(key.as_str()) == label_key {
                continue;
            }
            if let Some(s) = scalar_value_to_string(val) {
                extra_attrs.insert(key.clone(), s);
            }
        }

        result.push((region, label, extra_attrs));
    }
    result
}

/// Determine which key was used to resolve the label (so it can be excluded
/// from extra attributes). Mirrors `resolve_label` precedence.
fn determine_label_key(region_attributes: &BTreeMap<String, serde_json::Value>) -> Option<&str> {
    if let Some(val) = region_attributes.get("label") {
        if scalar_value_to_string(val).is_some_and(|s| !s.is_empty()) {
            return Some("label");
        }
    }
    if let Some(val) = region_attributes.get("class") {
        if scalar_value_to_string(val).is_some_and(|s| !s.is_empty()) {
            return Some("class");
        }
    }

    let scalars: Vec<&str> = region_attributes
        .iter()
        .filter(|(_, v)| scalar_value_to_string(v).is_some_and(|s| !s.is_empty()))
        .map(|(k, _)| k.as_str())
        .collect();

    if scalars.len() == 1 {
        Some(scalars[0])
    } else {
        None
    }
}

/// Resolve image dimensions by probing the filesystem.
///
/// Precedence: `base_dir/<filename>` then `base_dir/images/<filename>`.
fn resolve_image_dimensions(
    base_dir: &Path,
    image_ref: &str,
    source_path: &Path,
) -> Result<(u32, u32), PanlabelError> {
    // Reject absolute paths and path traversal
    if image_ref.starts_with('/') || image_ref.starts_with('\\') || image_ref.contains("..") {
        return Err(PanlabelError::ViaJsonInvalid {
            path: source_path.to_path_buf(),
            message: format!(
                "image reference '{}' must be a relative path without '..'",
                image_ref
            ),
        });
    }

    let candidate1 = base_dir.join(image_ref);
    let candidate2 = base_dir.join("images").join(image_ref);

    if let Ok(size) = imagesize::size(&candidate1) {
        return Ok((size.width as u32, size.height as u32));
    }
    if let Ok(size) = imagesize::size(&candidate2) {
        return Ok((size.width as u32, size.height as u32));
    }

    Err(PanlabelError::ViaImageNotFound {
        path: source_path.to_path_buf(),
        image_ref: image_ref.to_string(),
    })
}

// ============================================================================
// Conversion: IR -> VIA
// ============================================================================

fn ir_to_via_project(dataset: &Dataset) -> BTreeMap<String, ViaEntryOut> {
    // Build category lookup
    let cat_map: BTreeMap<CategoryId, &str> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();

    // Group annotations by image_id
    let mut anns_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    // Sort images by file_name for deterministic output
    let mut sorted_images: Vec<&Image> = dataset.images.iter().collect();
    sorted_images.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    let mut project = BTreeMap::new();

    for img in sorted_images {
        // Reconstruct VIA size from attributes, default to 0
        let via_size: u64 = img
            .attributes
            .get("via_size_bytes")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // Reconstruct file_attributes from via_file_attr_* image attributes
        let mut file_attributes: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        for (key, val) in &img.attributes {
            if let Some(stripped) = key.strip_prefix("via_file_attr_") {
                file_attributes
                    .insert(stripped.to_string(), serde_json::Value::String(val.clone()));
            }
        }

        // Build regions
        let mut img_anns: Vec<&Annotation> =
            anns_by_image.get(&img.id).cloned().unwrap_or_default();
        img_anns.sort_by_key(|a| a.id);

        let regions: Vec<ViaRegionOut> = img_anns
            .into_iter()
            .map(|ann| {
                let (x, y, w, h) = ann.bbox.to_xywh();
                let label = cat_map
                    .get(&ann.category_id)
                    .unwrap_or(&"unknown")
                    .to_string();

                let mut region_attributes: BTreeMap<String, serde_json::Value> = BTreeMap::new();
                region_attributes.insert("label".to_string(), serde_json::Value::String(label));

                // Reconstruct extra region attributes from via_region_attr_*
                for (key, val) in &ann.attributes {
                    if let Some(stripped) = key.strip_prefix("via_region_attr_") {
                        region_attributes
                            .insert(stripped.to_string(), serde_json::Value::String(val.clone()));
                    }
                }

                ViaRegionOut {
                    shape_attributes: ViaShapeAttrsOut {
                        name: "rect".to_string(),
                        x,
                        y,
                        width: w,
                        height: h,
                    },
                    region_attributes,
                }
            })
            .collect();

        // Key: <filename><size>
        let key = format!("{}{}", img.file_name, via_size);

        project.insert(
            key,
            ViaEntryOut {
                filename: img.file_name.clone(),
                size: via_size,
                regions,
                file_attributes,
            },
        );
    }

    project
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_via_json_array_regions() -> &'static str {
        r#"{
            "img001.jpg1234": {
                "filename": "img001.jpg",
                "size": 1234,
                "regions": [
                    {
                        "shape_attributes": { "name": "rect", "x": 10, "y": 20, "width": 80, "height": 60 },
                        "region_attributes": { "label": "cat" }
                    },
                    {
                        "shape_attributes": { "name": "rect", "x": 100, "y": 50, "width": 120, "height": 90 },
                        "region_attributes": { "label": "dog" }
                    }
                ],
                "file_attributes": {}
            },
            "img002.jpg5678": {
                "filename": "img002.jpg",
                "size": 5678,
                "regions": [],
                "file_attributes": { "source": "web" }
            }
        }"#
    }

    fn sample_via_json_map_regions() -> &'static str {
        r#"{
            "img001.jpg1234": {
                "filename": "img001.jpg",
                "size": 1234,
                "regions": {
                    "0": {
                        "shape_attributes": { "name": "rect", "x": 10, "y": 20, "width": 80, "height": 60 },
                        "region_attributes": { "label": "cat" }
                    },
                    "1": {
                        "shape_attributes": { "name": "rect", "x": 100, "y": 50, "width": 120, "height": 90 },
                        "region_attributes": { "label": "dog" }
                    }
                },
                "file_attributes": {}
            }
        }"#
    }

    #[test]
    fn parse_via_schema_array_regions() {
        let project: ViaProject =
            serde_json::from_str(sample_via_json_array_regions()).expect("parse failed");
        assert_eq!(project.0.len(), 2);

        let entry = &project.0["img001.jpg1234"];
        assert_eq!(entry.filename, "img001.jpg");
        assert_eq!(entry.size, 1234);
        match &entry.regions {
            ViaRegions::Array(v) => assert_eq!(v.len(), 2),
            ViaRegions::Map(_) => panic!("expected array regions"),
        }
    }

    #[test]
    fn parse_via_schema_map_regions() {
        let project: ViaProject =
            serde_json::from_str(sample_via_json_map_regions()).expect("parse failed");
        let entry = &project.0["img001.jpg1234"];
        match &entry.regions {
            ViaRegions::Map(m) => assert_eq!(m.len(), 2),
            ViaRegions::Array(_) => panic!("expected map regions"),
        }
    }

    #[test]
    fn parse_empty_project() {
        let project: ViaProject = serde_json::from_str("{}").expect("parse failed");
        assert!(project.0.is_empty());
    }

    #[test]
    fn resolve_label_precedence() {
        // "label" takes priority over "class"
        let mut attrs = BTreeMap::new();
        attrs.insert(
            "label".to_string(),
            serde_json::Value::String("cat".to_string()),
        );
        attrs.insert(
            "class".to_string(),
            serde_json::Value::String("feline".to_string()),
        );
        assert_eq!(resolve_label(&attrs), Some("cat".to_string()));

        // "class" used when "label" absent
        let mut attrs2 = BTreeMap::new();
        attrs2.insert(
            "class".to_string(),
            serde_json::Value::String("dog".to_string()),
        );
        assert_eq!(resolve_label(&attrs2), Some("dog".to_string()));

        // Single scalar fallback
        let mut attrs3 = BTreeMap::new();
        attrs3.insert(
            "species".to_string(),
            serde_json::Value::String("bird".to_string()),
        );
        assert_eq!(resolve_label(&attrs3), Some("bird".to_string()));

        // Multiple scalars, none named label/class -> None
        let mut attrs4 = BTreeMap::new();
        attrs4.insert("a".to_string(), serde_json::Value::String("x".to_string()));
        attrs4.insert("b".to_string(), serde_json::Value::String("y".to_string()));
        assert_eq!(resolve_label(&attrs4), None);

        // Empty attributes -> None
        let attrs5 = BTreeMap::new();
        assert_eq!(resolve_label(&attrs5), None);
    }

    #[test]
    fn scalar_value_conversion() {
        assert_eq!(
            scalar_value_to_string(&serde_json::Value::String("hi".into())),
            Some("hi".to_string())
        );
        assert_eq!(
            scalar_value_to_string(&serde_json::json!(42)),
            Some("42".to_string())
        );
        assert_eq!(
            scalar_value_to_string(&serde_json::json!(true)),
            Some("true".to_string())
        );
        assert_eq!(scalar_value_to_string(&serde_json::Value::Null), None);
        assert_eq!(scalar_value_to_string(&serde_json::json!([1, 2])), None);
        assert_eq!(scalar_value_to_string(&serde_json::json!({"a": 1})), None);
    }

    #[test]
    fn ir_to_via_roundtrip_string() {
        let dataset = Dataset {
            images: vec![
                Image::new(1u64, "a.jpg", 640, 480),
                Image::new(2u64, "b.jpg", 800, 600),
            ],
            categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
            annotations: vec![
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 90.0, 80.0),
                ),
                Annotation::new(
                    2u64,
                    1u64,
                    2u64,
                    BBoxXYXY::<Pixel>::from_xyxy(100.0, 50.0, 220.0, 140.0),
                ),
            ],
            ..Default::default()
        };

        let json = to_via_json_string(&dataset).expect("serialize failed");
        let project: BTreeMap<String, serde_json::Value> = serde_json::from_str(&json).unwrap();

        // Two entries, sorted by filename
        assert_eq!(project.len(), 2);
        assert!(project.contains_key("a.jpg0"));
        assert!(project.contains_key("b.jpg0"));

        // a.jpg has 2 annotations
        let a_entry = &project["a.jpg0"];
        let regions = a_entry["regions"].as_array().unwrap();
        assert_eq!(regions.len(), 2);

        // Check XYWH conversion: xyxy(10,20,90,80) -> x=10, y=20, w=80, h=60
        let shape = &regions[0]["shape_attributes"];
        assert_eq!(shape["name"], "rect");
        assert!((shape["x"].as_f64().unwrap() - 10.0).abs() < 1e-9);
        assert!((shape["y"].as_f64().unwrap() - 20.0).abs() < 1e-9);
        assert!((shape["width"].as_f64().unwrap() - 80.0).abs() < 1e-9);
        assert!((shape["height"].as_f64().unwrap() - 60.0).abs() < 1e-9);

        // Region uses canonical "label" key
        assert_eq!(regions[0]["region_attributes"]["label"], "cat");

        // b.jpg has 0 annotations (unannotated images preserved)
        let b_entry = &project["b.jpg0"];
        let b_regions = b_entry["regions"].as_array().unwrap();
        assert!(b_regions.is_empty());
    }

    #[test]
    fn writer_deterministic_order() {
        let dataset = Dataset {
            images: vec![
                Image::new(2u64, "z.jpg", 100, 100),
                Image::new(1u64, "a.jpg", 100, 100),
            ],
            categories: vec![Category::new(1u64, "obj")],
            annotations: vec![
                Annotation::new(
                    2u64,
                    2u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
                ),
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
                ),
            ],
            ..Default::default()
        };

        let json = to_via_json_string(&dataset).unwrap();
        let project: BTreeMap<String, serde_json::Value> = serde_json::from_str(&json).unwrap();

        let keys: Vec<&String> = project.keys().collect();
        assert_eq!(keys[0], "a.jpg0");
        assert_eq!(keys[1], "z.jpg0");
    }

    #[test]
    fn duplicate_filename_rejected() {
        let json = r#"{
            "a": { "filename": "dup.jpg", "size": 100, "regions": [], "file_attributes": {} },
            "b": { "filename": "dup.jpg", "size": 200, "regions": [], "file_attributes": {} }
        }"#;

        let project: ViaProject = serde_json::from_str(json).unwrap();
        let result = via_project_to_ir(project, Path::new("."), Path::new("test.json"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("duplicate filename"));
    }

    #[test]
    fn empty_filename_rejected() {
        let json = r#"{
            "a": { "filename": "", "size": 100, "regions": [], "file_attributes": {} }
        }"#;

        let project: ViaProject = serde_json::from_str(json).unwrap();
        let result = via_project_to_ir(project, Path::new("."), Path::new("test.json"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("empty 'filename' field"));
    }

    #[test]
    fn path_traversal_rejected() {
        let json = r#"{
            "a": { "filename": "../../../etc/passwd", "size": 100, "regions": [], "file_attributes": {} }
        }"#;

        let project: ViaProject = serde_json::from_str(json).unwrap();
        let result = via_project_to_ir(project, Path::new("."), Path::new("test.json"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("relative path without '..'"));
    }

    #[test]
    fn file_attributes_roundtrip() {
        let mut img = Image::new(1u64, "test.jpg", 100, 100);
        img.attributes
            .insert("via_size_bytes".to_string(), "9999".to_string());
        img.attributes
            .insert("via_file_attr_source".to_string(), "web".to_string());

        let dataset = Dataset {
            images: vec![img],
            categories: vec![],
            annotations: vec![],
            ..Default::default()
        };

        let json = to_via_json_string(&dataset).unwrap();
        let project: BTreeMap<String, serde_json::Value> = serde_json::from_str(&json).unwrap();

        let entry = &project["test.jpg9999"];
        assert_eq!(entry["size"], 9999);
        assert_eq!(entry["file_attributes"]["source"], "web");
    }

    #[test]
    fn region_attributes_roundtrip() {
        let ann = Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
        )
        .with_attribute("via_region_attr_difficult", "true");

        let dataset = Dataset {
            images: vec![Image::new(1u64, "test.jpg", 100, 100)],
            categories: vec![Category::new(1u64, "obj")],
            annotations: vec![ann],
            ..Default::default()
        };

        let json = to_via_json_string(&dataset).unwrap();
        let project: BTreeMap<String, serde_json::Value> = serde_json::from_str(&json).unwrap();

        let regions = project["test.jpg0"]["regions"].as_array().unwrap();
        assert_eq!(regions[0]["region_attributes"]["difficult"], "true");
        assert_eq!(regions[0]["region_attributes"]["label"], "obj");
    }

    #[test]
    fn regions_default_to_empty_array() {
        // Missing "regions" key should default to empty array
        let json = r#"{
            "a": { "filename": "test.jpg", "size": 100, "file_attributes": {} }
        }"#;
        let project: ViaProject = serde_json::from_str(json).expect("parse failed");
        let entry = &project.0["a"];
        match &entry.regions {
            ViaRegions::Array(v) => assert!(v.is_empty()),
            ViaRegions::Map(_) => panic!("expected default empty array"),
        }
    }

    #[test]
    fn non_rect_shapes_skipped() {
        // Polygon shape should be skipped (warning emitted to stderr)
        let json = r#"{
            "a": {
                "filename": "test.jpg",
                "size": 100,
                "regions": [
                    {
                        "shape_attributes": { "name": "polygon", "all_points_x": [1,2,3], "all_points_y": [4,5,6] },
                        "region_attributes": { "label": "cat" }
                    }
                ],
                "file_attributes": {}
            }
        }"#;

        let project: ViaProject = serde_json::from_str(json).expect("parse failed");
        let entry = project.0.values().next().unwrap();
        let rects = collect_rect_regions_with_attrs(entry, Path::new("test.json"));
        assert!(rects.is_empty());
    }

    #[test]
    fn label_from_class_key() {
        let json = r#"{
            "shape_attributes": { "name": "rect", "x": 0, "y": 0, "width": 10, "height": 10 },
            "region_attributes": { "class": "dog" }
        }"#;
        let region: ViaRegion = serde_json::from_str(json).unwrap();
        assert_eq!(
            resolve_label(&region.region_attributes),
            Some("dog".to_string())
        );
    }

    #[test]
    fn label_from_single_scalar_fallback() {
        let json = r#"{
            "shape_attributes": { "name": "rect", "x": 0, "y": 0, "width": 10, "height": 10 },
            "region_attributes": { "type": "bird" }
        }"#;
        let region: ViaRegion = serde_json::from_str(json).unwrap();
        assert_eq!(
            resolve_label(&region.region_attributes),
            Some("bird".to_string())
        );
    }

    #[test]
    fn number_label_resolved() {
        let mut attrs = BTreeMap::new();
        attrs.insert("label".to_string(), serde_json::json!(42));
        assert_eq!(resolve_label(&attrs), Some("42".to_string()));
    }
}
