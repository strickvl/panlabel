//! Microsoft VoTT JSON reader and writer.
//!
//! VoTT JSON exports appear in two common legacy shapes:
//! - an aggregate project JSON with a top-level `assets` object/map
//! - per-asset JSON files with one `{ asset, regions }` object per image
//!
//! The reader accepts either shape as a file, plus a directory layout containing
//! `vott-json-export/panlabel-export.json`, `panlabel-export.json`, or top-level
//! per-asset JSON files. Rectangle regions use `boundingBox` directly. Polygon-
//! like regions are converted to an axis-aligned bbox envelope from `points`.
//! Regions with multiple tags become one IR annotation per tag.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::io_adapter_common::{
    basename_from_uri_or_path, has_json_extension, is_safe_relative_image_ref, write_images_readme,
};
use super::model::{Annotation, Category, Dataset, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

const EXPORT_DIR_NAME: &str = "vott-json-export";
const EXPORT_FILE_NAME: &str = "panlabel-export.json";
const IMAGES_README: &str = "Panlabel wrote VoTT JSON annotations only. Copy your image files here if a downstream tool expects a self-contained VoTT export directory.\n";
const VOTT_VERSION: &str = "2.2.0";

#[derive(Debug, Deserialize)]
struct VottProjectFile {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tags: Vec<VottTag>,
    assets: VottAssets,
    #[serde(default)]
    version: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum VottAssets {
    Map(BTreeMap<String, VottAssetEntry>),
    Array(Vec<VottAssetEntry>),
}

#[derive(Debug, Clone, Deserialize)]
struct VottAssetEntry {
    asset: VottAsset,
    #[serde(default)]
    regions: Vec<VottRegion>,
    #[serde(default)]
    version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct VottAsset {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    size: Option<VottSize>,
    #[serde(default)]
    format: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct VottSize {
    width: f64,
    height: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct VottTag {
    name: String,
    #[serde(default)]
    color: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct VottRegion {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    region_type: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, rename = "boundingBox")]
    bounding_box: Option<VottBoundingBox>,
    #[serde(default)]
    points: Vec<VottPoint>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct VottBoundingBox {
    left: f64,
    top: f64,
    width: f64,
    height: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct VottPoint {
    x: f64,
    y: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VottProjectOut {
    name: String,
    version: String,
    tags: Vec<VottTagOut>,
    assets: BTreeMap<String, VottAssetEntryOut>,
}

#[derive(Debug, Serialize)]
struct VottTagOut {
    name: String,
    color: String,
}

#[derive(Debug, Serialize)]
struct VottAssetEntryOut {
    asset: VottAssetOut,
    regions: Vec<VottRegionOut>,
    version: String,
}

#[derive(Debug, Serialize)]
struct VottAssetOut {
    format: String,
    id: String,
    name: String,
    path: String,
    size: VottSizeOut,
    state: u8,
    #[serde(rename = "type")]
    kind: u8,
}

#[derive(Debug, Serialize)]
struct VottSizeOut {
    width: u32,
    height: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VottRegionOut {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    tags: Vec<String>,
    bounding_box: VottBoundingBox,
    points: Vec<VottPoint>,
}

pub fn read_vott_json(path: &Path) -> Result<Dataset, PanlabelError> {
    if path.is_file() {
        read_vott_json_file(path)
    } else if path.is_dir() {
        read_vott_json_dir(path)
    } else {
        Err(PanlabelError::VottJsonInvalid {
            path: path.to_path_buf(),
            message: "path must be a VoTT JSON file or directory".to_string(),
        })
    }
}

pub fn write_vott_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let output_path = output_json_path(path);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }

    if !is_json_file_path(path) {
        write_images_readme(
            output_path.parent().unwrap_or_else(|| Path::new(".")),
            IMAGES_README,
        )?;
    }

    let file = File::create(&output_path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);
    let project = ir_to_vott_project(dataset);
    serde_json::to_writer_pretty(writer, &project).map_err(|source| PanlabelError::VottJsonWrite {
        path: output_path,
        source,
    })
}

pub fn from_vott_json_str_with_base_dir(
    json: &str,
    base_dir: &Path,
) -> Result<Dataset, PanlabelError> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|source| PanlabelError::VottJsonParse {
            path: base_dir.join("<string>"),
            source,
        })?;
    value_to_ir(value, base_dir, &base_dir.join("<string>"))
}

pub fn to_vott_json_string(dataset: &Dataset) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&ir_to_vott_project(dataset))
}

#[cfg(feature = "fuzzing")]
pub fn parse_vott_json_slice(bytes: &[u8]) -> Result<(), serde_json::Error> {
    let _value: serde_json::Value = serde_json::from_slice(bytes)?;
    Ok(())
}

fn read_vott_json_file(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let value: serde_json::Value =
        serde_json::from_reader(reader).map_err(|source| PanlabelError::VottJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    value_to_ir(value, base_dir, path)
}

fn read_vott_json_dir(path: &Path) -> Result<Dataset, PanlabelError> {
    let canonical_export = path.join(EXPORT_DIR_NAME).join(EXPORT_FILE_NAME);
    if canonical_export.is_file() {
        return read_vott_json_file(&canonical_export);
    }

    let root_export = path.join(EXPORT_FILE_NAME);
    if root_export.is_file() {
        return read_vott_json_file(&root_export);
    }

    let mut asset_files = Vec::new();
    for entry in fs::read_dir(path).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        let entry_path = entry.path();
        if is_json_file_path(&entry_path) {
            asset_files.push(entry_path);
        }
    }
    asset_files.sort();

    if asset_files.is_empty() {
        return Err(PanlabelError::VottJsonInvalid {
            path: path.to_path_buf(),
            message: format!(
                "directory does not contain {EXPORT_DIR_NAME}/{EXPORT_FILE_NAME}, {EXPORT_FILE_NAME}, or top-level per-asset .json files"
            ),
        });
    }

    let mut entries = Vec::new();
    for asset_path in &asset_files {
        let file = File::open(asset_path).map_err(PanlabelError::Io)?;
        let reader = BufReader::new(file);
        let entry: VottAssetEntry =
            serde_json::from_reader(reader).map_err(|source| PanlabelError::VottJsonParse {
                path: asset_path.clone(),
                source,
            })?;
        entries.push(entry);
    }

    vott_entries_to_ir(entries, Vec::new(), path, path)
}

fn value_to_ir(
    value: serde_json::Value,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    if value.get("assets").is_some() {
        let project: VottProjectFile =
            serde_json::from_value(value).map_err(|source| PanlabelError::VottJsonParse {
                path: source_path.to_path_buf(),
                source,
            })?;
        let entries = match project.assets {
            VottAssets::Map(map) => map.into_values().collect(),
            VottAssets::Array(entries) => entries,
        };
        let _project_name = project.name;
        let _project_version = project.version;
        vott_entries_to_ir(entries, project.tags, base_dir, source_path)
    } else if value.get("asset").is_some() && value.get("regions").is_some() {
        let entry: VottAssetEntry =
            serde_json::from_value(value).map_err(|source| PanlabelError::VottJsonParse {
                path: source_path.to_path_buf(),
                source,
            })?;
        vott_entries_to_ir(vec![entry], Vec::new(), base_dir, source_path)
    } else {
        Err(PanlabelError::VottJsonInvalid {
            path: source_path.to_path_buf(),
            message: "expected aggregate VoTT JSON with top-level 'assets' or per-asset JSON with 'asset' and 'regions'".to_string(),
        })
    }
}

fn vott_entries_to_ir(
    entries: Vec<VottAssetEntry>,
    project_tags: Vec<VottTag>,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    if entries.is_empty() {
        return Err(PanlabelError::VottJsonInvalid {
            path: source_path.to_path_buf(),
            message: "VoTT JSON contains no assets".to_string(),
        });
    }

    let mut normalized_entries = Vec::new();
    let mut seen_filenames = BTreeSet::new();
    for entry in entries {
        let file_name = asset_file_name(&entry.asset, source_path)?;
        if !seen_filenames.insert(file_name.clone()) {
            return Err(PanlabelError::VottJsonInvalid {
                path: source_path.to_path_buf(),
                message: format!("duplicate asset filename '{file_name}'"),
            });
        }
        normalized_entries.push((file_name, entry));
    }
    normalized_entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut category_names = Vec::new();
    let mut seen_labels = BTreeSet::new();
    for tag in project_tags {
        if !tag.name.trim().is_empty() && seen_labels.insert(tag.name.clone()) {
            category_names.push(tag.name);
        }
        let _tag_color = tag.color;
    }

    let mut extra_labels = BTreeSet::new();
    for (_file_name, entry) in &normalized_entries {
        for region in &entry.regions {
            for tag in &region.tags {
                if tag.trim().is_empty() {
                    continue;
                }
                if !seen_labels.contains(tag) {
                    extra_labels.insert(tag.clone());
                }
            }
        }
    }
    for label in extra_labels {
        seen_labels.insert(label.clone());
        category_names.push(label);
    }

    let label_to_cat: BTreeMap<String, CategoryId> = category_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), CategoryId::new((idx + 1) as u64)))
        .collect();
    let categories: Vec<Category> = category_names
        .iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name.clone()))
        .collect();

    let mut images = Vec::new();
    let mut annotations = Vec::new();
    let mut ann_id = 1u64;

    for (image_idx, (file_name, entry)) in normalized_entries.into_iter().enumerate() {
        let (width, height) =
            resolve_image_dimensions(base_dir, &file_name, &entry.asset, source_path)?;
        let image_id = ImageId::new((image_idx + 1) as u64);
        let mut image = Image::new(image_id, file_name.clone(), width, height);
        if let Some(asset_id) = entry.asset.id.as_ref().filter(|value| !value.is_empty()) {
            image
                .attributes
                .insert("vott_asset_id".to_string(), asset_id.clone());
        }
        if let Some(asset_path) = entry.asset.path.as_ref().filter(|value| !value.is_empty()) {
            image
                .attributes
                .insert("vott_asset_path".to_string(), asset_path.clone());
        }
        if let Some(format) = entry
            .asset
            .format
            .as_ref()
            .filter(|value| !value.is_empty())
        {
            image
                .attributes
                .insert("vott_asset_format".to_string(), format.clone());
        }
        if let Some(version) = entry.version.as_ref().filter(|value| !value.is_empty()) {
            image
                .attributes
                .insert("vott_asset_version".to_string(), version.clone());
        }
        images.push(image);

        for region in &entry.regions {
            if region.tags.is_empty() {
                continue;
            }
            let bbox = region_to_bbox(region, source_path, &file_name)?;
            for tag in &region.tags {
                if tag.trim().is_empty() {
                    continue;
                }
                let Some(category_id) = label_to_cat.get(tag) else {
                    continue;
                };
                let mut annotation =
                    Annotation::new(AnnotationId::new(ann_id), image_id, *category_id, bbox);
                if let Some(region_id) = region.id.as_ref().filter(|value| !value.is_empty()) {
                    annotation
                        .attributes
                        .insert("vott_region_id".to_string(), region_id.clone());
                }
                if let Some(region_type) = region
                    .region_type
                    .as_ref()
                    .filter(|value| !value.is_empty())
                {
                    annotation
                        .attributes
                        .insert("vott_region_type".to_string(), region_type.clone());
                }
                if region.bounding_box.is_none() && !region.points.is_empty() {
                    annotation
                        .attributes
                        .insert("vott_geometry_enveloped".to_string(), "true".to_string());
                }
                annotations.push(annotation);
                ann_id += 1;
            }
        }
    }

    Ok(Dataset {
        images,
        categories,
        annotations,
        ..Default::default()
    })
}

fn region_to_bbox(
    region: &VottRegion,
    source_path: &Path,
    image_name: &str,
) -> Result<BBoxXYXY<Pixel>, PanlabelError> {
    if let Some(bbox) = region.bounding_box {
        return Ok(BBoxXYXY::<Pixel>::from_xywh(
            bbox.left,
            bbox.top,
            bbox.width,
            bbox.height,
        ));
    }

    if !region.points.is_empty() {
        let mut xmin = f64::INFINITY;
        let mut ymin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        for point in &region.points {
            xmin = xmin.min(point.x);
            ymin = ymin.min(point.y);
            xmax = xmax.max(point.x);
            ymax = ymax.max(point.y);
        }
        return Ok(BBoxXYXY::<Pixel>::from_xyxy(xmin, ymin, xmax, ymax));
    }

    Err(PanlabelError::VottJsonInvalid {
        path: source_path.to_path_buf(),
        message: format!(
            "region in image '{image_name}' has tags but no boundingBox or points geometry"
        ),
    })
}

fn ir_to_vott_project(dataset: &Dataset) -> VottProjectOut {
    let category_lookup: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|cat| (cat.id, cat)).collect();
    let mut sorted_categories: Vec<&Category> = dataset.categories.iter().collect();
    sorted_categories.sort_by_key(|category| category.id);
    let tags = sorted_categories
        .iter()
        .enumerate()
        .map(|(idx, category)| VottTagOut {
            name: category.name.clone(),
            color: palette_color(idx),
        })
        .collect();

    let mut anns_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    let mut sorted_images: Vec<&Image> = dataset.images.iter().collect();
    sorted_images.sort_by(|a, b| a.file_name.cmp(&b.file_name).then_with(|| a.id.cmp(&b.id)));

    let mut assets = BTreeMap::new();
    for image in sorted_images {
        let asset_id = format!("panlabel-asset-{}", image.id.as_u64());
        let mut anns = anns_by_image.get(&image.id).cloned().unwrap_or_default();
        anns.sort_by_key(|ann| ann.id);

        let mut regions = Vec::new();
        for ann in anns {
            let Some(category) = category_lookup.get(&ann.category_id) else {
                continue;
            };
            let (left, top, width, height) = ann.bbox.to_xywh();
            let bounding_box = VottBoundingBox {
                left,
                top,
                width,
                height,
            };
            let points = vec![
                VottPoint { x: left, y: top },
                VottPoint {
                    x: left + width,
                    y: top,
                },
                VottPoint {
                    x: left + width,
                    y: top + height,
                },
                VottPoint {
                    x: left,
                    y: top + height,
                },
            ];
            regions.push(VottRegionOut {
                id: format!("panlabel-region-{}", ann.id.as_u64()),
                kind: "RECTANGLE".to_string(),
                tags: vec![category.name.clone()],
                bounding_box,
                points,
            });
        }

        assets.insert(
            asset_id.clone(),
            VottAssetEntryOut {
                asset: VottAssetOut {
                    format: image_format(&image.file_name),
                    id: asset_id,
                    name: image.file_name.clone(),
                    path: format!("file:images/{}", image.file_name.replace('\\', "/")),
                    size: VottSizeOut {
                        width: image.width,
                        height: image.height,
                    },
                    state: 2,
                    kind: 1,
                },
                regions,
                version: VOTT_VERSION.to_string(),
            },
        );
    }

    VottProjectOut {
        name: dataset
            .info
            .name
            .clone()
            .unwrap_or_else(|| "panlabel-export".to_string()),
        version: VOTT_VERSION.to_string(),
        tags,
        assets,
    }
}

fn asset_file_name(asset: &VottAsset, source_path: &Path) -> Result<String, PanlabelError> {
    let candidate = asset
        .name
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(ToString::to_string)
        .or_else(|| asset.path.as_deref().and_then(basename_from_uri_or_path))
        .ok_or_else(|| PanlabelError::VottJsonInvalid {
            path: source_path.to_path_buf(),
            message: "asset is missing both non-empty name and path".to_string(),
        })?;
    let normalized = candidate.replace('\\', "/");
    validate_relative_image_ref(&normalized, source_path)?;
    Ok(normalized)
}

fn validate_relative_image_ref(image_ref: &str, source_path: &Path) -> Result<(), PanlabelError> {
    if !is_safe_relative_image_ref(image_ref) {
        return Err(PanlabelError::VottJsonInvalid {
            path: source_path.to_path_buf(),
            message: format!(
                "image reference '{}' must be a relative path without parent-directory components",
                image_ref
            ),
        });
    }
    Ok(())
}

fn resolve_image_dimensions(
    base_dir: &Path,
    file_name: &str,
    asset: &VottAsset,
    source_path: &Path,
) -> Result<(u32, u32), PanlabelError> {
    if let Some(size) = asset.size {
        if size.width.is_finite()
            && size.height.is_finite()
            && size.width > 0.0
            && size.height > 0.0
        {
            return Ok((size.width.round() as u32, size.height.round() as u32));
        }
    }

    for candidate in image_dimension_candidates(base_dir, file_name, asset) {
        if let Ok(size) = imagesize::size(&candidate) {
            return Ok((size.width as u32, size.height as u32));
        }
    }

    Err(PanlabelError::VottJsonImageNotFound {
        path: source_path.to_path_buf(),
        image_ref: file_name.to_string(),
    })
}

fn image_dimension_candidates(base_dir: &Path, file_name: &str, asset: &VottAsset) -> Vec<PathBuf> {
    let mut candidates = vec![
        base_dir.join(file_name),
        base_dir.join("images").join(file_name),
    ];
    if let Some(path) = asset.path.as_deref() {
        if let Some(local) = local_file_path_from_asset_path(path) {
            candidates.push(local);
        }
    }
    candidates
}

fn local_file_path_from_asset_path(path: &str) -> Option<PathBuf> {
    let trimmed = path
        .strip_prefix("file://")
        .or_else(|| path.strip_prefix("file:"))?;
    Some(PathBuf::from(trimmed))
}

fn output_json_path(path: &Path) -> PathBuf {
    if is_json_file_path(path) {
        path.to_path_buf()
    } else {
        path.join(EXPORT_DIR_NAME).join(EXPORT_FILE_NAME)
    }
}

fn is_json_file_path(path: &Path) -> bool {
    has_json_extension(path)
}

fn image_format(file_name: &str) -> String {
    Path::new(file_name)
        .extension()
        .and_then(|ext| ext.to_str())
        .filter(|ext| !ext.is_empty())
        .unwrap_or("jpg")
        .to_ascii_lowercase()
}

fn palette_color(idx: usize) -> String {
    const COLORS: [&str; 8] = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    ];
    COLORS[idx % COLORS.len()].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aggregate_assets_shape() {
        let json = r##"{
            "name": "demo",
            "version": "2.2.0",
            "tags": [{"name": "person", "color": "#ff0000"}, {"name": "car"}],
            "assets": {
                "asset-1": {
                    "asset": {"id": "asset-1", "name": "img1.bmp", "path": "file:img1.bmp", "size": {"width": 100, "height": 80}, "format": "bmp"},
                    "regions": [{"id": "r1", "type": "RECTANGLE", "tags": ["person"], "boundingBox": {"left": 10, "top": 20, "width": 40, "height": 50}}]
                }
            }
        }"##;
        let dataset = from_vott_json_str_with_base_dir(json, Path::new(".")).expect("parse");
        assert_eq!(dataset.images.len(), 1);
        assert_eq!(dataset.categories.len(), 2);
        assert_eq!(dataset.annotations.len(), 1);
        assert_eq!(dataset.annotations[0].bbox.xmax(), 50.0);
    }

    #[test]
    fn polygon_points_become_envelope_and_multi_tags_expand() {
        let json = r##"{
            "asset": {"name": "poly.bmp", "size": {"width": 100, "height": 80}},
            "regions": [{"id": "poly-1", "type": "POLYGON", "tags": ["cat", "animal"], "points": [{"x": 5, "y": 10}, {"x": 30, "y": 2}, {"x": 20, "y": 40}]}]
        }"##;
        let dataset = from_vott_json_str_with_base_dir(json, Path::new(".")).expect("parse");
        assert_eq!(dataset.annotations.len(), 2);
        assert_eq!(dataset.annotations[0].bbox.xmin(), 5.0);
        assert_eq!(dataset.annotations[0].bbox.ymin(), 2.0);
        assert_eq!(dataset.annotations[0].bbox.xmax(), 30.0);
        assert_eq!(dataset.annotations[0].bbox.ymax(), 40.0);
        assert_eq!(
            dataset.annotations[0].attributes["vott_geometry_enveloped"],
            "true"
        );
    }

    #[test]
    fn writer_emits_aggregate_assets() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "img.bmp", 100, 80)],
            categories: vec![Category::new(1u64, "person")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 50.0, 70.0),
            )],
            ..Default::default()
        };
        let json = to_vott_json_string(&dataset).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse output");
        let assets = value["assets"].as_object().expect("assets object");
        assert_eq!(assets.len(), 1);
        let entry = assets.values().next().unwrap();
        assert_eq!(entry["asset"]["name"], "img.bmp");
        assert_eq!(entry["regions"][0]["type"], "RECTANGLE");
        assert_eq!(entry["regions"][0]["tags"][0], "person");
        assert_eq!(entry["regions"][0]["boundingBox"]["width"], 40.0);
    }
}
