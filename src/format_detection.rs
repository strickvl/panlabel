//! Internal format auto-detection heuristics.
//!
//! This module keeps filesystem/content sniffing separate from CLI orchestration.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::{ir, ConvertFormat, PanlabelError};

/// Detect the format of an input path based on extension/content (files)
/// or structure (directories).
pub(crate) fn detect_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    if path.is_dir() {
        return detect_dir_format(path);
    }

    // Catch missing files early with a path-contextual message, instead of
    // letting downstream File::open produce a bare "IO error: No such file".
    if !path.exists() {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "file does not exist".to_string(),
        });
    }

    // First try extension-based detection
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "csv" => return detect_csv_format(path),
            "tfrecord" | "tfrecords" => return detect_tfrecord_format(path),
            "json" => return detect_json_format(path),
            "jsonl" | "ndjson" | "manifest" => return detect_jsonl_format(path),
            "xml" => return detect_xml_format(path),
            "txt" => return detect_txt_format(path),
            _ => {}
        }
    }

    // Keep message stable (existing CLI tests assert this substring).
    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "unrecognized file extension (expected .json, .jsonl, .ndjson, .manifest, .csv, .xml, .txt, or .tfrecord). Use --from to specify format explicitly.".to_string(),
    })
}

/// Evidence collected while probing a directory for a specific format.
struct FormatProbe {
    name: &'static str,
    format: ConvertFormat,
    found: Vec<String>,
    missing: Vec<String>,
}

impl FormatProbe {
    fn new(name: &'static str, format: ConvertFormat) -> Self {
        Self {
            name,
            format,
            found: Vec::new(),
            missing: Vec::new(),
        }
    }

    fn is_detected(&self) -> bool {
        !self.found.is_empty() && self.missing.is_empty()
    }

    fn is_partial(&self) -> bool {
        !self.found.is_empty() && !self.missing.is_empty()
    }
}

fn detect_dir_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let probes = probe_dir_formats(path)?;

    let detected: Vec<&FormatProbe> = probes.iter().filter(|p| p.is_detected()).collect();
    let partial: Vec<&FormatProbe> = probes.iter().filter(|p| p.is_partial()).collect();

    if detected.len() == 1 {
        return Ok(detected[0].format);
    }

    if detected.len() > 1 {
        let names: Vec<&str> = detected.iter().map(|p| p.name).collect();
        let header = if detected.len() == 2 {
            format!(
                "directory matches both {} and {} layouts",
                names[0], names[1]
            )
        } else {
            format!("directory matches multiple layouts ({})", names.join(", "))
        };

        let mut reason = format!("{}:\n", header);
        for p in &detected {
            reason.push_str(&format!("  - {}: found {}\n", p.name, p.found.join(", ")));
        }
        reason.push_str("Use --from to specify format explicitly.");

        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason,
        });
    }

    // No complete match — check for partial matches (e.g. labels/ without images/).
    if !partial.is_empty() {
        let mut reason = String::new();
        for p in &partial {
            reason.push_str(&format!(
                "found {}-style markers ({}), but missing: {}\n",
                p.name,
                p.found.join(", "),
                p.missing.join(", "),
            ));
        }
        reason.push_str("Use --from to specify format explicitly, or fix the directory layout.");

        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason,
        });
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "unrecognized directory layout. Expected one of:\n  \
                 - YOLO: labels/ with .txt files and sibling images/\n  \
                 - YOLO Keras / YOLOv4 PyTorch TXT: yolo_keras.txt, yolov4_pytorch.txt, annotations.txt, or train.txt\n  \
                 - OIDv4: Label/ directories with .txt labels\n  \
                 - Edge Impulse: bounding_boxes.labels at directory root\n  \
                 - VOC: Annotations/ with .xml files\n  \
                 - CVAT: annotations.xml at directory root\n  \
                 - IBM Cloud Annotations: _annotations.json at directory root\n  \
                 - VoTT JSON: vott-json-export/panlabel-export.json at directory root\n  \
                 - Scale AI: annotations/ with Scale AI .json files, or co-located .json files\n  \
                 - Unity Perception: SOLO frame/captures .json files\n  \
                 - HF: metadata.jsonl, metadata.parquet, or parquet shard files\n  \
                 - LabelMe: annotations/ with LabelMe .json files, or co-located .json files\n  \
                 - SuperAnnotate: annotations/ with SuperAnnotate .json files, or co-located .json files\n  \
                 - Cityscapes: gtFine/<split>/<city>/*_gtFine_polygons.json files\n  \
                 - Marmot: .xml files with Page@CropBox plus same-stem companion images\n  \
                 - Supervisely: ann/ with .json files, or project meta.json with dataset ann/ directories\n  \
                 - KITTI: label_2/ with .txt files and sibling image_2/\n\
                 Use --from to specify format explicitly."
            .to_string(),
    })
}

/// Probe a directory for all supported format markers. Returns one probe
/// per format, each with the markers it found and what's missing (if any).
fn probe_dir_formats(path: &Path) -> Result<Vec<FormatProbe>, PanlabelError> {
    let mut probes = Vec::with_capacity(20);

    // --- YOLO ---
    // Aligned with io_yolo::discover_layout/discover_source: requires labels/ with
    // .txt AND images/ for flat layout, OR data.yaml with split keys for split-aware.
    let mut yolo = FormatProbe::new("YOLO", ConvertFormat::Yolo);
    let (labels_dir_exists, has_txt) = if path.join("labels").is_dir() {
        (true, dir_contains_txt_files(&path.join("labels"))?)
    } else if is_labels_dir(path) {
        (true, dir_contains_txt_files(path)?)
    } else {
        (false, false)
    };
    if labels_dir_exists && has_txt {
        yolo.found.push("labels/ with .txt files".into());
        // Check for images/ sibling — aligned with reader requirement.
        let images_exists = if is_labels_dir(path) {
            path.parent()
                .map(|p| p.join("images").is_dir())
                .unwrap_or(false)
        } else {
            path.join("images").is_dir()
        };
        if images_exists {
            yolo.found.push("images/ directory".into());
        } else {
            yolo.missing.push("images/ directory".into());
        }
    }
    // Also detect split-aware YOLO via data.yaml with train/val/test keys.
    if yolo.found.is_empty() {
        if let Some(split_keys) = data_yaml_has_split_keys(path) {
            yolo.found.push(format!(
                "data.yaml with split keys: {}",
                split_keys.join(", ")
            ));
        }
    }
    let yolo_complete = yolo.is_detected();
    probes.push(yolo);

    // --- YOLO Keras / YOLOv4 PyTorch absolute-coordinate TXT ---
    probes.push(probe_yolo_keras_txt_dir(
        path,
        "YOLO Keras TXT",
        ConvertFormat::YoloKeras,
        &ir::io_yolo_keras_txt::YOLO_KERAS_ANNOTATION_CANDIDATES,
        !yolo_complete,
    )?);
    probes.push(probe_yolo_keras_txt_dir(
        path,
        "YOLOv4 PyTorch TXT",
        ConvertFormat::YoloV4Pytorch,
        &ir::io_yolo_keras_txt::YOLOV4_PYTORCH_ANNOTATION_CANDIDATES,
        !yolo_complete,
    )?);

    // --- VOC ---
    // Aligned with io_voc_xml::discover_layout: requires Annotations/ with
    // top-level .xml files, but JPEGImages/ is optional.
    let mut voc = FormatProbe::new("VOC", ConvertFormat::Voc);
    let (ann_dir, has_top_level_xml) = if path.join("Annotations").is_dir() {
        let ann = path.join("Annotations");
        (true, dir_contains_top_level_xml_files(&ann)?)
    } else if is_annotations_dir(path) {
        (true, dir_contains_top_level_xml_files(path)?)
    } else {
        (false, false)
    };
    if ann_dir && has_top_level_xml {
        voc.found
            .push("Annotations/ with top-level .xml files".into());
    }
    probes.push(voc);

    // --- CVAT ---
    let mut cvat = FormatProbe::new("CVAT", ConvertFormat::Cvat);
    if path.join("annotations.xml").is_file() {
        cvat.found.push("annotations.xml at root".into());
    }
    probes.push(cvat);

    // --- IBM Cloud Annotations ---
    let mut cloud_annotations =
        FormatProbe::new("IBM Cloud Annotations", ConvertFormat::IbmCloudAnnotations);
    let cloud_annotations_path = path.join("_annotations.json");
    if cloud_annotations_path.is_file() {
        if let Ok(contents) = std::fs::read_to_string(&cloud_annotations_path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                if is_likely_cloud_annotations_file(&value) {
                    cloud_annotations
                        .found
                        .push("_annotations.json localization file".into());
                }
            }
        }
    }
    probes.push(cloud_annotations);

    // --- VoTT JSON ---
    let mut vott_json = FormatProbe::new("VoTT JSON", ConvertFormat::VottJson);
    let vott_export_path = path.join("vott-json-export").join("panlabel-export.json");
    let root_vott_export_path = path.join("panlabel-export.json");
    for candidate in [&vott_export_path, &root_vott_export_path] {
        if candidate.is_file() {
            if let Ok(contents) = std::fs::read_to_string(candidate) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                    if is_likely_vott_json_file(&value) {
                        vott_json.found.push(format!(
                            "{} VoTT JSON export",
                            candidate.strip_prefix(path).unwrap_or(candidate).display()
                        ));
                        break;
                    }
                }
            }
        }
    }
    probes.push(vott_json);

    // --- Scale AI ---
    let mut scale_ai = FormatProbe::new("Scale AI", ConvertFormat::ScaleAi);
    let scale_ann_dir = path.join("annotations");
    if scale_ann_dir.is_dir() && dir_contains_scale_ai_json(&scale_ann_dir)? {
        scale_ai
            .found
            .push("annotations/ with Scale AI .json files".into());
    } else if dir_contains_top_level_scale_ai_json(path)? {
        scale_ai
            .found
            .push("co-located Scale AI .json files".into());
    }
    probes.push(scale_ai);

    // --- Unity Perception ---
    let mut unity = FormatProbe::new("Unity Perception", ConvertFormat::UnityPerception);
    if dir_contains_unity_perception_json(path)? {
        unity.found.push("SOLO frame/captures .json files".into());
    }
    probes.push(unity);

    // --- HF ---
    let mut hf = FormatProbe::new("HF", ConvertFormat::HfImagefolder);
    if dir_contains_hf_metadata(path)? {
        hf.found.push("metadata.jsonl or metadata.parquet".into());
    } else if dir_has_parquet_shards(path)? {
        hf.found.push("parquet shard files".into());
    }
    probes.push(hf);

    // --- KITTI ---
    let mut kitti = FormatProbe::new("KITTI", ConvertFormat::Kitti);
    let kitti_labels_dir = if path.join("label_2").is_dir() {
        Some(path.join("label_2"))
    } else if is_dir_named_ci(path, "label_2") {
        Some(path.to_path_buf())
    } else {
        None
    };
    if let Some(ref labels_dir) = kitti_labels_dir {
        if dir_contains_top_level_txt_files(labels_dir)? {
            kitti.found.push("label_2/ with .txt files".into());
            let images_exists = if is_dir_named_ci(path, "label_2") {
                path.parent()
                    .map(|p| p.join("image_2").is_dir())
                    .unwrap_or(false)
            } else {
                path.join("image_2").is_dir()
            };
            if images_exists {
                kitti.found.push("image_2/ directory".into());
            } else {
                kitti.missing.push("image_2/ directory".into());
            }
        }
    }
    probes.push(kitti);

    // --- LabelMe ---
    let mut labelme = FormatProbe::new("LabelMe", ConvertFormat::LabelMe);
    let labelme_ann_dir = path.join("annotations");
    if labelme_ann_dir.is_dir() && dir_contains_labelme_json(&labelme_ann_dir)? {
        labelme
            .found
            .push("annotations/ with LabelMe .json files".into());
    } else if dir_contains_labelme_json(path)? {
        labelme.found.push("co-located LabelMe .json files".into());
    }
    probes.push(labelme);

    // --- SuperAnnotate ---
    let mut superannotate = FormatProbe::new("SuperAnnotate", ConvertFormat::SuperAnnotate);
    let superannotate_ann_dir = path.join("annotations");
    if superannotate_ann_dir.is_dir() {
        superannotate.found.push("annotations/ directory".into());
        if dir_contains_superannotate_json(&superannotate_ann_dir)? {
            superannotate.found.push("SuperAnnotate .json files".into());
        } else {
            superannotate
                .missing
                .push("SuperAnnotate .json files".into());
        }
    } else if dir_contains_top_level_superannotate_json(path)? {
        superannotate
            .found
            .push("co-located SuperAnnotate .json files".into());
    }
    probes.push(superannotate);

    // --- Cityscapes ---
    let mut cityscapes = FormatProbe::new("Cityscapes", ConvertFormat::Cityscapes);
    let mut edge_impulse = FormatProbe::new("Edge Impulse", ConvertFormat::EdgeImpulse);
    if path.join("bounding_boxes.labels").is_file() {
        edge_impulse
            .found
            .push("bounding_boxes.labels at root".into());
    }
    probes.push(edge_impulse);

    let mut oidv4 = FormatProbe::new("OIDv4", ConvertFormat::Oidv4);
    if ir::io_oidv4_txt::dir_has_oidv4_label_files(path)? {
        oidv4
            .found
            .push("Label/ directories with .txt labels".into());
    }
    probes.push(oidv4);

    if path.join("gtFine").is_dir() {
        if dir_contains_cityscapes_json(&path.join("gtFine"))? {
            cityscapes
                .found
                .push("gtFine/ with Cityscapes polygon JSON files".into());
        } else {
            cityscapes
                .missing
                .push("Cityscapes *_gtFine_polygons.json files".into());
        }
    } else if dir_contains_cityscapes_json(path)? {
        cityscapes
            .found
            .push("Cityscapes polygon JSON files".into());
    }
    probes.push(cityscapes);

    // --- Marmot ---
    let mut marmot = FormatProbe::new("Marmot", ConvertFormat::Marmot);
    let marmot_status = dir_contains_marmot_xml(path)?;
    if marmot_status.found_xml {
        marmot.found.push("Marmot Page XML files".into());
        if marmot_status.missing_companion_images == 0 {
            marmot.found.push("same-stem companion images".into());
        } else {
            marmot.missing.push(format!(
                "same-stem companion image(s) for {} Marmot XML file(s)",
                marmot_status.missing_companion_images
            ));
        }
    }
    probes.push(marmot);

    // --- Supervisely ---
    let mut supervisely = FormatProbe::new("Supervisely", ConvertFormat::Supervisely);
    if path.join("ann").is_dir() {
        supervisely.found.push("ann/ directory".into());
        if dir_contains_supervisely_json(&path.join("ann"))? {
            supervisely.found.push("Supervisely .json files".into());
        } else {
            supervisely.missing.push("Supervisely .json files".into());
        }
    } else if path.join("meta.json").is_file() {
        supervisely.found.push("meta.json".into());
        let mut dataset_ann_dirs = 0usize;
        for entry in
            std::fs::read_dir(path).map_err(|source| PanlabelError::FormatDetectionFailed {
                path: path.to_path_buf(),
                reason: format!("failed while inspecting directory: {source}"),
            })?
        {
            let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
                path: path.to_path_buf(),
                reason: format!("failed while inspecting directory: {source}"),
            })?;
            let ann_dir = entry.path().join("ann");
            if entry.path().is_dir() && ann_dir.is_dir() && dir_contains_supervisely_json(&ann_dir)?
            {
                dataset_ann_dirs += 1;
            }
        }
        if dataset_ann_dirs > 0 {
            supervisely.found.push(format!(
                "meta.json with {dataset_ann_dirs} dataset ann/ director{}",
                if dataset_ann_dirs == 1 { "y" } else { "ies" }
            ));
        } else {
            supervisely
                .missing
                .push("dataset ann/ directories with Supervisely .json files".into());
        }
    }
    probes.push(supervisely);

    Ok(probes)
}

fn probe_yolo_keras_txt_dir(
    path: &Path,
    name: &'static str,
    format: ConvertFormat,
    candidates: &[&str],
    allow_generic_train_txt: bool,
) -> Result<FormatProbe, PanlabelError> {
    let mut probe = FormatProbe::new(name, format);
    for candidate_name in candidates {
        if *candidate_name == "train.txt" && !allow_generic_train_txt {
            continue;
        }
        let candidate = path.join(candidate_name);
        if !candidate.is_file() {
            continue;
        }
        if ir::io_yolo_keras_txt::looks_like_yolo_keras_txt_file(&candidate)? {
            probe.found.push(format!(
                "{} absolute-coordinate annotation file",
                candidate_name
            ));
            break;
        } else if candidate_name.contains("yolo") || *candidate_name == "annotations.txt" {
            probe
                .missing
                .push(format!("valid {} row grammar in {}", name, candidate_name));
        }
    }
    Ok(probe)
}

fn dir_contains_txt_files(path: &Path) -> Result<bool, PanlabelError> {
    dir_contains_extension_files(path, "txt")
}

fn dir_contains_hf_metadata(path: &Path) -> Result<bool, PanlabelError> {
    if path.join("metadata.jsonl").is_file() || path.join("metadata.parquet").is_file() {
        return Ok(true);
    }

    for entry in std::fs::read_dir(path).map_err(|source| PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!("failed while inspecting directory: {source}"),
    })? {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry_path.is_dir()
            && (entry_path.join("metadata.jsonl").is_file()
                || entry_path.join("metadata.parquet").is_file())
        {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Check for HF parquet shard files (e.g. `data/train-00000-of-00001.parquet`).
/// Scans up to depth 2 for .parquet files that aren't `metadata.parquet`.
fn dir_has_parquet_shards(path: &Path) -> Result<bool, PanlabelError> {
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(_) => return Ok(false),
    };
    for entry in entries {
        let entry = entry.map_err(PanlabelError::Io)?;
        let entry_path = entry.path();
        if !entry_path.is_dir() {
            continue;
        }
        let sub_entries = match std::fs::read_dir(&entry_path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for sub_entry in sub_entries {
            let sub_entry = sub_entry.map_err(PanlabelError::Io)?;
            let sub_path = sub_entry.path();
            if sub_path.is_file()
                && sub_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("parquet"))
                    .unwrap_or(false)
                && sub_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| !n.eq_ignore_ascii_case("metadata.parquet"))
                    .unwrap_or(false)
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Check if a directory contains .xml files at the top level only (non-recursive).
/// Aligned with VOC reader's `collect_xml_files()` which uses `fs::read_dir`.
fn dir_contains_top_level_xml_files(path: &Path) -> Result<bool, PanlabelError> {
    dir_contains_top_level_extension_files(path, "xml")
}

fn dir_contains_top_level_extension_files(
    path: &Path,
    extension: &str,
) -> Result<bool, PanlabelError> {
    for entry in std::fs::read_dir(path).map_err(|source| PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!("failed while inspecting directory: {source}"),
    })? {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry_path.is_file()
            && entry_path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case(extension))
                .unwrap_or(false)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn dir_contains_extension_files(path: &Path, extension: &str) -> Result<bool, PanlabelError> {
    for entry in walkdir::WalkDir::new(path).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;

        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case(extension))
                .unwrap_or(false)
        {
            return Ok(true);
        }
    }

    Ok(false)
}

fn is_labels_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("labels"))
        .unwrap_or(false)
}

fn is_annotations_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("annotations"))
        .unwrap_or(false)
}

fn is_dir_named_ci(path: &Path, dir_name: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case(dir_name))
        .unwrap_or(false)
}

fn dir_contains_top_level_txt_files(path: &Path) -> Result<bool, PanlabelError> {
    dir_contains_top_level_extension_files(path, "txt")
}

/// Check if a directory contains at least one LabelMe JSON file.
///
/// Quick structural check: looks for a .json file with a `shapes` array key.
fn dir_contains_labelme_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_top_level_json_matching(dir, is_likely_labelme_file)
}

/// Check if a directory contains at least one SuperAnnotate annotation JSON file.
fn dir_contains_superannotate_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_json_matching(dir, is_likely_superannotate_file)
}

/// Check if a directory contains at least one top-level SuperAnnotate JSON file.
fn dir_contains_top_level_superannotate_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_top_level_json_matching(dir, is_likely_superannotate_file)
}

fn dir_contains_scale_ai_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_json_matching(dir, ir::io_scale_ai_json::is_likely_scale_ai_file)
}

fn dir_contains_top_level_scale_ai_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_top_level_json_matching(dir, ir::io_scale_ai_json::is_likely_scale_ai_file)
}

fn dir_contains_unity_perception_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_json_matching(
        dir,
        ir::io_unity_perception_json::is_likely_unity_perception_file,
    )
}

struct MarmotDirStatus {
    found_xml: bool,
    missing_companion_images: usize,
}

fn dir_contains_marmot_xml(dir: &Path) -> Result<MarmotDirStatus, PanlabelError> {
    let mut found_xml = false;
    let mut missing_companion_images = 0usize;
    for entry in walkdir::WalkDir::new(dir).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: dir.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if !entry.file_type().is_file()
            || !entry_path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("xml"))
                .unwrap_or(false)
        {
            continue;
        }
        if ir::io_marmot_xml::is_likely_marmot_xml_file(entry_path)? {
            found_xml = true;
            if !ir::io_marmot_xml::has_companion_image(entry_path) {
                missing_companion_images += 1;
            }
        }
    }
    Ok(MarmotDirStatus {
        found_xml,
        missing_companion_images,
    })
}

fn dir_contains_cityscapes_json(dir: &Path) -> Result<bool, PanlabelError> {
    for entry in walkdir::WalkDir::new(dir).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: dir.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if !entry.file_type().is_file()
            || !entry_path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("_gtFine_polygons.json"))
                .unwrap_or(false)
        {
            continue;
        }
        if let Ok(contents) = std::fs::read_to_string(entry_path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                if ir::io_cityscapes_json::is_likely_cityscapes_file(&value) {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Check if a directory contains at least one Supervisely annotation JSON file.
fn dir_contains_supervisely_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_json_matching(dir, is_likely_supervisely_file)
}

fn dir_contains_top_level_json_matching(
    dir: &Path,
    predicate: fn(&serde_json::Value) -> bool,
) -> Result<bool, PanlabelError> {
    for entry in std::fs::read_dir(dir).map_err(|source| PanlabelError::FormatDetectionFailed {
        path: dir.to_path_buf(),
        reason: format!("failed while inspecting directory: {source}"),
    })? {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: dir.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry_path.is_file()
            && entry_path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        {
            if let Ok(contents) = std::fs::read_to_string(&entry_path) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                    if predicate(&value) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

fn dir_contains_json_matching(
    dir: &Path,
    predicate: fn(&serde_json::Value) -> bool,
) -> Result<bool, PanlabelError> {
    for entry in walkdir::WalkDir::new(dir).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: dir.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry.file_type().is_file()
            && entry_path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        {
            if let Ok(contents) = std::fs::read_to_string(entry_path) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                    if predicate(&value) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

/// Check if `data.yaml` exists and contains split keys (train/val/test).
/// Returns `Some(vec!["train", ...])` if found, `None` otherwise.
fn data_yaml_has_split_keys(path: &Path) -> Option<Vec<String>> {
    let yaml_path = path.join("data.yaml");
    let content = std::fs::read_to_string(&yaml_path).ok()?;
    let mapping: serde_yaml::Value = serde_yaml::from_str(&content).ok()?;
    let map = mapping.as_mapping()?;
    let mut found = Vec::new();
    for key in ["train", "val", "test"] {
        if map.contains_key(serde_yaml::Value::String(key.to_string())) {
            found.push(key.to_string());
        }
    }
    if found.is_empty() {
        None
    } else {
        Some(found)
    }
}

/// Detect whether a JSON file is Label Studio, COCO, or IR JSON format.
/// Detect CSV sub-format by sniffing column count and header.
///
/// Heuristics:
/// - 8 columns (filename,width,height,class,xmin,ymin,xmax,ymax) -> TFOD
/// - 6 columns (path,x1,y1,x2,y2,class_name or headerless data) -> RetinaNet
fn detect_tfrecord_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    if ir::io_tfrecord::is_supported_tfrecord_file(path)? {
        Ok(ConvertFormat::Tfrecord)
    } else {
        Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "TFRecord framing is valid only for uncompressed TFOD-style tf.train.Example records in v1, or the file is not a TFRecord. Use --from to specify format explicitly."
                .to_string(),
        })
    }
}

fn detect_txt_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    if ir::io_wider_face_txt::looks_like_wider_face_txt_file(path)? {
        return Ok(ConvertFormat::WiderFace);
    }

    let filename_lower = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if (filename_lower.contains("oidv4") || filename_lower.contains("openimages-v4"))
        && ir::io_oidv4_txt::looks_like_oidv4_txt_file(path)?
    {
        return Ok(ConvertFormat::Oidv4);
    }

    let looks_like = ir::io_yolo_keras_txt::looks_like_yolo_keras_txt_file(path)?;
    if !looks_like {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "TXT file does not match WIDER Face, conservative OIDv4 filename hints, or YOLO Keras / YOLOv4 PyTorch absolute-coordinate grammar. Use --from to specify format explicitly.".to_string(),
        });
    }

    let filename = filename_lower;
    let normalized = filename.replace('-', "_");

    if normalized.contains("yolo_keras") || normalized.contains("keras_yolo") {
        return Ok(ConvertFormat::YoloKeras);
    }
    if normalized.contains("yolov4_pytorch") || normalized.contains("pytorch_yolov4") {
        return Ok(ConvertFormat::YoloV4Pytorch);
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!(
            "TXT file matches both yolo-keras and yolov4-pytorch absolute-coordinate layouts, but filename '{}' is generic. Use --from yolo-keras or --from yolov4-pytorch to specify the intended public format.",
            filename
        ),
    })
}

fn detect_csv_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let file = std::fs::File::open(path).map_err(PanlabelError::Io)?;
    let reader = std::io::BufReader::new(file);
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(reader);

    // Read up to 8 records for sniffing
    let mut records: Vec<csv::StringRecord> = Vec::new();
    for result in csv_reader.records().take(8) {
        let record = result.map_err(|_| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "failed to parse CSV row while detecting format".to_string(),
        })?;
        records.push(record);
    }

    if records.is_empty() {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason:
                "CSV file is empty; cannot determine format. Use --from to specify format explicitly."
                    .to_string(),
        });
    }

    let first = &records[0];
    let ncols = first.len();

    // --- Header-based detection ---
    let col0 = first.get(0).unwrap_or("");
    let col1 = first.get(1).unwrap_or("");
    let col3 = first.get(3).unwrap_or("");

    // Kaggle Wheat: 5 columns, header starting with "image_id"
    if ncols == 5 && col0.eq_ignore_ascii_case("image_id") && col3.eq_ignore_ascii_case("bbox") {
        return Ok(ConvertFormat::KaggleWheat);
    }

    // Kaggle Wheat: 5 columns, headerless — col3 looks like bracketed bbox
    if ncols == 5 {
        let looks_like_bbox = col3.trim().starts_with('[') && col3.trim().ends_with(']');
        let col1_is_int = col1.parse::<u32>().is_ok();
        if looks_like_bbox && col1_is_int {
            return Ok(ConvertFormat::KaggleWheat);
        }
    }

    // VoTT CSV: exact 6-column header image,xmin,ymin,xmax,ymax,label.
    if ncols == 6
        && col0.eq_ignore_ascii_case("image")
        && col1.eq_ignore_ascii_case("xmin")
        && first
            .get(2)
            .map(|v| v.eq_ignore_ascii_case("ymin"))
            .unwrap_or(false)
        && col3.eq_ignore_ascii_case("xmax")
        && first
            .get(4)
            .map(|v| v.eq_ignore_ascii_case("ymax"))
            .unwrap_or(false)
        && first
            .get(5)
            .map(|v| v.eq_ignore_ascii_case("label"))
            .unwrap_or(false)
    {
        return Ok(ConvertFormat::VottCsv);
    }

    if ir::io_via_csv::is_via_csv_header(first) {
        return Ok(ConvertFormat::ViaCsv);
    }

    // RetinaNet: 6 columns
    if ncols == 6 {
        return Ok(ConvertFormat::Retinanet);
    }

    // OpenImages: 8 or 13 columns with header starting with "ImageID"
    if (ncols == 8 || ncols == 13) && col0.eq_ignore_ascii_case("ImageID") {
        return Ok(ConvertFormat::OpenImages);
    }

    // AutoML Vision: 9 or 11 columns
    if ncols == 9 || ncols == 11 {
        // Check for ML_USE-like first column or header alias
        let c0_lower = col0.to_ascii_lowercase();
        let is_automl_header = c0_lower == "set" || c0_lower == "ml_use";
        let is_automl_data = matches!(
            c0_lower.as_str(),
            "train" | "validation" | "test" | "unassigned"
        );
        if is_automl_header || is_automl_data {
            return Ok(ConvertFormat::AutoMlVision);
        }
        // Check if cols 5/6 (in 11-col form) are empty placeholders
        if ncols == 11 {
            let col5 = first.get(5).unwrap_or("_");
            let col6 = first.get(6).unwrap_or("_");
            if col5.is_empty() && col6.is_empty() {
                return Ok(ConvertFormat::AutoMlVision);
            }
        }
    }

    // 8-column formats: TFOD vs Udacity vs headerless OpenImages
    if ncols == 8 {
        // Check for OpenImages: column order is ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax
        // where Source is non-numeric and Confidence is a float
        let col1_str = first.get(1).unwrap_or("");
        let col3_str = first.get(3).unwrap_or("");
        let col1_not_numeric = col1_str.parse::<f64>().is_err();
        let col3_is_float = col3_str.parse::<f64>().is_ok();

        // Check for TFOD/Udacity header
        if col0.eq_ignore_ascii_case("filename") {
            // Has header — sniff data rows to distinguish TFOD vs Udacity
            return detect_tfod_vs_udacity(&records[1..]);
        }

        // Headerless 8-column: OpenImages if col1 is non-numeric and col3 looks like confidence
        if col1_not_numeric && col3_is_float && !col0.is_empty() {
            // Further check: are cols 4-7 in [0,1]? OpenImages uses normalized coords
            let all_normalized = (4..8).all(|i| {
                first
                    .get(i)
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|v| (0.0..=1.0).contains(&v))
                    .unwrap_or(false)
            });
            if all_normalized {
                return Ok(ConvertFormat::OpenImages);
            }
        }

        // Headerless 8-column TFOD/Udacity
        return detect_tfod_vs_udacity(&records);
    }

    // 13-column: likely OpenImages extended
    if ncols == 13 {
        return Ok(ConvertFormat::OpenImages);
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!(
            "CSV has {ncols} columns; not recognized as any supported format. Use --from to specify format explicitly."
        ),
    })
}

/// Distinguishes TFOD (normalized) from Udacity (absolute pixel) by inspecting coordinate values.
fn detect_tfod_vs_udacity(
    data_records: &[csv::StringRecord],
) -> Result<ConvertFormat, PanlabelError> {
    // If any sampled bbox coordinate is outside [0,1], it's Udacity (absolute pixels)
    for record in data_records {
        if record.len() < 8 {
            continue;
        }
        for i in 4..8 {
            if let Some(Ok(v)) = record.get(i).map(|s| s.parse::<f64>()) {
                if !(0.0..=1.0).contains(&v) {
                    return Ok(ConvertFormat::Udacity);
                }
            }
        }
    }
    // All in [0,1] or no data rows — default to TFOD
    Ok(ConvertFormat::Tfod)
}

/// Detect whether a JSON Lines file is Labelbox rows or a SageMaker Ground Truth manifest.
///
/// Heuristic: first non-empty line is an object with a string `source-ref`
/// and exactly one object-detection label attribute. The label attribute is
/// dynamic, so we accept either a sibling `<label>-metadata.type` of
/// `groundtruth/object-detection` or the canonical `annotations` +
/// `image_size` label-object shape.
fn detect_jsonl_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let mut first_non_empty = None;
    for line in reader.lines() {
        let line = line.map_err(PanlabelError::Io)?;
        if !line.trim().is_empty() {
            first_non_empty = Some(line);
            break;
        }
    }

    let Some(line) = first_non_empty else {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "JSON Lines file is empty; cannot determine format. Use --from to specify format explicitly."
                .to_string(),
        });
    };

    let value: serde_json::Value =
        serde_json::from_str(&line).map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!(
                "failed to parse first JSON Lines row while detecting format: {source}"
            ),
        })?;

    if ir::io_labelbox_json::is_likely_labelbox_row(&value) {
        Ok(ConvertFormat::Labelbox)
    } else if is_likely_sagemaker_manifest_row(&value) {
        Ok(ConvertFormat::SageMaker)
    } else {
        Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "JSON Lines file not recognized as Labelbox export rows or a SageMaker Ground Truth object-detection manifest. Use --from to specify format explicitly."
                .to_string(),
        })
    }
}

///
/// Heuristics:
/// - Array-root JSON: Label Studio task export
/// - Object-root JSON: inspect `annotations[0].bbox`
///   - array of 4 numbers -> COCO
///   - object min/max or xmin/ymin/xmax/ymax -> IR JSON
fn detect_json_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let value: serde_json::Value = serde_json::from_reader(reader).map_err(|source| {
        PanlabelError::FormatDetectionJsonParse {
            path: path.to_path_buf(),
            source,
        }
    })?;

    if let Some(items) = value.as_array() {
        if items.is_empty() {
            // Empty array is ambiguous between Label Studio and CreateML
            return Err(PanlabelError::FormatDetectionFailed {
                path: path.to_path_buf(),
                reason: "empty JSON array is ambiguous (could be Label Studio or CreateML). \
                         Use --from to specify format explicitly."
                    .to_string(),
            });
        }

        if ir::io_bdd100k_json::is_likely_bdd100k_file(&value) {
            return Ok(ConvertFormat::Bdd100k);
        }

        if ir::io_v7_darwin_json::is_likely_v7_darwin_file(&value) {
            return Ok(ConvertFormat::V7Darwin);
        }

        if ir::io_labelbox_json::is_likely_labelbox_row(&items[0]) {
            return Ok(ConvertFormat::Labelbox);
        }

        if ir::io_scale_ai_json::is_likely_scale_ai_file(&items[0]) {
            return Ok(ConvertFormat::ScaleAi);
        }

        if ir::io_unity_perception_json::is_likely_unity_perception_file(&items[0]) {
            return Ok(ConvertFormat::UnityPerception);
        }

        if is_likely_label_studio_task(&items[0]) {
            return Ok(ConvertFormat::LabelStudio);
        }

        if is_likely_createml_item(&items[0]) {
            return Ok(ConvertFormat::CreateMl);
        }

        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "array-root JSON not recognized (expected Labelbox export-row array, Scale AI task/response array, Unity Perception frame array, Label Studio task array, or CreateML image array). Use --from to specify format explicitly.".to_string(),
        });
    }

    if ir::io_edge_impulse_labels::is_likely_edge_impulse_labels(&value) {
        return Ok(ConvertFormat::EdgeImpulse);
    }

    if ir::io_openlabel_json::is_likely_openlabel_file(&value) {
        return Ok(ConvertFormat::OpenLabel);
    }

    if ir::io_datumaro_json::is_likely_datumaro_file(&value) {
        return Ok(ConvertFormat::Datumaro);
    }

    if ir::io_bdd100k_json::is_likely_bdd100k_file(&value) {
        return Ok(ConvertFormat::Bdd100k);
    }

    if ir::io_v7_darwin_json::is_likely_v7_darwin_file(&value) {
        return Ok(ConvertFormat::V7Darwin);
    }

    // Object-root: check for Labelbox export row before COCO/IR heuristic.
    if ir::io_labelbox_json::is_likely_labelbox_row(&value) {
        return Ok(ConvertFormat::Labelbox);
    }

    // Object-root: check for Scale AI task/response JSON before COCO/IR heuristic.
    if ir::io_scale_ai_json::is_likely_scale_ai_file(&value) {
        return Ok(ConvertFormat::ScaleAi);
    }

    // Object-root: check for Unity Perception/SOLO frame or captures JSON.
    if ir::io_unity_perception_json::is_likely_unity_perception_file(&value) {
        return Ok(ConvertFormat::UnityPerception);
    }

    // Object-root: check for LabelMe (has "shapes" key) before COCO/IR heuristic
    if is_likely_labelme_file(&value) {
        return Ok(ConvertFormat::LabelMe);
    }

    // Object-root: check for IBM Cloud Annotations before COCO/IR heuristic.
    if is_likely_cloud_annotations_file(&value) {
        return Ok(ConvertFormat::IbmCloudAnnotations);
    }

    // Object-root: check for VoTT JSON before COCO/IR heuristic.
    if is_likely_vott_json_file(&value) {
        return Ok(ConvertFormat::VottJson);
    }

    // Object-root: check for new per-image JSON formats before COCO/IR heuristic.
    if is_likely_superannotate_file(&value) {
        return Ok(ConvertFormat::SuperAnnotate);
    }

    if ir::io_cityscapes_json::is_likely_cityscapes_file(&value) {
        return Ok(ConvertFormat::Cityscapes);
    }

    if is_likely_supervisely_file(&value) {
        return Ok(ConvertFormat::Supervisely);
    }

    // Object-root: check for VIA project (entries with filename + regions)
    if is_likely_via_project(&value) {
        return Ok(ConvertFormat::Via);
    }

    // Object-root detection: COCO-vs-IR heuristic.

    // Get annotations array
    let annotations = value.get("annotations").and_then(|v| v.as_array());

    let Some(annotations) = annotations else {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "missing or invalid 'annotations' array. Cannot determine format.".to_string(),
        });
    };

    if annotations.is_empty() {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "empty 'annotations' array. Cannot determine format from empty dataset. Use --from to specify format explicitly.".to_string(),
        });
    }

    // Inspect the first annotation's bbox
    let first_ann = &annotations[0];
    let bbox = first_ann.get("bbox");

    let Some(bbox) = bbox else {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "first annotation has no 'bbox' field. Cannot determine format.".to_string(),
        });
    };

    // Check if bbox is an array (COCO) or object (IR JSON)
    if let Some(arr) = bbox.as_array() {
        // COCO uses [x, y, width, height] - array of 4 numbers
        if arr.len() == 4 && arr.iter().all(|v| v.is_number()) {
            return Ok(ConvertFormat::Coco);
        }
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!(
                "bbox is an array but not [x,y,w,h] format (found {} elements). Cannot determine format.",
                arr.len()
            ),
        });
    }

    if let Some(obj) = bbox.as_object() {
        // IR JSON uses {min: {x, y}, max: {x, y}} or {xmin, ymin, xmax, ymax}
        // Check for the serialized format from our bbox.rs
        if obj.contains_key("min") && obj.contains_key("max") {
            return Ok(ConvertFormat::IrJson);
        }
        // Alternative flat format
        if obj.contains_key("xmin")
            && obj.contains_key("ymin")
            && obj.contains_key("xmax")
            && obj.contains_key("ymax")
        {
            return Ok(ConvertFormat::IrJson);
        }
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "bbox is an object but doesn't match IR JSON format (expected min/max or xmin/ymin/xmax/ymax). Cannot determine format.".to_string(),
        });
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "bbox has unexpected type (expected array or object). Cannot determine format."
            .to_string(),
    })
}

/// Detect whether an XML file is CVAT XML.
///
/// Heuristic:
/// - root `<annotations>` => CVAT
/// - root `<annotation>` => looks like a single VOC XML (not auto-detected)
fn detect_xml_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let xml = std::fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let doc = roxmltree::Document::parse(&xml).map_err(|source| {
        PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed to parse XML while detecting format: {source}"),
        }
    })?;

    match doc.root_element().tag_name().name() {
        "annotations" => Ok(ConvertFormat::Cvat),
        "Page" => {
            if ir::io_marmot_xml::is_likely_marmot_xml_str(&xml, path)? {
                Ok(ConvertFormat::Marmot)
            } else {
                Err(PanlabelError::FormatDetectionFailed {
                    path: path.to_path_buf(),
                    reason: "XML root is <Page>, but Page@CropBox is missing or malformed; cannot determine format. Use --from to specify format explicitly.".to_string(),
                })
            }
        }
        "annotation" => Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "XML root is <annotation> (looks like a single VOC file). Panlabel expects VOC as a directory layout; use --from voc with a VOC dataset directory.".to_string(),
        }),
        other => Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("unrecognized XML root element <{other}>; cannot determine format. Use --from to specify format explicitly."),
        }),
    }
}

fn is_likely_label_studio_task(value: &serde_json::Value) -> bool {
    let Some(task_obj) = value.as_object() else {
        return false;
    };

    let Some(data_obj) = task_obj.get("data").and_then(|v| v.as_object()) else {
        return false;
    };

    data_obj
        .get("image")
        .map(|value| value.is_string())
        .unwrap_or(false)
}

/// Detect whether a JSON array element looks like a CreateML image row.
///
/// Heuristic: object with `image` (string) and `annotations` (array) keys.
fn is_likely_createml_item(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    let has_image = obj.get("image").map(|v| v.is_string()).unwrap_or(false);

    let has_annotations = obj
        .get("annotations")
        .map(|v| v.is_array())
        .unwrap_or(false);

    has_image && has_annotations
}

/// Detect whether a JSON object looks like a LabelMe annotation file.
///
/// Heuristic: object with `shapes` (array) key.
fn is_likely_labelme_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    obj.get("shapes").map(|v| v.is_array()).unwrap_or(false)
}

/// Detect whether a JSON object looks like an IBM Cloud Annotations localization file.
///
/// Heuristic: object with `type: "localization"`, `labels` array, and image-keyed
/// `annotations` object.
fn is_likely_cloud_annotations_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    obj.get("type").and_then(|v| v.as_str()) == Some("localization")
        && obj.get("labels").map(|v| v.is_array()).unwrap_or(false)
        && obj
            .get("annotations")
            .map(|v| v.is_object())
            .unwrap_or(false)
}

/// Detect whether a JSON object looks like a Microsoft VoTT JSON export.
///
/// Heuristic:
/// - aggregate project: top-level `assets` object/array whose first entry has
///   `asset` and `regions`
/// - per-asset file: top-level `asset` object and `regions` array
fn is_likely_vott_json_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    if obj.get("asset").map(|v| v.is_object()).unwrap_or(false)
        && obj.get("regions").map(|v| v.is_array()).unwrap_or(false)
    {
        return true;
    }

    let Some(assets) = obj.get("assets") else {
        return false;
    };

    if let Some(asset_map) = assets.as_object() {
        return asset_map.values().any(is_likely_vott_asset_entry);
    }

    if let Some(asset_array) = assets.as_array() {
        return asset_array.iter().any(is_likely_vott_asset_entry);
    }

    false
}

fn is_likely_vott_asset_entry(value: &serde_json::Value) -> bool {
    value.get("asset").map(|v| v.is_object()).unwrap_or(false)
        && value.get("regions").map(|v| v.is_array()).unwrap_or(false)
}

/// Detect whether a JSON object looks like a SuperAnnotate annotation file.
///
/// Heuristic: object with `metadata.width`, `metadata.height`, and an `instances` array.
fn is_likely_superannotate_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    let has_instances = obj.get("instances").map(|v| v.is_array()).unwrap_or(false);
    let has_dimensions = obj
        .get("metadata")
        .and_then(|metadata| metadata.as_object())
        .map(|metadata| {
            metadata
                .get("width")
                .is_some_and(serde_json::Value::is_number)
                && metadata
                    .get("height")
                    .is_some_and(serde_json::Value::is_number)
        })
        .unwrap_or(false);

    has_instances && has_dimensions
}

/// Detect whether a JSON object looks like a Supervisely annotation file.
///
/// Heuristic: object with `size.width`, `size.height`, and an `objects` array.
fn is_likely_supervisely_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    let has_objects = obj.get("objects").map(|v| v.is_array()).unwrap_or(false);
    let has_dimensions = obj
        .get("size")
        .and_then(|size| size.as_object())
        .map(|size| {
            size.get("width").is_some_and(serde_json::Value::is_number)
                && size.get("height").is_some_and(serde_json::Value::is_number)
        })
        .unwrap_or(false);

    has_objects && has_dimensions
}

fn is_likely_via_project(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    // VIA project JSON: top-level keys are image identifiers whose values are
    // objects containing "filename" and "regions" keys.
    obj.values()
        .any(|v| v.is_object() && v.get("filename").is_some() && v.get("regions").is_some())
}

fn is_likely_sagemaker_manifest_row(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    if !obj
        .get("source-ref")
        .and_then(|value| value.as_str())
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        return false;
    }

    obj.iter()
        .filter(|(key, value)| is_likely_sagemaker_label_attribute(obj, key, value))
        .count()
        == 1
}

fn is_likely_sagemaker_label_attribute(
    row: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: &serde_json::Value,
) -> bool {
    if key == "source-ref" || key.ends_with("-metadata") {
        return false;
    }

    let Some(label_obj) = value.as_object() else {
        return false;
    };

    let metadata_key = format!("{key}-metadata");
    let metadata_says_object_detection = row
        .get(&metadata_key)
        .and_then(|metadata| metadata.as_object())
        .and_then(|metadata| metadata.get("type"))
        .and_then(|value| value.as_str())
        .map(|metadata_type| metadata_type == "groundtruth/object-detection")
        .unwrap_or(false);

    let has_detection_shape = label_obj
        .get("annotations")
        .map(|value| value.is_array())
        .unwrap_or(false)
        && label_obj
            .get("image_size")
            .map(|value| value.is_array())
            .unwrap_or(false);

    metadata_says_object_detection || has_detection_shape
}
