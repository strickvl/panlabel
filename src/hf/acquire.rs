use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use walkdir::WalkDir;
use zip::ZipArchive;

use crate::error::PanlabelError;

use super::preflight::HfPreflight;
use super::HfRepoRef;

const HF_ZIP_MAX_UNCOMPRESSED_BYTES: u64 = 10 * 1024 * 1024 * 1024; // 10 GiB
const HF_ZIP_MAX_ENTRIES: usize = 200_000;

/// Metadata file format chosen during acquisition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HfMetadataFormat {
    Jsonl,
    Parquet,
}

/// Physical payload format selected after remote acquisition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HfAcquirePayloadFormat {
    HfImagefolder,
    Yolo,
    Voc,
    Coco,
}

/// Result of remote acquisition.
#[derive(Clone, Debug)]
pub struct HfAcquireResult {
    pub cache_paths: Vec<PathBuf>,
    pub payload_path: PathBuf,
    pub payload_format: HfAcquirePayloadFormat,
    pub split_dir: PathBuf,
    pub metadata_format: Option<HfMetadataFormat>,
    pub metadata_path: Option<PathBuf>,
    pub split_name: Option<String>,
}

/// Download the minimum files required to read an HF ImageFolder dataset.
pub fn acquire(
    repo_ref: &HfRepoRef,
    preflight: Option<&HfPreflight>,
    token: Option<&str>,
) -> Result<HfAcquireResult, PanlabelError> {
    let mut builder = ApiBuilder::new().with_progress(false);

    let token_from_env = std::env::var("HF_TOKEN").ok();
    let effective_token = token.map(str::to_string).or(token_from_env);
    if effective_token.is_some() {
        builder = builder.with_token(effective_token);
    }

    let api = builder
        .build()
        .map_err(|source| PanlabelError::HfApiError {
            repo_id: repo_ref.repo_id.clone(),
            message: source.to_string(),
        })?;

    let repo = if let Some(revision) = repo_ref.revision.as_ref() {
        api.repo(Repo::with_revision(
            repo_ref.repo_id.clone(),
            RepoType::Dataset,
            revision.clone(),
        ))
    } else {
        api.dataset(repo_ref.repo_id.clone())
    };

    let repo_info = repo.info().map_err(|source| PanlabelError::HfApiError {
        repo_id: repo_ref.repo_id.clone(),
        message: source.to_string(),
    })?;

    let sibling_paths: Vec<String> = repo_info
        .siblings
        .iter()
        .map(|sibling| sibling.rfilename.clone())
        .collect();
    let sibling_set: BTreeSet<String> = sibling_paths.iter().cloned().collect();

    let requested_split = repo_ref
        .split
        .as_deref()
        .or_else(|| preflight.and_then(|p| p.selected_split.as_deref()));

    if let Some(selected_metadata) = select_metadata_path(&sibling_paths, requested_split) {
        let metadata_local = repo.download(&selected_metadata.path).map_err(|source| {
            PanlabelError::HfAcquireError {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "failed downloading '{}': {}",
                    selected_metadata.path, source
                ),
            }
        })?;

        let mut downloaded = vec![metadata_local.clone()];

        let metadata_dir_remote = Path::new(&selected_metadata.path)
            .parent()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut files_to_download: BTreeSet<String> = match selected_metadata.format {
            HfMetadataFormat::Jsonl => {
                let referenced = read_jsonl_file_names(&metadata_local).map_err(|source| {
                    PanlabelError::HfAcquireError {
                        repo_id: repo_ref.repo_id.clone(),
                        message: format!(
                            "failed parsing '{}' for image references: {}",
                            selected_metadata.path, source
                        ),
                    }
                })?;

                referenced
                    .into_iter()
                    .filter_map(|file_name| {
                        resolve_remote_image_path(&metadata_dir_remote, &file_name, &sibling_set)
                    })
                    .collect()
            }
            HfMetadataFormat::Parquet => {
                if selected_metadata.is_metadata_file {
                    sibling_paths
                        .iter()
                        .filter(|path| {
                            is_image_file(path)
                                && if metadata_dir_remote.is_empty() {
                                    true
                                } else {
                                    path.starts_with(&format!("{}/", metadata_dir_remote))
                                }
                        })
                        .cloned()
                        .collect()
                } else {
                    select_related_parquet_shards(
                        &sibling_paths,
                        &selected_metadata,
                        requested_split,
                    )
                }
            }
        };

        files_to_download.remove(&selected_metadata.path);

        for remote_path in files_to_download {
            let local =
                repo.download(&remote_path)
                    .map_err(|source| PanlabelError::HfAcquireError {
                        repo_id: repo_ref.repo_id.clone(),
                        message: format!("failed downloading '{}': {}", remote_path, source),
                    })?;
            downloaded.push(local);
        }

        let split_dir = metadata_local
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| PanlabelError::HfAcquireError {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "downloaded metadata path '{}' has no parent directory",
                    metadata_local.display()
                ),
            })?;

        return Ok(HfAcquireResult {
            cache_paths: downloaded,
            payload_path: split_dir.clone(),
            payload_format: HfAcquirePayloadFormat::HfImagefolder,
            split_dir,
            metadata_format: Some(selected_metadata.format),
            metadata_path: Some(metadata_local),
            split_name: selected_metadata.split_name,
        });
    }

    let selected_zip = select_zip_path(&sibling_paths, requested_split).ok_or_else(|| {
        PanlabelError::HfAcquireError {
            repo_id: repo_ref.repo_id.clone(),
            message: format!(
                "could not find a supported HF annotation layout (metadata.jsonl, metadata.parquet, split parquet shards, or split .zip archives){}",
                requested_split
                    .map(|split| format!(" for split '{split}'"))
                    .unwrap_or_default()
            ),
        }
    })?;

    let zip_local =
        repo.download(&selected_zip.path)
            .map_err(|source| PanlabelError::HfAcquireError {
                repo_id: repo_ref.repo_id.clone(),
                message: format!("failed downloading '{}': {}", selected_zip.path, source),
            })?;

    let extract_root = build_extract_root(repo_ref, &selected_zip.path);
    if extract_root.exists() {
        std::fs::remove_dir_all(&extract_root).map_err(|source| {
            PanlabelError::HfZipLayoutInvalid {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "failed cleaning previous extracted directory '{}': {}",
                    extract_root.display(),
                    source
                ),
            }
        })?;
    }
    std::fs::create_dir_all(&extract_root).map_err(|source| PanlabelError::HfZipLayoutInvalid {
        repo_id: repo_ref.repo_id.clone(),
        message: format!(
            "failed creating extracted directory '{}': {}",
            extract_root.display(),
            source
        ),
    })?;

    extract_zip_archive(repo_ref, &selected_zip.path, &zip_local, &extract_root)?;
    let payload = select_zip_payload(repo_ref, &extract_root, requested_split)?;

    Ok(HfAcquireResult {
        cache_paths: vec![zip_local],
        payload_path: payload.path.clone(),
        payload_format: payload.format,
        split_dir: payload.path,
        metadata_format: None,
        metadata_path: None,
        split_name: selected_zip.split_name.or(payload.split_name),
    })
}

#[derive(Clone, Debug)]
struct MetadataCandidate {
    path: String,
    format: HfMetadataFormat,
    split_name: Option<String>,
    depth: usize,
    is_metadata_file: bool,
}

#[derive(Clone, Debug)]
struct ZipCandidate {
    path: String,
    split_name: Option<String>,
    depth: usize,
}

#[derive(Clone, Debug)]
struct ZipPayloadCandidate {
    path: PathBuf,
    format: HfAcquirePayloadFormat,
    split_name: Option<String>,
}

fn select_metadata_path(
    paths: &[String],
    requested_split: Option<&str>,
) -> Option<MetadataCandidate> {
    let mut candidates = metadata_candidates(paths);
    if candidates.is_empty() {
        candidates = parquet_shard_candidates(paths);
        if candidates.is_empty() {
            return None;
        }
    }

    if let Some(split) = requested_split {
        let normalized_split = normalize_split_name(split).unwrap_or(split);
        candidates.retain(|candidate| {
            candidate
                .split_name
                .as_deref()
                .map(|value| value == normalized_split)
                .unwrap_or(false)
        });
        if candidates.is_empty() {
            return None;
        }
    }

    candidates.sort_by(|a, b| match (a.format, b.format) {
        (HfMetadataFormat::Jsonl, HfMetadataFormat::Parquet) => std::cmp::Ordering::Less,
        (HfMetadataFormat::Parquet, HfMetadataFormat::Jsonl) => std::cmp::Ordering::Greater,
        _ => a.depth.cmp(&b.depth).then_with(|| a.path.cmp(&b.path)),
    });

    if requested_split.is_none() {
        if let Some(root_jsonl) = candidates
            .iter()
            .find(|candidate| candidate.depth == 1 && candidate.format == HfMetadataFormat::Jsonl)
            .cloned()
        {
            return Some(root_jsonl);
        }
        if let Some(root_parquet) = candidates
            .iter()
            .find(|candidate| candidate.depth == 1 && candidate.format == HfMetadataFormat::Parquet)
            .cloned()
        {
            return Some(root_parquet);
        }

        if let Some(train_split) = candidates
            .iter()
            .find(|candidate| candidate.split_name.as_deref() == Some("train"))
            .cloned()
        {
            return Some(train_split);
        }
    }

    candidates.into_iter().next()
}

fn select_zip_path(paths: &[String], requested_split: Option<&str>) -> Option<ZipCandidate> {
    let mut candidates = zip_candidates(paths);
    if candidates.is_empty() {
        return None;
    }

    if let Some(split) = requested_split {
        let normalized = normalize_split_name(split).unwrap_or(split);
        candidates.sort_by(|a, b| {
            let a_score = match a.split_name.as_deref() {
                Some(name) if name == normalized => 2_i32,
                Some(_) => 0_i32,
                None => 1_i32,
            };
            let b_score = match b.split_name.as_deref() {
                Some(name) if name == normalized => 2_i32,
                Some(_) => 0_i32,
                None => 1_i32,
            };
            b_score
                .cmp(&a_score)
                .then_with(|| a.depth.cmp(&b.depth))
                .then_with(|| a.path.cmp(&b.path))
        });

        let best = candidates.first().cloned()?;
        let best_score = match best.split_name.as_deref() {
            Some(name) if name == normalized => 2_i32,
            Some(_) => 0_i32,
            None => 1_i32,
        };
        if best_score == 0 {
            return None;
        }
        return Some(best);
    }

    candidates.sort_by(|a, b| {
        let a_score = default_split_preference(a.split_name.as_deref());
        let b_score = default_split_preference(b.split_name.as_deref());
        b_score
            .cmp(&a_score)
            .then_with(|| a.depth.cmp(&b.depth))
            .then_with(|| a.path.cmp(&b.path))
    });

    candidates.into_iter().next()
}

fn default_split_preference(split_name: Option<&str>) -> i32 {
    match split_name {
        Some("train") => 4,
        Some("validation") => 3,
        Some("test") => 2,
        Some("dev") => 1,
        Some(_) => 1,
        None => 2,
    }
}

fn zip_candidates(paths: &[String]) -> Vec<ZipCandidate> {
    let mut candidates = Vec::new();

    for path in paths {
        if !path.to_ascii_lowercase().ends_with(".zip") {
            continue;
        }
        let split_name = infer_split_from_dataset_path(path);
        let depth = path.split('/').count();
        candidates.push(ZipCandidate {
            path: path.clone(),
            split_name,
            depth,
        });
    }

    candidates
}

fn metadata_candidates(paths: &[String]) -> Vec<MetadataCandidate> {
    let mut candidates = Vec::new();

    for path in paths {
        let format = if path.ends_with("/metadata.jsonl") || path == "metadata.jsonl" {
            HfMetadataFormat::Jsonl
        } else if path.ends_with("/metadata.parquet") || path == "metadata.parquet" {
            HfMetadataFormat::Parquet
        } else {
            continue;
        };

        let depth = path.split('/').count();
        let split_name = infer_split_from_dataset_path(path);

        candidates.push(MetadataCandidate {
            path: path.clone(),
            format,
            split_name,
            depth,
            is_metadata_file: true,
        });
    }

    candidates
}

fn parquet_shard_candidates(paths: &[String]) -> Vec<MetadataCandidate> {
    let mut candidates = Vec::new();

    for path in paths {
        if !path.ends_with(".parquet") {
            continue;
        }
        if path.ends_with("metadata.parquet") {
            continue;
        }

        let depth = path.split('/').count();
        let split_name = infer_split_from_dataset_path(path);

        candidates.push(MetadataCandidate {
            path: path.clone(),
            format: HfMetadataFormat::Parquet,
            split_name,
            depth,
            is_metadata_file: false,
        });
    }

    candidates
}

fn infer_split_from_dataset_path(path: &str) -> Option<String> {
    let parsed = Path::new(path);
    if let Some(file_name) = parsed.file_name().and_then(|name| name.to_str()) {
        let lowered = file_name.to_ascii_lowercase();
        if let Some((prefix, _)) = lowered.split_once('-') {
            if let Some(normalized) = normalize_split_name(prefix) {
                return Some(normalized.to_string());
            }
        }

        let stem = Path::new(&lowered)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or(&lowered);
        if let Some(normalized) = normalize_split_name(stem) {
            return Some(normalized.to_string());
        }
        for token in stem.split(|ch: char| !ch.is_ascii_alphanumeric()) {
            if let Some(normalized) = normalize_split_name(token) {
                return Some(normalized.to_string());
            }
        }
    }

    for component in parsed.components().rev() {
        let Some(name) = component.as_os_str().to_str() else {
            continue;
        };
        if let Some(normalized) = normalize_split_name(name) {
            return Some(normalized.to_string());
        }
    }

    None
}

fn normalize_split_name(name: &str) -> Option<&str> {
    match name.to_ascii_lowercase().as_str() {
        "train" => Some("train"),
        "test" => Some("test"),
        "validation" | "valid" | "val" => Some("validation"),
        "dev" => Some("dev"),
        _ => None,
    }
}

fn select_related_parquet_shards(
    paths: &[String],
    selected: &MetadataCandidate,
    requested_split: Option<&str>,
) -> BTreeSet<String> {
    let mut selected_paths = BTreeSet::new();
    let target_split = requested_split
        .and_then(normalize_split_name)
        .map(str::to_string)
        .or_else(|| selected.split_name.clone());
    let selected_parent = Path::new(&selected.path)
        .parent()
        .map(|parent| parent.to_string_lossy().to_string());

    for path in paths {
        if !path.ends_with(".parquet") || path.ends_with("metadata.parquet") {
            continue;
        }

        if let Some(parent) = selected_parent.as_deref() {
            let path_parent = Path::new(path)
                .parent()
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_default();
            if path_parent != parent {
                continue;
            }
        }

        if let Some(split) = target_split.as_deref() {
            if infer_split_from_dataset_path(path).as_deref() != Some(split) {
                continue;
            }
        }

        selected_paths.insert(path.clone());
    }

    if selected_paths.is_empty() {
        selected_paths.insert(selected.path.clone());
    }

    selected_paths
}

fn read_jsonl_file_names(path: &Path) -> Result<Vec<String>, std::io::Error> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut file_names = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(file_name) = value.get("file_name").and_then(serde_json::Value::as_str) {
                file_names.push(file_name.to_string());
            }
        }
    }

    Ok(file_names)
}

fn resolve_remote_image_path(
    metadata_dir_remote: &str,
    file_name: &str,
    siblings: &BTreeSet<String>,
) -> Option<String> {
    let joined = if metadata_dir_remote.is_empty() {
        file_name.to_string()
    } else {
        format!("{}/{}", metadata_dir_remote, file_name)
    };

    if siblings.contains(&joined) {
        return Some(joined);
    }
    if siblings.contains(file_name) {
        return Some(file_name.to_string());
    }

    None
}

fn build_extract_root(repo_ref: &HfRepoRef, remote_zip_path: &str) -> PathBuf {
    let stem = Path::new(remote_zip_path)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("dataset");
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "panlabel-hf-{}-{}-{}",
        sanitize_for_path(&repo_ref.repo_id),
        sanitize_for_path(stem),
        timestamp
    ))
}

fn sanitize_for_path(raw: &str) -> String {
    raw.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch,
            _ => '-',
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn extract_zip_archive(
    repo_ref: &HfRepoRef,
    remote_zip_path: &str,
    zip_local_path: &Path,
    extract_root: &Path,
) -> Result<(), PanlabelError> {
    let file = std::fs::File::open(zip_local_path).map_err(PanlabelError::Io)?;
    let mut archive =
        ZipArchive::new(file).map_err(|source| PanlabelError::HfZipLayoutInvalid {
            repo_id: repo_ref.repo_id.clone(),
            message: format!(
                "failed opening zip archive '{}': {}",
                remote_zip_path, source
            ),
        })?;

    if archive.len() > HF_ZIP_MAX_ENTRIES {
        return Err(PanlabelError::HfZipLayoutInvalid {
            repo_id: repo_ref.repo_id.clone(),
            message: format!(
                "zip '{}' has too many entries ({} > {})",
                remote_zip_path,
                archive.len(),
                HF_ZIP_MAX_ENTRIES
            ),
        });
    }

    let mut total_uncompressed_bytes: u64 = 0;

    for index in 0..archive.len() {
        let mut entry =
            archive
                .by_index(index)
                .map_err(|source| PanlabelError::HfZipLayoutInvalid {
                    repo_id: repo_ref.repo_id.clone(),
                    message: format!(
                        "failed reading zip entry #{index} from '{}': {}",
                        remote_zip_path, source
                    ),
                })?;
        total_uncompressed_bytes = total_uncompressed_bytes.saturating_add(entry.size());
        if total_uncompressed_bytes > HF_ZIP_MAX_UNCOMPRESSED_BYTES {
            return Err(PanlabelError::HfZipLayoutInvalid {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "zip '{}' exceeds max uncompressed size (>{} bytes)",
                    remote_zip_path, HF_ZIP_MAX_UNCOMPRESSED_BYTES
                ),
            });
        }
        let Some(enclosed_name) = entry.enclosed_name().map(Path::to_path_buf) else {
            return Err(PanlabelError::HfZipLayoutInvalid {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "zip '{}' contains an unsafe path (possible traversal entry '{}')",
                    remote_zip_path,
                    entry.name()
                ),
            });
        };
        let output_path = extract_root.join(enclosed_name);
        if entry.is_dir() {
            std::fs::create_dir_all(&output_path).map_err(|source| {
                PanlabelError::HfZipLayoutInvalid {
                    repo_id: repo_ref.repo_id.clone(),
                    message: format!(
                        "failed creating directory '{}' while extracting '{}': {}",
                        output_path.display(),
                        remote_zip_path,
                        source
                    ),
                }
            })?;
            continue;
        }

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                PanlabelError::HfZipLayoutInvalid {
                    repo_id: repo_ref.repo_id.clone(),
                    message: format!(
                        "failed creating parent directory '{}' while extracting '{}': {}",
                        parent.display(),
                        remote_zip_path,
                        source
                    ),
                }
            })?;
        }
        let mut out_file = std::fs::File::create(&output_path).map_err(|source| {
            PanlabelError::HfZipLayoutInvalid {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "failed creating file '{}' while extracting '{}': {}",
                    output_path.display(),
                    remote_zip_path,
                    source
                ),
            }
        })?;
        std::io::copy(&mut entry, &mut out_file).map_err(|source| {
            PanlabelError::HfZipLayoutInvalid {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "failed writing '{}' from zip '{}': {}",
                    output_path.display(),
                    remote_zip_path,
                    source
                ),
            }
        })?;
    }

    Ok(())
}

fn select_zip_payload(
    repo_ref: &HfRepoRef,
    extract_root: &Path,
    requested_split: Option<&str>,
) -> Result<ZipPayloadCandidate, PanlabelError> {
    let mut candidates = collect_zip_payload_candidates(extract_root)?;
    if candidates.is_empty() {
        return Err(PanlabelError::HfZipLayoutInvalid {
            repo_id: repo_ref.repo_id.clone(),
            message: format!(
                "zip extracted to '{}' but no supported dataset layout was found (expected YOLO, VOC, COCO, or HF metadata layout)",
                extract_root.display()
            ),
        });
    }

    if let Some(split) = requested_split {
        let normalized = normalize_split_name(split).unwrap_or(split).to_string();
        let best_score = candidates
            .iter()
            .map(|candidate| requested_split_score(candidate.split_name.as_deref(), &normalized))
            .max()
            .unwrap_or(0);
        if best_score == 0 {
            return Err(PanlabelError::HfZipLayoutInvalid {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "zip archive does not contain a supported payload for split '{}'",
                    split
                ),
            });
        }
        candidates.retain(|candidate| {
            requested_split_score(candidate.split_name.as_deref(), &normalized) == best_score
        });

        let best_depth = candidates
            .iter()
            .map(|candidate| candidate.path.components().count())
            .min()
            .expect("checked non-empty");
        candidates.retain(|candidate| candidate.path.components().count() == best_depth);

        if has_format_ambiguity(&candidates) {
            return Err(PanlabelError::HfZipLayoutInvalid {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "zip split '{}' has ambiguous payload formats at depth {}: {}",
                    split,
                    best_depth,
                    summarize_payload_candidates(&candidates)
                ),
            });
        }

        candidates.sort_by(|a, b| {
            a.path
                .cmp(&b.path)
                .then_with(|| format_rank(a.format).cmp(&format_rank(b.format)))
        });
        return Ok(candidates.into_iter().next().expect("checked non-empty"));
    }

    let best_score = candidates
        .iter()
        .map(|candidate| default_split_preference(candidate.split_name.as_deref()))
        .max()
        .unwrap_or(0);
    candidates.retain(|candidate| {
        default_split_preference(candidate.split_name.as_deref()) == best_score
    });

    let best_depth = candidates
        .iter()
        .map(|candidate| candidate.path.components().count())
        .min()
        .expect("checked non-empty");
    candidates.retain(|candidate| candidate.path.components().count() == best_depth);

    if has_format_ambiguity(&candidates) {
        return Err(PanlabelError::HfZipLayoutInvalid {
            repo_id: repo_ref.repo_id.clone(),
            message: format!(
                "zip archive has ambiguous payload formats at best candidate depth {}: {}",
                best_depth,
                summarize_payload_candidates(&candidates)
            ),
        });
    }

    candidates.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then_with(|| format_rank(a.format).cmp(&format_rank(b.format)))
    });
    Ok(candidates.into_iter().next().expect("checked non-empty"))
}

fn requested_split_score(split_name: Option<&str>, requested_split: &str) -> i32 {
    match split_name {
        Some(name) if name == requested_split => 2,
        Some(_) => 0,
        None => 1,
    }
}

fn has_format_ambiguity(candidates: &[ZipPayloadCandidate]) -> bool {
    if candidates.len() <= 1 {
        return false;
    }
    let mut seen = BTreeSet::new();
    for candidate in candidates {
        seen.insert(format_rank(candidate.format));
    }
    seen.len() > 1
}

fn summarize_payload_candidates(candidates: &[ZipPayloadCandidate]) -> String {
    candidates
        .iter()
        .map(|candidate| {
            let split = candidate
                .split_name
                .as_deref()
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            format!(
                "{} [{} split={}]",
                candidate.path.display(),
                payload_format_name(candidate.format),
                split
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn payload_format_name(format: HfAcquirePayloadFormat) -> &'static str {
    match format {
        HfAcquirePayloadFormat::HfImagefolder => "hf",
        HfAcquirePayloadFormat::Yolo => "yolo",
        HfAcquirePayloadFormat::Voc => "voc",
        HfAcquirePayloadFormat::Coco => "coco",
    }
}

fn collect_zip_payload_candidates(
    extract_root: &Path,
) -> Result<Vec<ZipPayloadCandidate>, PanlabelError> {
    let mut candidates = Vec::new();

    if directory_has_hf_metadata(extract_root)? {
        candidates.push(ZipPayloadCandidate {
            path: extract_root.to_path_buf(),
            format: HfAcquirePayloadFormat::HfImagefolder,
            split_name: infer_split_from_extracted_path(extract_root, extract_root),
        });
    }
    if directory_looks_like_yolo(extract_root)? {
        candidates.push(ZipPayloadCandidate {
            path: extract_root.to_path_buf(),
            format: HfAcquirePayloadFormat::Yolo,
            split_name: infer_split_from_extracted_path(extract_root, extract_root),
        });
    }
    if directory_looks_like_voc(extract_root)? {
        candidates.push(ZipPayloadCandidate {
            path: extract_root.to_path_buf(),
            format: HfAcquirePayloadFormat::Voc,
            split_name: infer_split_from_extracted_path(extract_root, extract_root),
        });
    }

    for entry in WalkDir::new(extract_root).follow_links(false) {
        let entry = match entry {
            Ok(value) => value,
            Err(_) => continue,
        };
        let path = entry.path();

        if entry.file_type().is_dir() {
            if directory_has_hf_metadata(path)? {
                candidates.push(ZipPayloadCandidate {
                    path: path.to_path_buf(),
                    format: HfAcquirePayloadFormat::HfImagefolder,
                    split_name: infer_split_from_extracted_path(extract_root, path),
                });
            }
            if directory_looks_like_yolo(path)? {
                candidates.push(ZipPayloadCandidate {
                    path: path.to_path_buf(),
                    format: HfAcquirePayloadFormat::Yolo,
                    split_name: infer_split_from_extracted_path(extract_root, path),
                });
            }
            if directory_looks_like_voc(path)? {
                candidates.push(ZipPayloadCandidate {
                    path: path.to_path_buf(),
                    format: HfAcquirePayloadFormat::Voc,
                    split_name: infer_split_from_extracted_path(extract_root, path),
                });
            }
            continue;
        }

        if entry.file_type().is_file() && file_looks_like_coco(path) {
            candidates.push(ZipPayloadCandidate {
                path: path.to_path_buf(),
                format: HfAcquirePayloadFormat::Coco,
                split_name: infer_split_from_extracted_path(extract_root, path),
            });
        }
    }

    candidates.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then_with(|| format_rank(a.format).cmp(&format_rank(b.format)))
    });
    candidates.dedup_by(|a, b| a.path == b.path && a.format == b.format);

    Ok(candidates)
}

fn infer_split_from_extracted_path(extract_root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(extract_root).unwrap_or(path);
    if relative.as_os_str().is_empty() || relative == Path::new(".") {
        return None;
    }
    infer_split_from_dataset_path(&relative.to_string_lossy())
}

fn format_rank(format: HfAcquirePayloadFormat) -> i32 {
    match format {
        HfAcquirePayloadFormat::HfImagefolder => 0,
        HfAcquirePayloadFormat::Yolo => 1,
        HfAcquirePayloadFormat::Voc => 2,
        HfAcquirePayloadFormat::Coco => 3,
    }
}

fn directory_has_hf_metadata(path: &Path) -> Result<bool, PanlabelError> {
    if path.join("metadata.jsonl").is_file() || path.join("metadata.parquet").is_file() {
        return Ok(true);
    }
    for entry in std::fs::read_dir(path).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        let child = entry.path();
        if !child.is_dir() {
            continue;
        }
        if child.join("metadata.jsonl").is_file() || child.join("metadata.parquet").is_file() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn directory_looks_like_yolo(path: &Path) -> Result<bool, PanlabelError> {
    if !path.is_dir() {
        return Ok(false);
    }

    let labels_dir = if path.join("labels").is_dir() {
        path.join("labels")
    } else if path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("labels"))
        .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        return Ok(false);
    };

    directory_contains_extension(&labels_dir, "txt")
}

fn directory_looks_like_voc(path: &Path) -> Result<bool, PanlabelError> {
    if !path.is_dir() {
        return Ok(false);
    }

    let annotations_dir = path.join("Annotations");
    let images_dir = path.join("JPEGImages");
    if annotations_dir.is_dir() && images_dir.is_dir() {
        return directory_contains_extension(&annotations_dir, "xml");
    }

    if path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("Annotations"))
        .unwrap_or(false)
    {
        let has_images_sibling = path
            .parent()
            .map(|parent| parent.join("JPEGImages").is_dir())
            .unwrap_or(false);
        if has_images_sibling {
            return directory_contains_extension(path, "xml");
        }
    }

    Ok(false)
}

fn directory_contains_extension(path: &Path, ext: &str) -> Result<bool, PanlabelError> {
    for entry in std::fs::read_dir(path).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        let child = entry.path();
        if !child.is_file() {
            continue;
        }
        if child
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.eq_ignore_ascii_case(ext))
            .unwrap_or(false)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn file_looks_like_coco(path: &Path) -> bool {
    if !path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
    {
        return false;
    }

    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };

    let normalized = name.to_ascii_lowercase();
    normalized.contains("coco")
        || normalized.contains("instances")
        || normalized.ends_with("annotations.json")
}

fn is_image_file(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "webp" | "gif" | "tif" | "tiff"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn metadata_selection_prefers_root_jsonl() {
        let files = vec![
            "train/metadata.jsonl".to_string(),
            "metadata.parquet".to_string(),
            "metadata.jsonl".to_string(),
        ];

        let selected = select_metadata_path(&files, None).expect("selection");
        assert_eq!(selected.path, "metadata.jsonl");
        assert_eq!(selected.format, HfMetadataFormat::Jsonl);
    }

    #[test]
    fn metadata_selection_uses_requested_split() {
        let files = vec![
            "train/metadata.parquet".to_string(),
            "validation/metadata.jsonl".to_string(),
            "train/metadata.jsonl".to_string(),
        ];

        let selected = select_metadata_path(&files, Some("train")).expect("selection");
        assert_eq!(selected.path, "train/metadata.jsonl");
        assert_eq!(selected.split_name.as_deref(), Some("train"));
    }

    #[test]
    fn resolve_remote_image_path_checks_split_dir_then_root() {
        let siblings = BTreeSet::from(["train/img1.jpg".to_string(), "img2.jpg".to_string()]);

        assert_eq!(
            resolve_remote_image_path("train", "img1.jpg", &siblings).as_deref(),
            Some("train/img1.jpg")
        );
        assert_eq!(
            resolve_remote_image_path("train", "img2.jpg", &siblings).as_deref(),
            Some("img2.jpg")
        );
    }

    #[test]
    fn metadata_selection_falls_back_to_parquet_shards() {
        let files = vec![
            "data/train-00000-of-00002.parquet".to_string(),
            "data/train-00001-of-00002.parquet".to_string(),
            "data/test-00000-of-00001.parquet".to_string(),
        ];

        let selected = select_metadata_path(&files, Some("train")).expect("selection");
        assert_eq!(selected.format, HfMetadataFormat::Parquet);
        assert!(!selected.is_metadata_file);
        assert_eq!(selected.split_name.as_deref(), Some("train"));
    }

    #[test]
    fn related_parquet_shards_selects_all_for_split() {
        let files = vec![
            "data/train-00000-of-00002.parquet".to_string(),
            "data/train-00001-of-00002.parquet".to_string(),
            "data/validation-00000-of-00001.parquet".to_string(),
            "data/test-00000-of-00001.parquet".to_string(),
        ];
        let selected = select_metadata_path(&files, Some("train")).expect("selection");
        let related = select_related_parquet_shards(&files, &selected, Some("train"));
        assert_eq!(
            related,
            BTreeSet::from([
                "data/train-00000-of-00002.parquet".to_string(),
                "data/train-00001-of-00002.parquet".to_string()
            ])
        );
    }

    #[test]
    fn related_parquet_shards_stays_with_selected_parent_dir() {
        let files = vec![
            "config_a/train-00000-of-00002.parquet".to_string(),
            "config_a/train-00001-of-00002.parquet".to_string(),
            "config_b/train-00000-of-00001.parquet".to_string(),
        ];

        let selected = select_metadata_path(&files, Some("train")).expect("selection");
        assert_eq!(
            Path::new(&selected.path)
                .parent()
                .and_then(|value| value.to_str()),
            Some("config_a")
        );

        let related = select_related_parquet_shards(&files, &selected, Some("train"));
        assert_eq!(
            related,
            BTreeSet::from([
                "config_a/train-00000-of-00002.parquet".to_string(),
                "config_a/train-00001-of-00002.parquet".to_string()
            ])
        );
    }

    #[test]
    fn zip_selection_prefers_requested_split() {
        let files = vec![
            "data/train.zip".to_string(),
            "data/valid.zip".to_string(),
            "data/test.zip".to_string(),
        ];

        let selected = select_zip_path(&files, Some("train")).expect("zip selection");
        assert_eq!(selected.path, "data/train.zip");
        assert_eq!(selected.split_name.as_deref(), Some("train"));

        let selected_validation =
            select_zip_path(&files, Some("validation")).expect("validation selection");
        assert_eq!(selected_validation.path, "data/valid.zip");
        assert_eq!(
            selected_validation.split_name.as_deref(),
            Some("validation")
        );
    }

    #[test]
    fn zip_selection_defaults_to_train_when_unspecified() {
        let files = vec![
            "data/test.zip".to_string(),
            "data/train.zip".to_string(),
            "data/valid.zip".to_string(),
        ];
        let selected = select_zip_path(&files, None).expect("zip selection");
        assert_eq!(selected.path, "data/train.zip");
    }

    #[test]
    fn zip_payload_detection_finds_yolo_split_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let split_dir = temp.path().join("train");
        fs::create_dir_all(split_dir.join("labels")).expect("labels dir");
        fs::create_dir_all(split_dir.join("images")).expect("images dir");
        fs::write(split_dir.join("labels/sample.txt"), "0 0.5 0.5 0.1 0.1\n").expect("label file");

        let repo_ref = HfRepoRef {
            repo_id: "org/dataset".to_string(),
            revision: None,
            config: None,
            split: Some("train".to_string()),
        };

        let payload =
            select_zip_payload(&repo_ref, temp.path(), Some("train")).expect("payload select");
        assert_eq!(payload.format, HfAcquirePayloadFormat::Yolo);
        assert_eq!(payload.split_name.as_deref(), Some("train"));
        assert_eq!(payload.path, split_dir);
    }

    #[test]
    fn zip_payload_detection_errors_on_ambiguous_best_candidate() {
        let temp = tempfile::tempdir().expect("tempdir");
        let split_dir = temp.path().join("train");
        fs::create_dir_all(split_dir.join("labels")).expect("labels dir");
        fs::create_dir_all(split_dir.join("images")).expect("images dir");
        fs::write(split_dir.join("labels/sample.txt"), "0 0.5 0.5 0.1 0.1\n").expect("label file");

        fs::create_dir_all(split_dir.join("Annotations")).expect("annotations dir");
        fs::create_dir_all(split_dir.join("JPEGImages")).expect("voc images dir");
        fs::write(
            split_dir.join("Annotations/sample.xml"),
            "<annotation><filename>sample.jpg</filename><size><width>100</width><height>100</height><depth>3</depth></size><object><name>obj</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>",
        )
        .expect("voc xml");

        let repo_ref = HfRepoRef {
            repo_id: "org/dataset".to_string(),
            revision: None,
            config: None,
            split: Some("train".to_string()),
        };

        let err =
            select_zip_payload(&repo_ref, temp.path(), Some("train")).expect_err("should fail");
        match err {
            PanlabelError::HfZipLayoutInvalid { message, .. } => {
                assert!(message.contains("ambiguous payload formats"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn split_inference_ignores_extract_root_name_tokens() {
        let root = Path::new("/tmp/panlabel-hf-org-dataset-train-1234");
        assert_eq!(infer_split_from_extracted_path(root, root), None);
        assert_eq!(
            infer_split_from_extracted_path(root, &root.join("validation")).as_deref(),
            Some("validation")
        );
    }
}
