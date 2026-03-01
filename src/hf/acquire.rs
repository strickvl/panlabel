use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};

use crate::error::PanlabelError;

use super::preflight::HfPreflight;
use super::HfRepoRef;

/// Metadata file format chosen during acquisition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HfMetadataFormat {
    Jsonl,
    Parquet,
}

/// Result of remote acquisition.
#[derive(Clone, Debug)]
pub struct HfAcquireResult {
    pub cache_paths: Vec<PathBuf>,
    pub split_dir: PathBuf,
    pub metadata_format: HfMetadataFormat,
    pub metadata_path: PathBuf,
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

    let selected_metadata =
        select_metadata_path(&sibling_paths, requested_split).ok_or_else(|| {
            PanlabelError::HfAcquireError {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "could not find metadata.jsonl or metadata.parquet{}",
                    requested_split
                        .map(|split| format!(" for split '{split}'"))
                        .unwrap_or_default()
                ),
            }
        })?;

    let metadata_local =
        repo.download(&selected_metadata.path)
            .map_err(|source| PanlabelError::HfAcquireError {
                repo_id: repo_ref.repo_id.clone(),
                message: format!(
                    "failed downloading '{}': {}",
                    selected_metadata.path, source
                ),
            })?;

    let mut downloaded = vec![metadata_local.clone()];

    let metadata_dir_remote = Path::new(&selected_metadata.path)
        .parent()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_default();

    let files_to_download: BTreeSet<String> = match selected_metadata.format {
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
        HfMetadataFormat::Parquet => sibling_paths
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
            .collect(),
    };

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

    Ok(HfAcquireResult {
        cache_paths: downloaded,
        split_dir,
        metadata_format: selected_metadata.format,
        metadata_path: metadata_local,
        split_name: selected_metadata.split_name,
    })
}

#[derive(Clone, Debug)]
struct MetadataCandidate {
    path: String,
    format: HfMetadataFormat,
    split_name: Option<String>,
    depth: usize,
}

fn select_metadata_path(
    paths: &[String],
    requested_split: Option<&str>,
) -> Option<MetadataCandidate> {
    let mut candidates = metadata_candidates(paths);
    if candidates.is_empty() {
        return None;
    }

    if let Some(split) = requested_split {
        candidates.retain(|candidate| candidate.split_name.as_deref() == Some(split));
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

fn metadata_candidates(paths: &[String]) -> Vec<MetadataCandidate> {
    let mut candidates = Vec::new();

    for path in paths {
        let (format, file_name) = if path.ends_with("/metadata.jsonl") || path == "metadata.jsonl" {
            (HfMetadataFormat::Jsonl, "metadata.jsonl")
        } else if path.ends_with("/metadata.parquet") || path == "metadata.parquet" {
            (HfMetadataFormat::Parquet, "metadata.parquet")
        } else {
            continue;
        };

        let depth = path.split('/').count();
        let split_name = Path::new(path)
            .parent()
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .filter(|name| !name.is_empty() && name != file_name);

        candidates.push(MetadataCandidate {
            path: path.clone(),
            format,
            split_name,
            depth,
        });
    }

    candidates
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
}
