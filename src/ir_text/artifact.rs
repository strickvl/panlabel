use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::error::PanlabelError;

use super::model::{ArtifactRef, TextDataset, TextExample};

#[derive(Clone, Debug)]
pub struct TextDatasetBundle {
    pub dataset: TextDataset,
    pub source_root: PathBuf,
    pub artifact_manifest: ArtifactManifest,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArtifactManifest {
    pub entries: Vec<ArtifactManifestEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactManifestEntry {
    pub example_id: String,
    pub artifact: ArtifactRef,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArtifactWritePlan {
    pub entries: Vec<ArtifactWritePlanEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactWritePlanEntry {
    pub example_id: String,
    pub source_ref: String,
    pub source_path: Option<PathBuf>,
    pub target_ref: Option<String>,
    pub record_ref: Option<String>,
    pub inline: Option<String>,
    pub disposition: ArtifactDisposition,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactDisposition {
    NativeTargetLocation,
    SidecarCopy,
    ReferenceOnly,
    Dropped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactOutputLayout {
    File,
    TextIrDirectory,
    TargetDirectory,
}

impl ArtifactManifest {
    pub fn from_dataset(dataset: &TextDataset) -> Self {
        let entries = dataset
            .examples
            .iter()
            .flat_map(|example| {
                artifact_refs_for_example(example)
                    .into_iter()
                    .cloned()
                    .map(|artifact| ArtifactManifestEntry {
                        example_id: example.id.clone(),
                        artifact,
                    })
            })
            .collect();
        Self { entries }
    }
}

pub fn artifact_refs_for_example(example: &TextExample) -> Vec<&ArtifactRef> {
    let mut artifacts = Vec::new();
    let mut seen = HashSet::new();
    push_unique_artifact_refs(&mut artifacts, &mut seen, &example.data.artifacts);
    if let Some(reward) = &example.data.reward {
        push_unique_artifact_refs(&mut artifacts, &mut seen, &reward.artifacts);
    }
    if let Some(runtime) = &example.data.runtime {
        push_unique_artifact_refs(&mut artifacts, &mut seen, &runtime.artifacts);
    }
    if let Some(harness) = &example.data.harness {
        push_unique_artifact_refs(&mut artifacts, &mut seen, &harness.artifacts);
    }
    if let Some(solution) = &example.data.solution {
        push_unique_artifact_refs(&mut artifacts, &mut seen, &solution.artifacts);
    }
    artifacts
}

fn push_unique_artifact_refs<'a>(
    output: &mut Vec<&'a ArtifactRef>,
    seen: &mut HashSet<String>,
    artifacts: &'a [ArtifactRef],
) {
    for artifact in artifacts {
        if seen.insert(artifact_logical_key(artifact)) {
            output.push(artifact);
        }
    }
}

pub(crate) fn artifact_logical_key(artifact: &ArtifactRef) -> String {
    format!(
        "{}|{:?}|{:?}|{:?}",
        artifact.path, artifact.kind, artifact.inline, artifact.media_type
    )
}

impl TextDatasetBundle {
    pub fn new(dataset: TextDataset, source_root: impl Into<PathBuf>) -> Self {
        let artifact_manifest = ArtifactManifest::from_dataset(&dataset);
        Self {
            dataset,
            source_root: source_root.into(),
            artifact_manifest,
        }
    }
}

pub fn validate_relative_artifact_path(path: &str) -> Result<(), String> {
    if path.trim().is_empty() {
        return Err("artifact path is empty".to_string());
    }
    if looks_like_windows_drive_path(path) {
        return Err("Windows drive-prefixed artifact paths are not allowed".to_string());
    }
    let as_path = Path::new(path);
    if as_path.is_absolute() {
        return Err("absolute artifact paths are not allowed".to_string());
    }
    for component in as_path.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => return Err("artifact paths cannot contain '..'".to_string()),
            Component::RootDir => {
                return Err("root-relative artifact paths are not allowed".to_string())
            }
            Component::Prefix(_) => return Err("path prefixes are not allowed".to_string()),
        }
    }
    Ok(())
}

pub fn safe_example_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> BTreeMap<String, String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut mapping = BTreeMap::new();
    for id in ids {
        if mapping.contains_key(id) {
            continue;
        }
        let base = sanitize_example_id(id);
        let count = counts.entry(base.clone()).or_insert(0);
        let safe = if *count == 0 {
            base.clone()
        } else {
            format!("{base}-{}", *count + 1)
        };
        *count += 1;
        mapping.insert(id.to_string(), safe);
    }
    mapping
}

pub fn sanitize_example_id(id: &str) -> String {
    let sanitized: String = id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "example".to_string()
    } else {
        sanitized
    }
}

pub fn build_sidecar_artifact_write_plan(
    manifest: &ArtifactManifest,
    source_root: &Path,
    output_path: &Path,
    layout: ArtifactOutputLayout,
) -> Result<ArtifactWritePlan, PanlabelError> {
    let id_map = safe_example_ids(
        manifest
            .entries
            .iter()
            .map(|entry| entry.example_id.as_str()),
    );
    let mut entries = Vec::new();
    let mut seen = HashSet::new();

    for entry in &manifest.entries {
        validate_relative_artifact_path(&entry.artifact.path).map_err(|message| {
            PanlabelError::TextArtifactInvalid {
                path: PathBuf::from(&entry.artifact.path),
                message,
            }
        })?;
        let plan_key = format!(
            "{}|{}",
            entry.example_id,
            artifact_logical_key(&entry.artifact)
        );
        if !seen.insert(plan_key) {
            continue;
        }
        let source = source_root.join(&entry.artifact.path);
        let source_path = if entry.artifact.inline.is_some() {
            None
        } else {
            Some(resolve_artifact_source(source_root, &source)?)
        };
        let safe_id = id_map
            .get(&entry.example_id)
            .cloned()
            .unwrap_or_else(|| sanitize_example_id(&entry.example_id));
        let (target_ref, record_ref) =
            sidecar_target_refs(output_path, layout, &safe_id, &entry.artifact.path);
        entries.push(ArtifactWritePlanEntry {
            example_id: entry.example_id.clone(),
            source_ref: entry.artifact.path.clone(),
            source_path,
            target_ref: Some(target_ref),
            record_ref: Some(record_ref),
            inline: entry.artifact.inline.clone(),
            disposition: ArtifactDisposition::SidecarCopy,
        });
    }

    Ok(ArtifactWritePlan { entries })
}

pub fn copy_artifacts(plan: &ArtifactWritePlan) -> Result<(), PanlabelError> {
    for entry in &plan.entries {
        if entry.disposition != ArtifactDisposition::SidecarCopy {
            continue;
        }
        let Some(target_ref) = &entry.target_ref else {
            continue;
        };
        let target = Path::new(target_ref);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
        }
        if let Some(source) = &entry.source_path {
            fs::copy(source, target).map_err(PanlabelError::Io)?;
            preserve_executable_bit(source, target)?;
        } else if let Some(inline) = &entry.inline {
            fs::write(target, inline).map_err(PanlabelError::Io)?;
        }
    }
    Ok(())
}

pub fn apply_artifact_write_plan_to_dataset(dataset: &mut TextDataset, plan: &ArtifactWritePlan) {
    let mut entries_by_example: HashMap<&str, Vec<&ArtifactWritePlanEntry>> = HashMap::new();
    for entry in &plan.entries {
        entries_by_example
            .entry(entry.example_id.as_str())
            .or_default()
            .push(entry);
    }

    for example in &mut dataset.examples {
        let Some(entries) = entries_by_example.get(example.id.as_str()) else {
            continue;
        };
        let rewrites = entries
            .iter()
            .filter_map(|entry| {
                entry
                    .record_ref
                    .as_deref()
                    .map(|record_ref| (entry.source_ref.as_str(), record_ref))
            })
            .collect::<HashMap<_, _>>();
        rewrite_example_artifact_refs(example, &rewrites);
    }
}

fn rewrite_example_artifact_refs(example: &mut TextExample, rewrites: &HashMap<&str, &str>) {
    rewrite_artifact_refs(&mut example.data.artifacts, rewrites);
    if let Some(reward) = &mut example.data.reward {
        rewrite_artifact_refs(&mut reward.artifacts, rewrites);
    }
    if let Some(runtime) = &mut example.data.runtime {
        rewrite_artifact_refs(&mut runtime.artifacts, rewrites);
    }
    if let Some(harness) = &mut example.data.harness {
        rewrite_artifact_refs(&mut harness.artifacts, rewrites);
    }
    if let Some(solution) = &mut example.data.solution {
        rewrite_artifact_refs(&mut solution.artifacts, rewrites);
    }
}

fn rewrite_artifact_refs(artifacts: &mut [ArtifactRef], rewrites: &HashMap<&str, &str>) {
    for artifact in artifacts {
        if let Some(record_ref) = rewrites.get(artifact.path.as_str()) {
            artifact.path = (*record_ref).to_string();
        }
    }
}

fn sidecar_target_refs(
    output_path: &Path,
    layout: ArtifactOutputLayout,
    safe_id: &str,
    artifact_path: &str,
) -> (String, String) {
    let (target_root, record_root) = match layout {
        ArtifactOutputLayout::File => {
            let stem = output_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("tasks");
            (
                output_path.with_file_name(format!("{stem}.artifacts")),
                PathBuf::from(format!("{stem}.artifacts")),
            )
        }
        ArtifactOutputLayout::TextIrDirectory => (
            output_path.join("tasks.artifacts"),
            PathBuf::from("tasks.artifacts"),
        ),
        ArtifactOutputLayout::TargetDirectory => (
            output_path.join(".panlabel").join("artifacts"),
            PathBuf::from(".panlabel").join("artifacts"),
        ),
    };
    let suffix = Path::new(safe_id).join(artifact_path);
    (
        target_root.join(&suffix).to_string_lossy().to_string(),
        record_root.join(suffix).to_string_lossy().to_string(),
    )
}

pub fn materialize_artifact_to_target(
    artifact: &ArtifactRef,
    source_root: Option<&Path>,
    target: &Path,
) -> Result<(), PanlabelError> {
    validate_relative_artifact_path(&artifact.path).map_err(|message| {
        PanlabelError::TextArtifactInvalid {
            path: PathBuf::from(&artifact.path),
            message,
        }
    })?;
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }
    if let Some(inline) = &artifact.inline {
        fs::write(target, inline).map_err(PanlabelError::Io)?;
        return Ok(());
    }
    let Some(source_root) = source_root else {
        return Err(PanlabelError::TextArtifactMissing {
            path: PathBuf::from(&artifact.path),
        });
    };
    let source = source_root.join(&artifact.path);
    let resolved = resolve_artifact_source(source_root, &source)?;
    fs::copy(&resolved, target).map_err(PanlabelError::Io)?;
    preserve_executable_bit(&resolved, target)
}

fn resolve_artifact_source(source_root: &Path, path: &Path) -> Result<PathBuf, PanlabelError> {
    if !path.exists() {
        return Err(PanlabelError::TextArtifactMissing {
            path: path.to_path_buf(),
        });
    }
    let root = source_root.canonicalize().map_err(PanlabelError::Io)?;
    let resolved = path.canonicalize().map_err(PanlabelError::Io)?;
    if !resolved.starts_with(&root) {
        return Err(PanlabelError::TextArtifactInvalid {
            path: path.to_path_buf(),
            message: "artifact path resolves outside the source dataset root".to_string(),
        });
    }
    Ok(resolved)
}

fn looks_like_windows_drive_path(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic()
}

#[cfg(unix)]
fn preserve_executable_bit(source: &Path, target: &Path) -> Result<(), PanlabelError> {
    use std::os::unix::fs::PermissionsExt;

    let source_mode = fs::metadata(source)
        .map_err(PanlabelError::Io)?
        .permissions()
        .mode();
    let mut target_permissions = fs::metadata(target)
        .map_err(PanlabelError::Io)?
        .permissions();
    let target_mode = target_permissions.mode();
    let executable_bits = source_mode & 0o111;
    target_permissions.set_mode((target_mode & !0o111) | executable_bits);
    fs::set_permissions(target, target_permissions).map_err(PanlabelError::Io)
}

#[cfg(not(unix))]
fn preserve_executable_bit(_source: &Path, _target: &Path) -> Result<(), PanlabelError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_text::model::{ArtifactRef, TaskInput, TextDataset, TextExample};

    #[test]
    fn rejects_unsafe_paths() {
        assert!(validate_relative_artifact_path("tests/test.sh").is_ok());
        assert!(validate_relative_artifact_path("../secret").is_err());
        assert!(validate_relative_artifact_path("/tmp/secret").is_err());
        assert!(validate_relative_artifact_path("C:/secret").is_err());
    }

    #[test]
    fn duplicate_safe_ids_get_deterministic_suffixes() {
        let ids = safe_example_ids(["a/b", "a:b", "plain"]);
        assert_eq!(ids.get("a/b").map(String::as_str), Some("a_b"));
        assert_eq!(ids.get("a:b").map(String::as_str), Some("a_b-2"));
        assert_eq!(ids.get("plain").map(String::as_str), Some("plain"));
    }

    #[test]
    fn write_plan_reports_file_sidecar_destinations() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::create_dir_all(dir.path().join("tests")).expect("mkdir");
        fs::write(dir.path().join("tests/test.sh"), "echo ok").expect("write artifact");

        let mut dataset = TextDataset::default();
        let mut example = TextExample::task("task/1", TaskInput::Text { text: "Run".into() });
        example.data.artifacts.push(ArtifactRef {
            path: "tests/test.sh".into(),
            kind: None,
            inline: None,
            media_type: None,
        });
        dataset.examples.push(example);
        let manifest = ArtifactManifest::from_dataset(&dataset);

        let plan = build_sidecar_artifact_write_plan(
            &manifest,
            dir.path(),
            &dir.path().join("out.jsonl"),
            ArtifactOutputLayout::File,
        )
        .expect("plan");

        assert_eq!(plan.entries.len(), 1);
        assert!(plan.entries[0]
            .target_ref
            .as_deref()
            .unwrap()
            .ends_with("out.artifacts/task_1/tests/test.sh"));
    }

    #[test]
    fn manifest_deduplicates_logical_artifacts() {
        let mut dataset = TextDataset::default();
        let artifact = ArtifactRef {
            path: "tests/reward.sh".into(),
            kind: None,
            inline: None,
            media_type: None,
        };
        let mut example = TextExample::task("task-1", TaskInput::Text { text: "Run".into() });
        example.data.artifacts.push(artifact.clone());
        example.data.reward = Some(crate::ir_text::RewardDescriptor {
            kind: crate::ir_text::RewardKind::ShellVerifier,
            artifacts: vec![artifact],
            metadata: Default::default(),
        });
        dataset.examples.push(example);

        let manifest = ArtifactManifest::from_dataset(&dataset);
        assert_eq!(manifest.entries.len(), 1);
        assert_eq!(manifest.entries[0].artifact.path, "tests/reward.sh");
    }

    #[test]
    fn manifest_collects_descriptor_artifacts() {
        let mut dataset = TextDataset::default();
        let mut example = TextExample::task("task-1", TaskInput::Text { text: "Run".into() });
        example.data.reward = Some(crate::ir_text::RewardDescriptor {
            kind: crate::ir_text::RewardKind::ShellVerifier,
            artifacts: vec![ArtifactRef {
                path: "tests/reward.sh".into(),
                kind: None,
                inline: None,
                media_type: None,
            }],
            metadata: Default::default(),
        });
        dataset.examples.push(example);

        let manifest = ArtifactManifest::from_dataset(&dataset);
        assert_eq!(manifest.entries.len(), 1);
        assert_eq!(manifest.entries[0].artifact.path, "tests/reward.sh");
    }

    #[test]
    fn text_ir_directory_plan_uses_tasks_artifacts() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::create_dir_all(dir.path().join("tests")).expect("mkdir");
        fs::write(dir.path().join("tests/test.sh"), "echo ok").expect("write artifact");

        let manifest = ArtifactManifest {
            entries: vec![ArtifactManifestEntry {
                example_id: "task-1".into(),
                artifact: ArtifactRef {
                    path: "tests/test.sh".into(),
                    kind: None,
                    inline: None,
                    media_type: None,
                },
            }],
        };
        let plan = build_sidecar_artifact_write_plan(
            &manifest,
            dir.path(),
            &dir.path().join("out_dir"),
            ArtifactOutputLayout::TextIrDirectory,
        )
        .expect("plan");

        assert!(plan.entries[0]
            .target_ref
            .as_deref()
            .unwrap()
            .ends_with("out_dir/tasks.artifacts/task-1/tests/test.sh"));
    }
}
