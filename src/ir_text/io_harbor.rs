use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::PanlabelError;

use super::model::{
    ArtifactKind, ArtifactRef, HarnessDescriptor, Metadata, RewardDescriptor, RewardKind,
    RuntimeDescriptor, SolutionDescriptor, TaskInput, TaskStep, TextDataset, TextExample,
};

#[derive(Clone, Debug, Default)]
pub struct HarborWriteOptions {
    pub scaffold: bool,
    pub source_root: Option<PathBuf>,
}

pub fn read_harbor(path: &Path) -> Result<TextDataset, PanlabelError> {
    let task_dirs = discover_task_dirs(path)?;
    let mut dataset = TextDataset::default();
    dataset.info.name = Some("harbor".to_string());
    for task_dir in task_dirs {
        dataset.examples.push(read_harbor_task(path, &task_dir)?);
    }
    Ok(dataset)
}

pub fn write_harbor(
    path: &Path,
    dataset: &TextDataset,
    options: &HarborWriteOptions,
) -> Result<(), PanlabelError> {
    if dataset.examples.len() == 1 {
        write_one_harbor_task(path, &dataset.examples[0], options)
    } else {
        fs::create_dir_all(path).map_err(PanlabelError::Io)?;
        for example in &dataset.examples {
            let safe_id = super::sanitize_example_id(&example.id);
            write_one_harbor_task(&path.join(safe_id), example, options)?;
        }
        Ok(())
    }
}

pub fn harbor_can_write_without_scaffold(dataset: &TextDataset) -> bool {
    dataset.examples.iter().all(|example| {
        example
            .data
            .reward
            .as_ref()
            .map(reward_has_harbor_native_verifier)
            .unwrap_or(false)
    })
}

fn reward_has_harbor_native_verifier(reward: &RewardDescriptor) -> bool {
    matches!(
        reward.kind,
        RewardKind::ShellVerifier | RewardKind::UnitTests | RewardKind::ExternalHarness
    ) && reward
        .artifacts
        .iter()
        .any(is_harbor_native_verifier_artifact)
}

fn is_harbor_native_verifier_artifact(artifact: &ArtifactRef) -> bool {
    matches!(
        artifact.kind,
        Some(ArtifactKind::Test | ArtifactKind::Script)
    ) || artifact
        .path
        .rsplit('/')
        .next()
        .map(|name| name.ends_with(".sh"))
        .unwrap_or(false)
}

fn discover_task_dirs(path: &Path) -> Result<Vec<PathBuf>, PanlabelError> {
    if path.join("task.toml").exists()
        && (path.join("instruction.md").exists() || path.join("steps").exists())
    {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut dirs = Vec::new();
    for entry in fs::read_dir(path).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        if !entry.file_type().map_err(PanlabelError::Io)?.is_dir() {
            continue;
        }
        let child = entry.path();
        if child.join("task.toml").exists()
            && (child.join("instruction.md").exists() || child.join("steps").exists())
        {
            dirs.push(child);
        }
    }
    if dirs.is_empty() {
        return Err(PanlabelError::UnsupportedFormat(format!(
            "{} is not a Harbor task directory (expected task.toml plus instruction.md or steps/)",
            path.display()
        )));
    }
    dirs.sort();
    Ok(dirs)
}

fn read_harbor_task(root: &Path, task_dir: &Path) -> Result<TextExample, PanlabelError> {
    let toml_path = task_dir.join("task.toml");
    let toml = fs::read_to_string(&toml_path).map_err(PanlabelError::Io)?;
    let task_name = find_toml_string(&toml, "name").unwrap_or_else(|| {
        task_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("harbor-task")
            .to_string()
    });
    let schema_version = find_toml_string(&toml, "schema_version");

    let instruction_path = task_dir.join("instruction.md");
    let input = if instruction_path.exists() {
        TaskInput::Text {
            text: fs::read_to_string(&instruction_path).map_err(PanlabelError::Io)?,
        }
    } else {
        TaskInput::Text {
            text: find_toml_string(&toml, "description").unwrap_or_else(|| task_name.clone()),
        }
    };
    let mut example = TextExample::task(task_name, input);
    example.provenance.insert(
        "source_format".to_string(),
        Value::String("harbor".to_string()),
    );
    if let Some(version) = schema_version.clone() {
        example.metadata.insert(
            "harbor_schema_version".to_string(),
            Value::String(version.clone()),
        );
        if version != "1.3" {
            example.metadata.insert(
                "harbor_compatibility".to_string(),
                Value::String(
                    "source schema_version is not 1.3; output will be normalized to 1.3"
                        .to_string(),
                ),
            );
        }
    }
    example
        .metadata
        .insert("harbor_task_toml".to_string(), Value::String(toml));

    let rel_base = task_dir.strip_prefix(root).unwrap_or(task_dir);
    let tests_dir = task_dir.join("tests");
    if tests_dir.exists() {
        let artifact = ArtifactRef {
            path: rel_base.join("tests/test.sh").to_string_lossy().to_string(),
            kind: Some(ArtifactKind::Test),
            inline: None,
            media_type: Some("text/x-shellscript".to_string()),
        };
        example.data.reward = Some(RewardDescriptor {
            kind: RewardKind::ShellVerifier,
            artifacts: vec![artifact.clone()],
            metadata: Metadata::new(),
        });
        example.data.harness = Some(HarnessDescriptor {
            name: Some("harbor".to_string()),
            artifacts: vec![artifact],
            metadata: Metadata::new(),
        });
    }
    let solution = task_dir.join("solution/solve.sh");
    if solution.exists() {
        example.data.solution = Some(SolutionDescriptor {
            answer: None,
            artifacts: vec![ArtifactRef {
                path: rel_base
                    .join("solution/solve.sh")
                    .to_string_lossy()
                    .to_string(),
                kind: Some(ArtifactKind::Solution),
                inline: None,
                media_type: Some("text/x-shellscript".to_string()),
            }],
            metadata: Metadata::new(),
        });
    }
    let dockerfile = task_dir.join("environment/Dockerfile");
    if dockerfile.exists() {
        example.data.runtime = Some(RuntimeDescriptor {
            image: None,
            workspace: None,
            artifacts: vec![ArtifactRef {
                path: rel_base
                    .join("environment/Dockerfile")
                    .to_string_lossy()
                    .to_string(),
                kind: Some(ArtifactKind::Dockerfile),
                inline: None,
                media_type: Some("text/x-dockerfile".to_string()),
            }],
            metadata: Metadata::new(),
        });
    }

    let steps_dir = task_dir.join("steps");
    if steps_dir.exists() {
        for entry in fs::read_dir(&steps_dir).map_err(PanlabelError::Io)? {
            let entry = entry.map_err(PanlabelError::Io)?;
            if !entry.file_type().map_err(PanlabelError::Io)?.is_dir() {
                continue;
            }
            let step_dir = entry.path();
            let id = step_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("step")
                .to_string();
            let instruction = step_dir.join("instruction.md");
            if instruction.exists() {
                example.data.steps.push(TaskStep {
                    id,
                    input: TaskInput::Text {
                        text: fs::read_to_string(instruction).map_err(PanlabelError::Io)?,
                    },
                    expected_output: None,
                    metadata: Metadata::new(),
                });
            }
        }
        example.data.steps.sort_by(|a, b| a.id.cmp(&b.id));
    }

    Ok(example)
}

fn write_one_harbor_task(
    path: &Path,
    example: &TextExample,
    options: &HarborWriteOptions,
) -> Result<(), PanlabelError> {
    fs::create_dir_all(path).map_err(PanlabelError::Io)?;
    fs::write(
        path.join("instruction.md"),
        task_input_to_text(&example.data.input),
    )
    .map_err(PanlabelError::Io)?;

    let has_native_verifier = example
        .data
        .reward
        .as_ref()
        .map(reward_has_harbor_native_verifier)
        .unwrap_or(false);
    let mut written_artifacts = HashSet::new();
    if !has_native_verifier && !options.scaffold {
        return Err(PanlabelError::UnsupportedFormat(
            "writing Harbor requires an executable verifier artifact; pass --scaffold to write a fail-fast placeholder".to_string(),
        ));
    }

    if let Some(reward) = &example.data.reward {
        if !reward.artifacts.is_empty() {
            fs::create_dir_all(path.join("tests")).map_err(PanlabelError::Io)?;
            for (idx, artifact) in reward.artifacts.iter().enumerate() {
                let target = if idx == 0 && has_native_verifier {
                    path.join("tests/test.sh")
                } else {
                    path.join("tests")
                        .join(artifact_file_name(artifact, "verifier-artifact"))
                };
                materialize_harbor_artifact(
                    artifact,
                    options.source_root.as_deref(),
                    &target,
                    &mut written_artifacts,
                )?;
            }
        }
    }

    if !has_native_verifier && options.scaffold {
        fs::create_dir_all(path.join("tests")).map_err(PanlabelError::Io)?;
        let mut file = fs::File::create(path.join("tests/test.sh")).map_err(PanlabelError::Io)?;
        file.write_all(b"#!/usr/bin/env bash\nset -euo pipefail\necho 'Panlabel scaffold verifier: replace this file with real Harbor tests.' >&2\nexit 1\n")
            .map_err(PanlabelError::Io)?;
    }

    if let Some(solution) = &example.data.solution {
        fs::create_dir_all(path.join("solution")).map_err(PanlabelError::Io)?;
        for (idx, artifact) in solution.artifacts.iter().enumerate() {
            let target = if idx == 0 {
                path.join("solution/solve.sh")
            } else {
                path.join("solution")
                    .join(artifact_file_name(artifact, "solution-artifact"))
            };
            materialize_harbor_artifact(
                artifact,
                options.source_root.as_deref(),
                &target,
                &mut written_artifacts,
            )?;
        }
    }

    if let Some(runtime) = &example.data.runtime {
        fs::create_dir_all(path.join("environment")).map_err(PanlabelError::Io)?;
        for artifact in &runtime.artifacts {
            let default_name = if artifact.kind == Some(ArtifactKind::Dockerfile) {
                "Dockerfile"
            } else {
                "environment-artifact"
            };
            let target = path
                .join("environment")
                .join(artifact_file_name(artifact, default_name));
            materialize_harbor_artifact(
                artifact,
                options.source_root.as_deref(),
                &target,
                &mut written_artifacts,
            )?;
        }
    }

    if let Some(harness) = &example.data.harness {
        fs::create_dir_all(path.join("tests")).map_err(PanlabelError::Io)?;
        for artifact in &harness.artifacts {
            let target = path
                .join("tests")
                .join(artifact_file_name(artifact, "harness-artifact"));
            materialize_harbor_artifact(
                artifact,
                options.source_root.as_deref(),
                &target,
                &mut written_artifacts,
            )?;
        }
    }

    for artifact in &example.data.artifacts {
        let target = native_or_sidecar_target(path, artifact);
        materialize_harbor_artifact(
            artifact,
            options.source_root.as_deref(),
            &target,
            &mut written_artifacts,
        )?;
    }

    let task_name = escape_toml_string(&example.id);
    let task_toml = format!(
        "schema_version = \"1.3\"\n\n[task]\nname = \"{task_name}\"\ndescription = \"Generated by panlabel text convert.\"\n\n[verifier]\ntimeout_sec = 30.0\n\n[agent]\ntimeout_sec = 30.0\n\n[environment]\nnetwork_mode = \"no-network\"\n"
    );
    fs::write(path.join("task.toml"), task_toml).map_err(PanlabelError::Io)
}

fn native_or_sidecar_target(task_dir: &Path, artifact: &ArtifactRef) -> PathBuf {
    match artifact.kind {
        Some(ArtifactKind::Test | ArtifactKind::Script) => task_dir
            .join("tests")
            .join(artifact_file_name(artifact, "test-artifact")),
        Some(ArtifactKind::Solution) => task_dir
            .join("solution")
            .join(artifact_file_name(artifact, "solution-artifact")),
        Some(ArtifactKind::Dockerfile | ArtifactKind::Environment) => task_dir
            .join("environment")
            .join(artifact_file_name(artifact, "environment-artifact")),
        _ => task_dir
            .join(".panlabel")
            .join("artifacts")
            .join(&artifact.path),
    }
}

fn artifact_file_name(artifact: &ArtifactRef, default: &str) -> String {
    Path::new(&artifact.path)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(default)
        .to_string()
}

fn materialize_harbor_artifact(
    artifact: &ArtifactRef,
    source_root: Option<&Path>,
    target: &Path,
    written_artifacts: &mut HashSet<String>,
) -> Result<(), PanlabelError> {
    let key = format!(
        "{}|{}",
        target.to_string_lossy(),
        super::artifact::artifact_logical_key(artifact)
    );
    if !written_artifacts.insert(key) {
        return Ok(());
    }
    super::materialize_artifact_to_target(artifact, source_root, target)
}

fn task_input_to_text(input: &TaskInput) -> String {
    match input {
        TaskInput::Text { text } => text.clone(),
        TaskInput::Messages { messages } => messages
            .iter()
            .map(|message| {
                let role = format!("{:?}", message.role).to_ascii_lowercase();
                let content = message
                    .content
                    .iter()
                    .map(|block| match block {
                        super::model::Block::Text { text } => text.clone(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{role}: {content}")
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
    }
}

fn find_toml_string(toml: &str, key: &str) -> Option<String> {
    for line in toml.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with(key) {
            continue;
        }
        let (_, rhs) = trimmed.split_once('=')?;
        let value = rhs.trim();
        if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
            return Some(value[1..value.len() - 1].to_string());
        }
    }
    None
}

fn escape_toml_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
