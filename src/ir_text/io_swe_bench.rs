use std::path::Path;

use serde_json::Value;

use crate::error::PanlabelError;

use super::io_common::{object_to_metadata, read_rows, SWE_BENCH_REQUIRED_FIELDS};
use super::model::{
    ArtifactKind, ArtifactRef, Metadata, RewardDescriptor, RewardKind, RuntimeDescriptor,
    SolutionDescriptor, TaskInput, TextDataset, TextExample,
};

pub fn read_swe_bench(path: &Path) -> Result<TextDataset, PanlabelError> {
    let rows = read_rows(path)?;
    let mut dataset = TextDataset::default();
    dataset.info.name = Some("swe-bench".to_string());

    for row in rows {
        let object = row
            .value
            .as_object()
            .ok_or_else(|| PanlabelError::TextIrReadError {
                path: path.to_path_buf(),
                message: "SWE-bench rows must be JSON objects".to_string(),
            })?;
        for field in SWE_BENCH_REQUIRED_FIELDS {
            if !object.contains_key(*field) {
                return Err(PanlabelError::UnsupportedFormat(format!(
                    "SWE-bench row is missing required field '{field}'"
                )));
            }
        }
        let instance_id = object
            .get("instance_id")
            .and_then(Value::as_str)
            .unwrap_or("swe-bench-instance")
            .to_string();
        let problem = object
            .get("problem_statement")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let input = if let Some(hints) = object
            .get("hints_text")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        {
            format!("{problem}\n\nHints:\n{hints}")
        } else {
            problem.to_string()
        };
        let mut example = TextExample::task(instance_id, TaskInput::Text { text: input });
        example.split = row.split;
        example.provenance.insert(
            "source_format".to_string(),
            Value::String("swe-bench".to_string()),
        );
        for key in [
            "repo",
            "base_commit",
            "created_at",
            "version",
            "environment_setup_commit",
        ] {
            if let Some(value) = object.get(key) {
                example.provenance.insert(key.to_string(), value.clone());
            }
        }

        let test_patch = object
            .get("test_patch")
            .cloned()
            .unwrap_or(Value::String(String::new()));
        let test_patch_artifact = ArtifactRef {
            path: "test_patch.diff".to_string(),
            kind: Some(ArtifactKind::Patch),
            inline: Some(
                test_patch
                    .as_str()
                    .unwrap_or(&test_patch.to_string())
                    .to_string(),
            ),
            media_type: Some("text/x-diff".to_string()),
        };
        example.data.reward = Some(RewardDescriptor {
            kind: RewardKind::UnitTests,
            artifacts: vec![test_patch_artifact.clone()],
            metadata: Metadata::new(),
        });
        example.data.artifacts.push(test_patch_artifact);

        if let Some(patch) = object.get("patch") {
            let patch_artifact = ArtifactRef {
                path: "gold_patch.diff".to_string(),
                kind: Some(ArtifactKind::Solution),
                inline: Some(patch.as_str().unwrap_or(&patch.to_string()).to_string()),
                media_type: Some("text/x-diff".to_string()),
            };
            example.data.solution = Some(SolutionDescriptor {
                answer: None,
                artifacts: vec![patch_artifact],
                metadata: Metadata::new(),
            });
        }
        let mut runtime_meta = Metadata::new();
        for key in ["repo", "base_commit", "environment_setup_commit"] {
            if let Some(value) = object.get(key) {
                runtime_meta.insert(key.to_string(), value.clone());
            }
        }
        example.data.runtime = Some(RuntimeDescriptor {
            metadata: runtime_meta,
            ..RuntimeDescriptor::default()
        });

        let mut metadata = object_to_metadata(object);
        for key in [
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
            "hints_text",
            "patch",
            "test_patch",
            "environment_setup_commit",
        ] {
            metadata.remove(key);
        }
        example.metadata = metadata;
        dataset.examples.push(example);
    }

    Ok(dataset)
}
