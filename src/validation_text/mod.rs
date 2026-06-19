mod report;

pub use report::{
    TextIssueContext, TextValidationIssue, TextValidationIssueCode, TextValidationReport,
    TextValidationSeverity,
};

use std::collections::HashMap;
use std::path::Path;

use crate::ir_text::{
    artifact_refs_for_example, validate_relative_artifact_path, RewardKind, TaskInput, TextDataset,
    TextExampleKind,
};
use crate::text_format_catalog::TextFormat;

#[derive(Clone, Debug, Default)]
pub struct TextValidateOptions {
    pub strict: bool,
}

pub fn validate_text_dataset(
    dataset: &TextDataset,
    source_root: Option<&Path>,
    _opts: &TextValidateOptions,
) -> TextValidationReport {
    let mut report = TextValidationReport::new();
    let mut seen: HashMap<&str, usize> = HashMap::new();

    for (idx, example) in dataset.examples.iter().enumerate() {
        if example.id.trim().is_empty() {
            report.add(TextValidationIssue::error(
                TextValidationIssueCode::EmptyId,
                "example ID is empty",
                TextIssueContext::Example {
                    id: example.id.clone(),
                },
            ));
        }
        if let Some(first_idx) = seen.insert(example.id.as_str(), idx) {
            report.add(TextValidationIssue::error(
                TextValidationIssueCode::DuplicateId,
                format!(
                    "duplicate example ID '{}' (first seen at index {first_idx})",
                    example.id
                ),
                TextIssueContext::Example {
                    id: example.id.clone(),
                },
            ));
        }
        if example.kind != TextExampleKind::Task {
            report.add(TextValidationIssue::error(
                TextValidationIssueCode::UnsupportedKind,
                "only kind=\"task\" is supported in this implementation pass",
                TextIssueContext::Example {
                    id: example.id.clone(),
                },
            ));
        }
        if task_input_is_empty(&example.data.input) {
            report.add(TextValidationIssue::error(
                TextValidationIssueCode::EmptyTaskInput,
                "task input is empty",
                TextIssueContext::Example {
                    id: example.id.clone(),
                },
            ));
        }
        if let Some(split) = &example.split {
            if split.trim().is_empty() {
                report.add(TextValidationIssue::warning(
                    TextValidationIssueCode::EmptySplitName,
                    "split name is empty",
                    TextIssueContext::Example {
                        id: example.id.clone(),
                    },
                ));
            }
        }
        if example.data.gold_answer.is_none() && example.data.reward.is_none() {
            report.add(TextValidationIssue::warning(
                TextValidationIssueCode::MissingGoldAnswerOrReward,
                "task has neither a gold answer nor a reward descriptor",
                TextIssueContext::Example {
                    id: example.id.clone(),
                },
            ));
        }
        if let Some(gold_answer) = &example.data.gold_answer {
            if !gold_answer.is_string() {
                report.add(TextValidationIssue::warning(
                    TextValidationIssueCode::ComplexGoldAnswer,
                    "gold answer is structured; some targets only accept strings",
                    TextIssueContext::Example {
                        id: example.id.clone(),
                    },
                ));
            }
        }
        if let Some(reward) = &example.data.reward {
            if matches!(
                reward.kind,
                RewardKind::ShellVerifier | RewardKind::PythonRubric | RewardKind::UnitTests
            ) && reward.artifacts.is_empty()
            {
                report.add(TextValidationIssue::warning(
                    TextValidationIssueCode::ExecutableRewardWithoutArtifacts,
                    "executable reward descriptor has no artifact references",
                    TextIssueContext::Example {
                        id: example.id.clone(),
                    },
                ));
            }
        }
        for artifact in artifact_refs_for_example(example) {
            if let Err(message) = validate_relative_artifact_path(&artifact.path) {
                report.add(TextValidationIssue::error(
                    TextValidationIssueCode::UnsafeArtifactPath,
                    message,
                    TextIssueContext::Artifact {
                        example_id: example.id.clone(),
                        path: artifact.path.clone(),
                    },
                ));
                continue;
            }
            if artifact.inline.is_none() {
                if let Some(root) = source_root {
                    if !root.join(&artifact.path).exists() {
                        report.add(TextValidationIssue::error(
                            TextValidationIssueCode::MissingArtifact,
                            "artifact file is missing",
                            TextIssueContext::Artifact {
                                example_id: example.id.clone(),
                                path: artifact.path.clone(),
                            },
                        ));
                    }
                }
            }
        }
    }

    report
}

pub fn validate_text_target_compatibility(
    dataset: &TextDataset,
    target: TextFormat,
) -> TextValidationReport {
    let mut report = TextValidationReport::new();
    let capabilities = target.capabilities();
    for example in &dataset.examples {
        if example.data.runtime.is_some() && !capabilities.runtime_image_workspace {
            report.add(TextValidationIssue::warning(
                TextValidationIssueCode::RuntimeNotRepresentable,
                format!(
                    "target '{}' cannot represent runtime descriptors",
                    target.name()
                ),
                TextIssueContext::Example {
                    id: example.id.clone(),
                },
            ));
        }
    }
    report
}

fn task_input_is_empty(input: &TaskInput) -> bool {
    match input {
        TaskInput::Text { text } => text.trim().is_empty(),
        TaskInput::Messages { messages } => messages.iter().all(|message| {
            message.content.iter().all(|block| match block {
                crate::ir_text::Block::Text { text } => text.trim().is_empty(),
            })
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_text::{TaskInput, TextExample};

    #[test]
    fn detects_duplicate_ids_and_empty_input() {
        let dataset = TextDataset {
            examples: vec![
                TextExample::task("same", TaskInput::Text { text: "".into() }),
                TextExample::task(
                    "same",
                    TaskInput::Text {
                        text: "hello".into(),
                    },
                ),
            ],
            ..TextDataset::default()
        };
        let report = validate_text_dataset(&dataset, None, &TextValidateOptions::default());
        assert_eq!(report.error_count(), 2);
    }
}
