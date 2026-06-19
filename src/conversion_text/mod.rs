pub mod report;

pub use report::{
    TaskConversionIssue, TaskConversionIssueCode, TaskConversionSeverity, TaskConversionStage,
    TextConversionCounts, TextConversionReport,
};

use crate::ir_text::{ArtifactManifest, RewardKind, TextDataset};
use crate::text_format_catalog::TextFormat;

pub fn build_text_conversion_report(
    dataset: &TextDataset,
    artifact_manifest: &ArtifactManifest,
    from: TextFormat,
    to: TextFormat,
) -> TextConversionReport {
    let mut report = TextConversionReport::new(from.name(), to.name());
    report.input = TextConversionCounts {
        examples: dataset.examples.len(),
        artifacts: artifact_manifest.entries.len(),
    };
    report.output = report.input.clone();

    if !artifact_manifest.entries.is_empty() {
        report.add(TaskConversionIssue::writer_info(
            TaskConversionIssueCode::TaskArtifactPreserved,
            format!(
                "{} artifact reference(s) will be preserved by reference or sidecar copy",
                artifact_manifest.entries.len()
            ),
        ));
    }

    for example in &dataset.examples {
        if let Some(reward) = &example.data.reward {
            match reward.kind {
                RewardKind::ShellVerifier | RewardKind::PythonRubric | RewardKind::UnitTests => {
                    if !to.capabilities().executable_reward_artifacts {
                        report.add(TaskConversionIssue::warning(
                            TaskConversionIssueCode::TaskRewardNonportable,
                            format!(
                                "example '{}' has executable reward behavior that target '{}' cannot represent directly",
                                example.id,
                                to.name()
                            ),
                        ));
                    }
                }
                RewardKind::ExternalHarness | RewardKind::LlmJudge => {
                    if !to.capabilities().external_harness {
                        report.add(TaskConversionIssue::warning(
                            TaskConversionIssueCode::TaskHarnessNonportable,
                            format!(
                                "example '{}' depends on harness or judge behavior that target '{}' cannot represent directly",
                                example.id,
                                to.name()
                            ),
                        ));
                    }
                }
                RewardKind::AnswerMatch | RewardKind::MetadataOnly | RewardKind::Unknown(_) => {}
            }
        }
        if example.data.runtime.is_some() && !to.capabilities().runtime_image_workspace {
            report.add(TaskConversionIssue::warning(
                TaskConversionIssueCode::TaskRuntimeDropped,
                format!(
                    "example '{}' has runtime metadata that target '{}' cannot represent",
                    example.id,
                    to.name()
                ),
            ));
        }
        if example.data.solution.is_some() && !to.capabilities().solution_artifacts {
            report.add(TaskConversionIssue::warning(
                TaskConversionIssueCode::TaskSolutionDropped,
                format!(
                    "example '{}' has solution data that target '{}' cannot represent",
                    example.id,
                    to.name()
                ),
            ));
        }
    }

    report
}
