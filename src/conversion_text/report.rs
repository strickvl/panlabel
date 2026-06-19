use serde::{Serialize, Serializer};
use std::fmt;

#[derive(Clone, Debug, Default, Serialize)]
pub struct TextConversionReport {
    pub from: String,
    pub to: String,
    pub input: TextConversionCounts,
    pub output: TextConversionCounts,
    pub issues: Vec<TaskConversionIssue>,
}

impl TextConversionReport {
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            ..Self::default()
        }
    }

    pub fn add(&mut self, issue: TaskConversionIssue) {
        self.issues.push(issue);
    }

    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == TaskConversionSeverity::Warning)
            .count()
    }

    pub fn info_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == TaskConversionSeverity::Info)
            .count()
    }

    pub fn is_lossy(&self) -> bool {
        self.warning_count() > 0
    }
}

impl fmt::Display for TextConversionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  {} task examples", self.input.examples)?;
        if self.output != self.input {
            writeln!(f, "  output: {} task examples", self.output.examples)?;
        }
        if self.warning_count() > 0 {
            writeln!(f)?;
            writeln!(f, "Warnings ({}):", self.warning_count())?;
            for issue in self
                .issues
                .iter()
                .filter(|issue| issue.severity == TaskConversionSeverity::Warning)
            {
                writeln!(f, "  - [{}] {}", issue.code.as_str(), issue.message)?;
            }
        }
        if self.info_count() > 0 {
            writeln!(f)?;
            writeln!(f, "Notes ({}):", self.info_count())?;
            for issue in self
                .issues
                .iter()
                .filter(|issue| issue.severity == TaskConversionSeverity::Info)
            {
                writeln!(f, "  - [{}] {}", issue.code.as_str(), issue.message)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TextConversionCounts {
    pub examples: usize,
    pub artifacts: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TaskConversionIssue {
    pub severity: TaskConversionSeverity,
    pub stage: TaskConversionStage,
    pub code: TaskConversionIssueCode,
    pub message: String,
}

impl TaskConversionIssue {
    pub fn warning(code: TaskConversionIssueCode, message: impl Into<String>) -> Self {
        Self {
            severity: TaskConversionSeverity::Warning,
            stage: TaskConversionStage::Analysis,
            code,
            message: message.into(),
        }
    }

    pub fn info(code: TaskConversionIssueCode, message: impl Into<String>) -> Self {
        Self {
            severity: TaskConversionSeverity::Info,
            stage: TaskConversionStage::Analysis,
            code,
            message: message.into(),
        }
    }

    pub fn writer_info(code: TaskConversionIssueCode, message: impl Into<String>) -> Self {
        Self {
            severity: TaskConversionSeverity::Info,
            stage: TaskConversionStage::TargetWriter,
            code,
            message: message.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskConversionSeverity {
    Warning,
    Info,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskConversionStage {
    Analysis,
    SourceReader,
    TargetWriter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TaskConversionIssueCode {
    TaskRewardNonportable,
    TaskRewardDropped,
    TaskRewardStub,
    TaskRuntimeDropped,
    TaskHarnessNonportable,
    TaskSolutionDropped,
    TaskMetadataDropped,
    TaskArtifactPreserved,
    TaskRlvrColumnMapping,
    TaskHarborLayoutPolicy,
    TaskSwebBenchPatchPreserved,
}

impl TaskConversionIssueCode {
    pub const ALL: &'static [TaskConversionIssueCode] = &[
        TaskConversionIssueCode::TaskRewardNonportable,
        TaskConversionIssueCode::TaskRewardDropped,
        TaskConversionIssueCode::TaskRewardStub,
        TaskConversionIssueCode::TaskRuntimeDropped,
        TaskConversionIssueCode::TaskHarnessNonportable,
        TaskConversionIssueCode::TaskSolutionDropped,
        TaskConversionIssueCode::TaskMetadataDropped,
        TaskConversionIssueCode::TaskArtifactPreserved,
        TaskConversionIssueCode::TaskRlvrColumnMapping,
        TaskConversionIssueCode::TaskHarborLayoutPolicy,
        TaskConversionIssueCode::TaskSwebBenchPatchPreserved,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            TaskConversionIssueCode::TaskRewardNonportable => "TASK-REWARD-NONPORTABLE",
            TaskConversionIssueCode::TaskRewardDropped => "TASK-REWARD-DROPPED",
            TaskConversionIssueCode::TaskRewardStub => "TASK-REWARD-STUB",
            TaskConversionIssueCode::TaskRuntimeDropped => "TASK-RUNTIME-DROPPED",
            TaskConversionIssueCode::TaskHarnessNonportable => "TASK-HARNESS-NONPORTABLE",
            TaskConversionIssueCode::TaskSolutionDropped => "TASK-SOLUTION-DROPPED",
            TaskConversionIssueCode::TaskMetadataDropped => "TASK-METADATA-DROPPED",
            TaskConversionIssueCode::TaskArtifactPreserved => "TASK-ARTIFACT-PRESERVED",
            TaskConversionIssueCode::TaskRlvrColumnMapping => "TASK-RLVR-COLUMN-MAPPING",
            TaskConversionIssueCode::TaskHarborLayoutPolicy => "TASK-HARBOR-LAYOUT-POLICY",
            TaskConversionIssueCode::TaskSwebBenchPatchPreserved => "TASK-SWEBENCH-PATCH-PRESERVED",
        }
    }
}

impl fmt::Display for TaskConversionIssueCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for TaskConversionIssueCode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn task_issue_codes_are_unique_and_stable() {
        let mut seen = HashSet::new();
        for code in TaskConversionIssueCode::ALL {
            assert!(
                seen.insert(code.as_str()),
                "duplicate code {}",
                code.as_str()
            );
            assert!(code.as_str().starts_with("TASK-"));
        }
        assert_eq!(
            TaskConversionIssueCode::TaskRewardNonportable.as_str(),
            "TASK-REWARD-NONPORTABLE"
        );
        assert_eq!(
            TaskConversionIssueCode::TaskRewardStub.to_string(),
            "TASK-REWARD-STUB"
        );
        assert_eq!(
            serde_json::to_string(&TaskConversionIssueCode::TaskRewardStub).unwrap(),
            "\"TASK-REWARD-STUB\""
        );
    }

    #[test]
    fn task_issue_codes_are_documented() {
        let docs = include_str!("../../docs/conversion.md");
        for code in TaskConversionIssueCode::ALL {
            assert!(
                docs.contains(code.as_str()),
                "{} is missing from docs/conversion.md",
                code.as_str()
            );
        }
    }
}
