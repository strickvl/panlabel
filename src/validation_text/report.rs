use serde::Serialize;
use std::fmt;

#[derive(Clone, Debug, Default, Serialize)]
pub struct TextValidationReport {
    pub issues: Vec<TextValidationIssue>,
}

impl TextValidationReport {
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    pub fn add(&mut self, issue: TextValidationIssue) {
        self.issues.push(issue);
    }

    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == TextValidationSeverity::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == TextValidationSeverity::Warning)
            .count()
    }

    pub fn is_ok(&self) -> bool {
        self.error_count() == 0
    }

    pub fn as_json(&self) -> impl Serialize + '_ {
        TextValidationReportJson {
            error_count: self.error_count(),
            warning_count: self.warning_count(),
            report: self,
        }
    }
}

#[derive(Serialize)]
struct TextValidationReportJson<'a> {
    error_count: usize,
    warning_count: usize,
    #[serde(flatten)]
    report: &'a TextValidationReport,
}

impl fmt::Display for TextValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.issues.is_empty() {
            return writeln!(f, "Text validation passed: no issues found");
        }
        writeln!(
            f,
            "Text validation completed with {} error(s) and {} warning(s):",
            self.error_count(),
            self.warning_count()
        )?;
        for issue in &self.issues {
            writeln!(f, "  {}", issue)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TextValidationIssue {
    pub severity: TextValidationSeverity,
    pub code: TextValidationIssueCode,
    pub message: String,
    pub context: TextIssueContext,
}

impl TextValidationIssue {
    pub fn error(
        code: TextValidationIssueCode,
        message: impl Into<String>,
        context: TextIssueContext,
    ) -> Self {
        Self {
            severity: TextValidationSeverity::Error,
            code,
            message: message.into(),
            context,
        }
    }

    pub fn warning(
        code: TextValidationIssueCode,
        message: impl Into<String>,
        context: TextIssueContext,
    ) -> Self {
        Self {
            severity: TextValidationSeverity::Warning,
            code,
            message: message.into(),
            context,
        }
    }
}

impl fmt::Display for TextValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let severity = match self.severity {
            TextValidationSeverity::Error => "ERROR",
            TextValidationSeverity::Warning => "WARN ",
        };
        write!(
            f,
            "[{}] {:?} in {}: {}",
            severity, self.code, self.context, self.message
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextValidationSeverity {
    Warning,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextValidationIssueCode {
    DuplicateId,
    EmptyId,
    UnsupportedKind,
    EmptyTaskInput,
    MissingArtifact,
    UnsafeArtifactPath,
    InvalidHarborLayout,
    UnreadableTaskToml,
    MissingSweBenchRequiredField,
    MissingGoldAnswerOrReward,
    ExecutableRewardWithoutArtifacts,
    RuntimeNotRepresentable,
    EmptySplitName,
    ComplexGoldAnswer,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextIssueContext {
    Dataset,
    Example { id: String },
    Artifact { example_id: String, path: String },
    Format { name: String },
}

impl fmt::Display for TextIssueContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextIssueContext::Dataset => f.write_str("dataset"),
            TextIssueContext::Example { id } => write!(f, "example {id}"),
            TextIssueContext::Artifact { example_id, path } => {
                write!(f, "artifact {path} in example {example_id}")
            }
            TextIssueContext::Format { name } => write!(f, "format {name}"),
        }
    }
}
