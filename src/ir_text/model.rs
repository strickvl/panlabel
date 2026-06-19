use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type Metadata = BTreeMap<String, Value>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TextDataset {
    #[serde(default)]
    pub info: TextDatasetInfo,
    #[serde(default)]
    pub examples: Vec<TextExample>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDatasetInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextExample {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
    pub kind: TextExampleKind,
    pub data: TaskData,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub provenance: Metadata,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextExampleKind {
    Task,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TaskData {
    pub input: TaskInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gold_answer: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<TaskStep>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<RewardDescriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<RuntimeDescriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness: Option<HarnessDescriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solution: Option<SolutionDescriptor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ArtifactRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskInput {
    Text { text: String },
    Messages { messages: Vec<Message> },
}

impl Default for TaskInput {
    fn default() -> Self {
        Self::Text {
            text: String::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<Block>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text { text: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RewardDescriptor {
    pub kind: RewardKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardKind {
    AnswerMatch,
    ShellVerifier,
    PythonRubric,
    UnitTests,
    LlmJudge,
    ExternalHarness,
    MetadataOnly,
    Unknown(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RuntimeDescriptor {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HarnessDescriptor {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolutionDescriptor {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TaskStep {
    pub id: String,
    pub input: TaskInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_output: Option<Value>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<ArtifactKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Script,
    Dockerfile,
    Patch,
    Test,
    Solution,
    Environment,
    Metadata,
    Other(String),
}

impl TextExample {
    pub fn task(id: impl Into<String>, input: TaskInput) -> Self {
        Self {
            id: id.into(),
            split: None,
            kind: TextExampleKind::Task,
            data: TaskData {
                input,
                ..TaskData::default()
            },
            provenance: Metadata::new(),
            metadata: Metadata::new(),
        }
    }
}
