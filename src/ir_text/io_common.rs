use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::PanlabelError;

use super::model::{Block, Message, Metadata, Role, TaskInput};

pub(crate) const PROMPT_COLUMNS: &[&str] = &[
    "messages",
    "prompt",
    "question",
    "problem",
    "problem_statement",
    "instruction",
    "input",
];
pub(crate) const ANSWER_COLUMNS: &[&str] = &[
    "answer",
    "ground_truth",
    "gold_answer",
    "solution",
    "expected_answer",
    "final_answer",
    "reference_answer",
    "expected_action",
];
pub(crate) const ID_COLUMNS: &[&str] = &["id", "instance_id", "task_id", "trajectory_id"];
pub(crate) const SWE_BENCH_REQUIRED_FIELDS: &[&str] = &[
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "test_patch",
];

#[derive(Clone, Debug)]
pub(crate) struct RowWithContext {
    pub value: Value,
    pub split: Option<String>,
}

pub(crate) fn read_rows(path: &Path) -> Result<Vec<RowWithContext>, PanlabelError> {
    if path.is_dir() {
        let mut rows = Vec::new();
        for entry in fs::read_dir(path).map_err(PanlabelError::Io)? {
            let entry = entry.map_err(PanlabelError::Io)?;
            if !entry.file_type().map_err(PanlabelError::Io)?.is_file() {
                continue;
            }
            let entry_path = entry.path();
            let Some(ext) = entry_path.extension().and_then(|ext| ext.to_str()) else {
                continue;
            };
            if !matches!(ext, "jsonl" | "ndjson" | "json") {
                continue;
            }
            let split = entry_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string);
            rows.extend(read_rows_from_file(&entry_path, split)?);
        }
        rows.sort_by(|a, b| a.split.cmp(&b.split));
        return Ok(rows);
    }
    read_rows_from_file(path, None)
}

fn read_rows_from_file(
    path: &Path,
    split: Option<String>,
) -> Result<Vec<RowWithContext>, PanlabelError> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("jsonl") | Some("ndjson") => read_jsonl_rows(path, split),
        Some("json") => read_json_rows(path, split),
        _ => Err(PanlabelError::UnsupportedFormat(format!(
            "unsupported text row file extension for {}",
            path.display()
        ))),
    }
}

fn read_jsonl_rows(
    path: &Path,
    split: Option<String>,
) -> Result<Vec<RowWithContext>, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(PanlabelError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value =
            serde_json::from_str(trimmed).map_err(|source| PanlabelError::TextIrJsonlParse {
                path: path.to_path_buf(),
                line: idx + 1,
                message: source.to_string(),
            })?;
        rows.push(RowWithContext {
            value,
            split: split.clone(),
        });
    }
    Ok(rows)
}

fn read_json_rows(
    path: &Path,
    split: Option<String>,
) -> Result<Vec<RowWithContext>, PanlabelError> {
    let mut file = File::open(path).map_err(PanlabelError::Io)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_str(&buf).map_err(|source| PanlabelError::TextIrJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let values = match value {
        Value::Array(values) => values,
        Value::Object(mut object) => {
            match object.remove("examples").or_else(|| object.remove("rows")) {
                Some(Value::Array(values)) => values,
                _ => vec![Value::Object(object)],
            }
        }
        other => vec![other],
    };
    Ok(values
        .into_iter()
        .map(|value| RowWithContext {
            value,
            split: split.clone(),
        })
        .collect())
}

pub(crate) fn value_to_task_input(value: &Value) -> Result<TaskInput, PanlabelError> {
    match value {
        Value::String(text) => Ok(TaskInput::Text { text: text.clone() }),
        Value::Array(values) => Ok(TaskInput::Messages {
            messages: values
                .iter()
                .map(value_to_message)
                .collect::<Result<Vec<_>, _>>()?,
        }),
        Value::Object(object) => {
            if let Some(Value::Array(values)) =
                object.get("messages").or_else(|| object.get("input"))
            {
                return Ok(TaskInput::Messages {
                    messages: values
                        .iter()
                        .map(value_to_message)
                        .collect::<Result<Vec<_>, _>>()?,
                });
            }
            Ok(TaskInput::Text {
                text: value.to_string(),
            })
        }
        other => Ok(TaskInput::Text {
            text: other.to_string(),
        }),
    }
}

fn value_to_message(value: &Value) -> Result<Message, PanlabelError> {
    let object = value.as_object().ok_or_else(|| {
        PanlabelError::UnsupportedFormat("message entries must be JSON objects".to_string())
    })?;
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .map(role_from_str)
        .unwrap_or(Role::User);
    let content_value = object
        .get("content")
        .cloned()
        .unwrap_or(Value::String(String::new()));
    let content = match content_value {
        Value::String(text) => vec![Block::Text { text }],
        Value::Array(values) => values
            .into_iter()
            .map(|value| match value {
                Value::String(text) => Block::Text { text },
                Value::Object(object) => object
                    .get("text")
                    .and_then(Value::as_str)
                    .map(|text| Block::Text {
                        text: text.to_string(),
                    })
                    .unwrap_or_else(|| Block::Text {
                        text: Value::Object(object).to_string(),
                    }),
                other => Block::Text {
                    text: other.to_string(),
                },
            })
            .collect(),
        other => vec![Block::Text {
            text: other.to_string(),
        }],
    };
    Ok(Message { role, content })
}

fn role_from_str(value: &str) -> Role {
    match value.to_ascii_lowercase().as_str() {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => Role::Other(other.to_string()),
    }
}

fn role_to_str(role: &Role) -> String {
    match role {
        Role::System => "system".to_string(),
        Role::User => "user".to_string(),
        Role::Assistant => "assistant".to_string(),
        Role::Tool => "tool".to_string(),
        Role::Other(value) => value.clone(),
    }
}

pub(crate) fn task_input_to_json(input: &TaskInput) -> (&'static str, Value) {
    match input {
        TaskInput::Text { text } => ("prompt", Value::String(text.clone())),
        TaskInput::Messages { messages } => {
            let values = messages
                .iter()
                .map(|message| {
                    let content = message
                        .content
                        .iter()
                        .map(|block| match block {
                            Block::Text { text } => text.clone(),
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    serde_json::json!({"role": role_to_str(&message.role), "content": content})
                })
                .collect();
            ("messages", Value::Array(values))
        }
    }
}

pub(crate) fn scalar_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

pub(crate) fn object_to_metadata(object: &Map<String, Value>) -> Metadata {
    object
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>()
}
