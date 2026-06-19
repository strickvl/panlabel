use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::PanlabelError;

use super::artifact_refs_for_example;
use super::io_common::{
    object_to_metadata, read_rows, scalar_to_string, task_input_to_json, value_to_task_input,
};
use super::model::{
    ArtifactKind, ArtifactRef, HarnessDescriptor, Message, Metadata, RewardDescriptor, RewardKind,
    Role, RuntimeDescriptor, TaskInput, TextDataset, TextExample,
};

#[derive(Clone, Debug, Default)]
pub struct VerifiersTasksetReadOptions {
    pub split: Option<String>,
}

pub fn read_verifiers_taskset(
    path: &Path,
    options: &VerifiersTasksetReadOptions,
) -> Result<TextDataset, PanlabelError> {
    let rows = read_rows(path)?;
    let mut dataset = TextDataset::default();
    dataset.info.name = Some("verifiers-taskset".to_string());

    let source_root = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };
    let rubric = source_root
        .join("rubric.py")
        .exists()
        .then_some(ArtifactRef {
            path: "rubric.py".to_string(),
            kind: Some(ArtifactKind::Script),
            inline: None,
            media_type: Some("text/x-python".to_string()),
        });
    let env = ["env.py", "environment.py"]
        .iter()
        .find(|name| source_root.join(name).exists())
        .map(|name| ArtifactRef {
            path: (*name).to_string(),
            kind: Some(ArtifactKind::Environment),
            inline: None,
            media_type: Some("text/x-python".to_string()),
        });

    for (idx, row) in rows.into_iter().enumerate() {
        let object = row
            .value
            .as_object()
            .ok_or_else(|| PanlabelError::TextIrReadError {
                path: path.to_path_buf(),
                message: "Verifiers taskset rows must be JSON objects".to_string(),
            })?;
        let input = verifiers_input(object)?;
        let id = object
            .get("id")
            .or_else(|| object.get("task_id"))
            .and_then(scalar_to_string)
            .unwrap_or_else(|| format!("row-{:06}", idx + 1));
        let mut example = TextExample::task(id, input);
        example.split = object
            .get("split")
            .and_then(scalar_to_string)
            .or_else(|| options.split.clone())
            .or(row.split);
        if let Some(answer) = object.get("answer").or_else(|| object.get("ground_truth")) {
            example.data.gold_answer = Some(answer.clone());
            example.data.reward = Some(RewardDescriptor {
                kind: if rubric.is_some() {
                    RewardKind::PythonRubric
                } else {
                    RewardKind::AnswerMatch
                },
                artifacts: rubric.clone().into_iter().collect(),
                metadata: Metadata::new(),
            });
        } else if let Some(rubric) = rubric.clone() {
            example.data.reward = Some(RewardDescriptor {
                kind: RewardKind::PythonRubric,
                artifacts: vec![rubric],
                metadata: Metadata::new(),
            });
        }
        if let Some(env) = env.clone() {
            example.data.runtime = Some(RuntimeDescriptor {
                artifacts: vec![env],
                ..RuntimeDescriptor::default()
            });
        }
        if object.contains_key("max_turns")
            || object.contains_key("toolsets")
            || object.contains_key("sandbox")
        {
            let mut metadata = Metadata::new();
            for key in ["max_turns", "toolsets", "sandbox", "program"] {
                if let Some(value) = object.get(key) {
                    metadata.insert(key.to_string(), value.clone());
                }
            }
            example.data.harness = Some(HarnessDescriptor {
                name: Some("verifiers".to_string()),
                artifacts: Vec::new(),
                metadata,
            });
        }
        let mut metadata = object_to_metadata(object);
        for key in [
            "prompt",
            "question",
            "answer",
            "ground_truth",
            "id",
            "task_id",
            "split",
            "system_prompt",
        ] {
            metadata.remove(key);
        }
        example.metadata = metadata;
        example.provenance.insert(
            "source_format".to_string(),
            Value::String("verifiers-taskset".to_string()),
        );
        dataset.examples.push(example);
    }

    Ok(dataset)
}

pub fn write_verifiers_taskset(path: &Path, dataset: &TextDataset) -> Result<(), PanlabelError> {
    let output_file = if path.extension().is_none() {
        fs::create_dir_all(path).map_err(PanlabelError::Io)?;
        path.join("dataset.jsonl")
    } else {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
        }
        path.to_path_buf()
    };
    let file = File::create(&output_file).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    for example in &dataset.examples {
        let mut row = Map::new();
        row.insert("id".to_string(), Value::String(example.id.clone()));
        match task_input_to_json(&example.data.input) {
            ("messages", value) => {
                row.insert("prompt".to_string(), value);
            }
            (_, value) => {
                row.insert("question".to_string(), value);
            }
        }
        if let Some(answer) = &example.data.gold_answer {
            row.insert("answer".to_string(), answer.clone());
        }
        if let Some(split) = &example.split {
            row.insert("split".to_string(), Value::String(split.clone()));
        }
        let artifacts = artifact_refs_for_example(example);
        if !artifacts.is_empty() {
            row.insert(
                "artifacts".to_string(),
                serde_json::to_value(artifacts).map_err(|source| {
                    PanlabelError::TextIrJsonWrite {
                        path: output_file.clone(),
                        source,
                    }
                })?,
            );
        }
        if !example.metadata.is_empty() {
            row.insert(
                "info".to_string(),
                Value::Object(example.metadata.clone().into_iter().collect()),
            );
        }
        serde_json::to_writer(&mut writer, &Value::Object(row)).map_err(|source| {
            PanlabelError::TextIrJsonWrite {
                path: output_file.clone(),
                source,
            }
        })?;
        writer.write_all(b"\n").map_err(PanlabelError::Io)?;
    }
    writer.flush().map_err(PanlabelError::Io)
}

fn verifiers_input(object: &Map<String, Value>) -> Result<TaskInput, PanlabelError> {
    let input = if let Some(prompt) = object.get("prompt") {
        value_to_task_input(prompt)?
    } else if let Some(question) = object.get("question") {
        value_to_task_input(question)?
    } else {
        return Err(PanlabelError::UnsupportedFormat(
            "Verifiers rows require a prompt or question field".to_string(),
        ));
    };

    let Some(system_prompt) = object.get("system_prompt").and_then(Value::as_str) else {
        return Ok(input);
    };
    let system = Message {
        role: Role::System,
        content: vec![super::model::Block::Text {
            text: system_prompt.to_string(),
        }],
    };
    match input {
        TaskInput::Messages { mut messages } => {
            messages.insert(0, system);
            Ok(TaskInput::Messages { messages })
        }
        TaskInput::Text { text } => Ok(TaskInput::Messages {
            messages: vec![
                system,
                Message {
                    role: Role::User,
                    content: vec![super::model::Block::Text { text }],
                },
            ],
        }),
    }
}
