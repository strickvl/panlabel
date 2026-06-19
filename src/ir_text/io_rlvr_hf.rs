use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::PanlabelError;

use super::artifact_refs_for_example;
use super::io_common::{
    object_to_metadata, read_rows, scalar_to_string, task_input_to_json, value_to_task_input,
    RowWithContext, ANSWER_COLUMNS, ID_COLUMNS, PROMPT_COLUMNS,
};
use super::model::{Metadata, RewardDescriptor, RewardKind, TextDataset, TextExample};

#[derive(Clone, Debug, Default)]
pub struct RlvrHfReadOptions {
    pub prompt_column: Option<String>,
    pub answer_column: Option<String>,
    pub id_column: Option<String>,
    pub split_column: Option<String>,
    pub split: Option<String>,
}

pub fn read_rlvr_hf(
    path: &Path,
    options: &RlvrHfReadOptions,
) -> Result<TextDataset, PanlabelError> {
    let rows = read_rows(path)?;
    let mut dataset = TextDataset::default();
    dataset.info.name = Some("rlvr-hf".to_string());

    for (idx, row) in rows.into_iter().enumerate() {
        let RowWithContext { value, split } = row;
        let object = value
            .as_object()
            .ok_or_else(|| PanlabelError::TextIrReadError {
                path: path.to_path_buf(),
                message: "RLVR-HF rows must be JSON objects".to_string(),
            })?;
        let prompt_mapping = find_prompt_mapping(object, options.prompt_column.as_deref(), path)?;
        let answer_mapping = find_column(
            object,
            ANSWER_COLUMNS,
            options.answer_column.as_deref(),
            "answer",
            path,
        )?;
        let id_mapping = find_column(object, ID_COLUMNS, options.id_column.as_deref(), "id", path)?;
        let split_mapping = options.split_column.as_deref().and_then(|column| {
            object
                .get(column)
                .map(|value| (column.to_string(), value.clone()))
        });

        let input = value_to_task_input(&prompt_mapping.value)?;
        let id = id_mapping
            .as_ref()
            .and_then(|mapping| scalar_to_string(&mapping.value))
            .unwrap_or_else(|| format!("row-{:06}", idx + 1));

        let mut example = TextExample::task(id, input);
        example.split = split_mapping
            .as_ref()
            .and_then(|(_, value)| scalar_to_string(value))
            .or_else(|| options.split.clone())
            .or(split);
        if let Some(answer) = answer_mapping.as_ref() {
            example.data.gold_answer = Some(answer.value.clone());
            example.data.reward = Some(RewardDescriptor {
                kind: RewardKind::AnswerMatch,
                artifacts: Vec::new(),
                metadata: Metadata::new(),
            });
        }

        let mut metadata = object_to_metadata(object);
        metadata.remove(&prompt_mapping.column);
        if let Some(answer) = answer_mapping.as_ref() {
            metadata.remove(&answer.column);
        }
        if let Some(id_mapping) = id_mapping.as_ref() {
            metadata.remove(&id_mapping.column);
        }
        if let Some((column, _)) = split_mapping.as_ref() {
            metadata.remove(column);
        }
        example.metadata = metadata;
        example.provenance.insert(
            "source_format".to_string(),
            Value::String("rlvr-hf".to_string()),
        );
        example.provenance.insert(
            "prompt_column".to_string(),
            Value::String(prompt_mapping.column.clone()),
        );
        if let Some(answer) = answer_mapping {
            example
                .provenance
                .insert("answer_column".to_string(), Value::String(answer.column));
        }
        dataset.examples.push(example);
    }

    Ok(dataset)
}

pub fn write_rlvr_hf(path: &Path, dataset: &TextDataset) -> Result<(), PanlabelError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    for example in &dataset.examples {
        let mut row = Map::new();
        row.insert("id".to_string(), Value::String(example.id.clone()));
        let (column, value) = task_input_to_json(&example.data.input);
        row.insert(column.to_string(), value);
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
                        path: path.to_path_buf(),
                        source,
                    }
                })?,
            );
        }
        if !example.metadata.is_empty() {
            row.insert(
                "metadata".to_string(),
                Value::Object(example.metadata.clone().into_iter().collect()),
            );
        }
        serde_json::to_writer(&mut writer, &Value::Object(row)).map_err(|source| {
            PanlabelError::TextIrJsonWrite {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writer.write_all(b"\n").map_err(PanlabelError::Io)?;
    }
    writer.flush().map_err(PanlabelError::Io)
}

struct ColumnMapping {
    column: String,
    value: Value,
}

fn find_prompt_mapping(
    object: &Map<String, Value>,
    explicit: Option<&str>,
    path: &Path,
) -> Result<ColumnMapping, PanlabelError> {
    if let Some(column) = explicit {
        let value = object
            .get(column)
            .ok_or_else(|| PanlabelError::TextIrReadError {
                path: path.to_path_buf(),
                message: format!("explicit prompt column '{column}' was not found"),
            })?;
        return Ok(ColumnMapping {
            column: column.to_string(),
            value: value.clone(),
        });
    }

    if let Some(Value::Object(params)) = object.get("responses_create_params") {
        if let Some(value) = params.get("input") {
            return Ok(ColumnMapping {
                column: "responses_create_params.input".to_string(),
                value: value.clone(),
            });
        }
    }

    let matches = PROMPT_COLUMNS
        .iter()
        .filter_map(|column| object.get(*column).map(|value| (*column, value.clone())))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [(column, value)] => Ok(ColumnMapping {
            column: (*column).to_string(),
            value: value.clone(),
        }),
        [] => Err(PanlabelError::TextIrReadError {
            path: path.to_path_buf(),
            message: "could not detect a prompt/input column; pass --prompt-column".to_string(),
        }),
        _ => Err(PanlabelError::TextIrReadError {
            path: path.to_path_buf(),
            message: format!(
                "multiple plausible prompt/input columns found ({}) — pass --prompt-column",
                matches
                    .iter()
                    .map(|(column, _)| *column)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }),
    }
}

fn find_column(
    object: &Map<String, Value>,
    candidates: &[&str],
    explicit: Option<&str>,
    label: &str,
    path: &Path,
) -> Result<Option<ColumnMapping>, PanlabelError> {
    if let Some(column) = explicit {
        let value = object
            .get(column)
            .ok_or_else(|| PanlabelError::TextIrReadError {
                path: path.to_path_buf(),
                message: format!("explicit {label} column '{column}' was not found"),
            })?;
        return Ok(Some(ColumnMapping {
            column: column.to_string(),
            value: value.clone(),
        }));
    }
    let matches = candidates
        .iter()
        .filter_map(|column| object.get(*column).map(|value| (*column, value.clone())))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [(column, value)] => Ok(Some(ColumnMapping {
            column: (*column).to_string(),
            value: value.clone(),
        })),
        _ => Err(PanlabelError::TextIrReadError {
            path: path.to_path_buf(),
            message: format!(
                "multiple plausible {label} columns found ({}) — pass --{label}-column",
                matches
                    .iter()
                    .map(|(column, _)| *column)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }),
    }
}
