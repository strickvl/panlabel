use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use serde_json::Value;

use crate::error::PanlabelError;
use crate::ir_text::io_common::{ANSWER_COLUMNS, PROMPT_COLUMNS, SWE_BENCH_REQUIRED_FIELDS};
use crate::text_format_catalog::TextFormat;

const MAX_JSONL_ROWS: usize = 16;
const MAX_JSON_BYTES: usize = 256 * 1024;

pub fn detect_text_format(path: &Path) -> Result<TextFormat, PanlabelError> {
    if path.is_dir() {
        return detect_text_dir_format(path);
    }
    if !path.exists() {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "file does not exist".to_string(),
        });
    }
    match path.extension().and_then(|ext| ext.to_str()).map(str::to_ascii_lowercase) {
        Some(ext) if ext == "jsonl" || ext == "ndjson" => detect_jsonl_text_format(path),
        Some(ext) if ext == "json" => detect_json_text_format(path),
        _ => Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "unrecognized text dataset extension (expected .jsonl, .ndjson, or .json). Use --from to specify format explicitly.".to_string(),
        }),
    }
}

fn detect_text_dir_format(path: &Path) -> Result<TextFormat, PanlabelError> {
    if path.join("tasks.jsonl").exists() {
        return Ok(TextFormat::TextIrJsonl);
    }
    if is_harbor_task_dir(path) {
        return Ok(TextFormat::Harbor);
    }

    let mut harbor_children = 0;
    for entry in fs::read_dir(path).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        if entry.file_type().map_err(PanlabelError::Io)?.is_dir()
            && is_harbor_task_dir(&entry.path())
        {
            harbor_children += 1;
        }
    }
    if harbor_children > 0 {
        return Ok(TextFormat::Harbor);
    }

    for name in ["dataset.jsonl", "train.jsonl", "eval.jsonl", "test.jsonl"] {
        let candidate = path.join(name);
        if candidate.exists() {
            return detect_jsonl_text_format(&candidate);
        }
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "unrecognized text/task directory layout. Expected tasks.jsonl, Harbor task.toml plus instruction.md, or JSONL split files. Use --from to specify format explicitly.".to_string(),
    })
}

fn is_harbor_task_dir(path: &Path) -> bool {
    path.join("task.toml").exists() && path.join("instruction.md").exists()
}

fn detect_jsonl_text_format(path: &Path) -> Result<TextFormat, PanlabelError> {
    let rows = sample_jsonl_rows(path)?;
    detect_from_rows(path, &rows)
}

fn detect_json_text_format(path: &Path) -> Result<TextFormat, PanlabelError> {
    let mut file = File::open(path).map_err(PanlabelError::Io)?;
    let mut buf = Vec::new();
    file.by_ref()
        .take(MAX_JSON_BYTES as u64)
        .read_to_end(&mut buf)
        .map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_slice(&buf).map_err(|source| PanlabelError::FormatDetectionJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let rows = match value {
        Value::Array(values) => values,
        Value::Object(mut object) => {
            match object.remove("examples").or_else(|| object.remove("rows")) {
                Some(Value::Array(values)) => values,
                _ => vec![Value::Object(object)],
            }
        }
        other => vec![other],
    };
    detect_from_rows(path, &rows)
}

fn sample_jsonl_rows(path: &Path) -> Result<Vec<Value>, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let mut bytes_read = 0usize;
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(PanlabelError::Io)?;
        bytes_read += line.len();
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: Value =
            serde_json::from_str(trimmed).map_err(|source| PanlabelError::TextIrJsonlParse {
                path: path.to_path_buf(),
                line: idx + 1,
                message: source.to_string(),
            })?;
        rows.push(value);
        if rows.len() >= MAX_JSONL_ROWS || bytes_read >= MAX_JSON_BYTES {
            break;
        }
    }
    Ok(rows)
}

fn detect_from_rows(path: &Path, rows: &[Value]) -> Result<TextFormat, PanlabelError> {
    if rows.is_empty() {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "no non-empty JSON rows found".to_string(),
        });
    }

    let has_text_ir = rows.iter().all(is_text_ir_row);
    if has_text_ir {
        return Ok(TextFormat::TextIrJsonl);
    }

    if rows.iter().any(is_swe_bench_row) {
        return Ok(TextFormat::SweBench);
    }
    if rows.iter().any(is_verifiers_row) {
        return Ok(TextFormat::VerifiersTaskset);
    }
    if rows.iter().any(is_rlvr_hf_row) {
        return Ok(TextFormat::RlvrHf);
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "JSON rows do not match text-ir-jsonl, SWE-bench, Verifiers, or generic RLVR-HF task shapes. Use --from to specify format explicitly.".to_string(),
    })
}

fn is_text_ir_row(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.get("kind").and_then(Value::as_str) == Some("task") && object.contains_key("data")
}

fn is_swe_bench_row(value: &Value) -> bool {
    has_all_fields(value, SWE_BENCH_REQUIRED_FIELDS)
}

fn is_verifiers_row(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.contains_key("system_prompt")
        || object.contains_key("rubric")
        || object.contains_key("rubric_path")
        || object.contains_key("env_path")
        || object.contains_key("max_turns")
        || object.contains_key("toolsets")
        || object.contains_key("sandbox")
}

fn is_rlvr_hf_row(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let has_prompt = PROMPT_COLUMNS
        .iter()
        .any(|field| object.contains_key(*field));
    let has_answer = ANSWER_COLUMNS
        .iter()
        .any(|field| object.contains_key(*field));
    let has_nested_chat_input = object
        .get("responses_create_params")
        .and_then(Value::as_object)
        .map(|params| params.contains_key("input"))
        .unwrap_or(false);
    has_prompt || has_answer || has_nested_chat_input
}

fn has_all_fields(value: &Value, fields: &[&str]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    fields.iter().all(|field| object.contains_key(*field))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swe_bench_wins_over_generic_prompt_shape() {
        let row = serde_json::json!({
            "instance_id": "x",
            "repo": "owner/repo",
            "base_commit": "abc",
            "problem_statement": "Fix it",
            "test_patch": "diff --git ...",
            "answer": "ignored for detection"
        });
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rows.jsonl");
        fs::write(&path, format!("{}\n", row)).expect("write rows");
        assert_eq!(detect_text_format(&path).unwrap(), TextFormat::SweBench);
    }

    #[test]
    fn plain_prompt_answer_is_rlvr_not_verifiers() {
        let row = serde_json::json!({"prompt": "2+2?", "answer": "4"});
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rows.jsonl");
        fs::write(&path, format!("{}\n", row)).expect("write rows");
        assert_eq!(detect_text_format(&path).unwrap(), TextFormat::RlvrHf);
    }
}
