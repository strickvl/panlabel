use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::PanlabelError;

use super::artifact::artifact_refs_for_example;
use super::model::{TextDataset, TextExample, TextExampleKind};

#[derive(Clone, Debug, Default)]
pub struct TextIrJsonlWriteOptions {
    pub pretty_meta: bool,
}

pub fn read_text_ir_jsonl(path: &Path) -> Result<TextDataset, PanlabelError> {
    let layout = TextIrJsonlLayout::from_input(path);
    let examples_path = layout.examples_path();
    let file = File::open(&examples_path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    let mut examples = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(PanlabelError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let example: TextExample =
            serde_json::from_str(trimmed).map_err(|source| PanlabelError::TextIrJsonlParse {
                path: examples_path.clone(),
                line: idx + 1,
                message: source.to_string(),
            })?;
        if example.kind != TextExampleKind::Task {
            return Err(PanlabelError::TextIrJsonlParse {
                path: examples_path.clone(),
                line: idx + 1,
                message: "only kind=\"task\" examples are supported in this implementation pass"
                    .to_string(),
            });
        }
        examples.push(example);
    }

    let mut dataset = if layout.meta_path().exists() {
        let file = File::open(layout.meta_path()).map_err(PanlabelError::Io)?;
        serde_json::from_reader(file).map_err(|source| PanlabelError::TextIrJsonParse {
            path: layout.meta_path(),
            source,
        })?
    } else {
        TextDataset::default()
    };
    dataset.examples = examples;
    Ok(dataset)
}

pub fn write_text_ir_jsonl(path: &Path, dataset: &TextDataset) -> Result<(), PanlabelError> {
    write_text_ir_jsonl_with_options(path, dataset, &TextIrJsonlWriteOptions::default())
}

pub fn write_text_ir_jsonl_with_options(
    path: &Path,
    dataset: &TextDataset,
    options: &TextIrJsonlWriteOptions,
) -> Result<(), PanlabelError> {
    let layout = TextIrJsonlLayout::from_output(path);
    if let Some(parent) = layout.examples_path().parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }

    let file = File::create(layout.examples_path()).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    for example in &dataset.examples {
        if example.kind != TextExampleKind::Task {
            return Err(PanlabelError::TextIrWriteError {
                path: layout.examples_path(),
                message: "only kind=\"task\" examples can be written in this implementation pass"
                    .to_string(),
            });
        }
        serde_json::to_writer(&mut writer, example).map_err(|source| {
            PanlabelError::TextIrJsonWrite {
                path: layout.examples_path(),
                source,
            }
        })?;
        writer.write_all(b"\n").map_err(PanlabelError::Io)?;
    }
    writer.flush().map_err(PanlabelError::Io)?;

    let meta_dataset = TextDataset {
        info: dataset.info.clone(),
        examples: Vec::new(),
        metadata: dataset.metadata.clone(),
    };
    let meta_file = File::create(layout.meta_path()).map_err(PanlabelError::Io)?;
    if options.pretty_meta {
        serde_json::to_writer_pretty(meta_file, &meta_dataset)
    } else {
        serde_json::to_writer(meta_file, &meta_dataset)
    }
    .map_err(|source| PanlabelError::TextIrJsonWrite {
        path: layout.meta_path(),
        source,
    })?;

    if dataset
        .examples
        .iter()
        .any(|example| !artifact_refs_for_example(example).is_empty())
    {
        fs::create_dir_all(layout.artifacts_dir()).map_err(PanlabelError::Io)?;
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct TextIrJsonlLayout {
    path: PathBuf,
    is_dir: bool,
}

impl TextIrJsonlLayout {
    fn from_input(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            is_dir: path.is_dir(),
        }
    }

    fn from_output(path: &Path) -> Self {
        let is_dir = path.extension().is_none();
        Self {
            path: path.to_path_buf(),
            is_dir,
        }
    }

    fn examples_path(&self) -> PathBuf {
        if self.is_dir {
            self.path.join("tasks.jsonl")
        } else {
            self.path.clone()
        }
    }

    fn meta_path(&self) -> PathBuf {
        if self.is_dir {
            return self.path.join("tasks.meta.json");
        }
        sibling_with_suffix(&self.path, "meta.json")
    }

    fn artifacts_dir(&self) -> PathBuf {
        if self.is_dir {
            return self.path.join("tasks.artifacts");
        }
        sibling_with_suffix(&self.path, "artifacts")
    }
}

fn sibling_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("tasks");
    let file_name = format!("{stem}.{suffix}");
    path.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_text::model::{ArtifactRef, TaskInput, TextDatasetInfo, TextExample};

    #[test]
    fn roundtrips_task_examples_and_meta() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tasks.jsonl");
        let mut dataset = TextDataset {
            info: TextDatasetInfo {
                name: Some("demo".to_string()),
                version: Some("1".to_string()),
                description: None,
            },
            examples: vec![TextExample::task(
                "task-1",
                TaskInput::Text {
                    text: "Answer carefully".to_string(),
                },
            )],
            metadata: Default::default(),
        };
        dataset.examples[0].split = Some("train".to_string());
        dataset.examples[0].data.artifacts.push(ArtifactRef {
            path: "tests/test.sh".to_string(),
            kind: None,
            inline: None,
            media_type: None,
        });

        write_text_ir_jsonl(&path, &dataset).expect("write text ir");
        let reread = read_text_ir_jsonl(&path).expect("read text ir");

        assert_eq!(reread.info.name.as_deref(), Some("demo"));
        assert_eq!(reread.examples.len(), 1);
        assert_eq!(reread.examples[0].id, "task-1");
        assert!(dir.path().join("tasks.artifacts").exists());
    }
}
