//! JSON serialization for the panlabel IR format.
//!
//! This provides a simple JSON format for reading and writing datasets
//! in the panlabel IR. This is useful for:
//! - Testing the validation system before format readers exist
//! - Debugging conversions by inspecting the intermediate representation
//! - Exchanging data between panlabel instances

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use super::model::Dataset;
use crate::error::PanlabelError;

/// Reads a dataset from a JSON file in the panlabel IR format.
///
/// # Arguments
/// * `path` - Path to the JSON file
///
/// # Errors
/// Returns an error if the file cannot be read or parsed.
pub fn read_ir_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    serde_json::from_reader(reader).map_err(|source| PanlabelError::IrJsonParse {
        path: path.to_path_buf(),
        source,
    })
}

/// Writes a dataset to a JSON file in the panlabel IR format.
///
/// # Arguments
/// * `path` - Path to the output file
/// * `dataset` - The dataset to write
///
/// # Errors
/// Returns an error if the file cannot be written.
pub fn write_ir_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, dataset).map_err(|source| PanlabelError::IrJsonWrite {
        path: path.to_path_buf(),
        source,
    })
}

/// Reads a dataset from a JSON string in the panlabel IR format.
///
/// Useful for testing without file I/O.
pub fn from_json_str(json: &str) -> Result<Dataset, serde_json::Error> {
    serde_json::from_str(json)
}

/// Writes a dataset to a JSON string in the panlabel IR format.
///
/// Useful for testing without file I/O.
pub fn to_json_string(dataset: &Dataset) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Annotation, BBoxXYXY, Category, Dataset, DatasetInfo, Image, Pixel};

    fn sample_dataset() -> Dataset {
        Dataset {
            info: DatasetInfo {
                name: Some("Test Dataset".into()),
                version: Some("1.0".into()),
                description: Some("A test dataset".into()),
                url: None,
                year: None,
                contributor: None,
                date_created: None,
            },
            licenses: vec![],
            images: vec![
                Image::new(1u64, "image001.jpg", 640, 480),
                Image::new(2u64, "image002.jpg", 1920, 1080),
            ],
            categories: vec![
                Category::new(1u64, "person"),
                Category::with_supercategory(2u64, "dog", "animal"),
            ],
            annotations: vec![
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 200.0),
                ),
                Annotation::new(
                    2u64,
                    1u64,
                    2u64,
                    BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 150.0, 160.0),
                )
                .with_confidence(0.95),
            ],
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let original = sample_dataset();

        let json = to_json_string(&original).expect("serialization failed");
        let restored: Dataset = from_json_str(&json).expect("deserialization failed");

        assert_eq!(original.images.len(), restored.images.len());
        assert_eq!(original.categories.len(), restored.categories.len());
        assert_eq!(original.annotations.len(), restored.annotations.len());

        // Check specific values
        assert_eq!(restored.info.name, Some("Test Dataset".into()));
        assert_eq!(restored.images[0].file_name, "image001.jpg");
        assert_eq!(restored.categories[1].supercategory, Some("animal".into()));
        assert_eq!(restored.annotations[1].confidence, Some(0.95));
    }

    #[test]
    fn test_json_format() {
        let dataset = sample_dataset();
        let json = to_json_string(&dataset).expect("serialization failed");

        // Verify it's valid JSON and contains expected structure
        assert!(json.contains("\"images\""));
        assert!(json.contains("\"categories\""));
        assert!(json.contains("\"annotations\""));
        assert!(json.contains("\"image001.jpg\""));
    }
}
