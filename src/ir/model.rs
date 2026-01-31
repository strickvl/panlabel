//! Core dataset model for the panlabel intermediate representation.
//!
//! This module defines the canonical format-agnostic representation of
//! object detection datasets. All format-specific readers convert to this
//! IR, and all writers convert from it.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::bbox::BBoxXYXY;
use super::ids::{AnnotationId, CategoryId, ImageId, LicenseId};
use super::space::Pixel;

/// A complete object detection dataset in the panlabel IR format.
///
/// This is the central data structure that all format conversions work through.
/// Think of it as the "AST" in a compiler - formats parse into this representation,
/// and this representation renders out to target formats.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Dataset {
    /// Metadata about the dataset (name, version, license, etc.)
    #[serde(default)]
    pub info: DatasetInfo,

    /// License definitions for the dataset.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub licenses: Vec<License>,

    /// All images in the dataset.
    pub images: Vec<Image>,

    /// All category definitions.
    pub categories: Vec<Category>,

    /// All annotations (bounding boxes with labels).
    pub annotations: Vec<Annotation>,
}

/// Metadata about the dataset.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Optional name of the dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Optional version string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Optional URL for more information.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Optional year the dataset was created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub year: Option<u32>,

    /// Optional contributor name or organization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contributor: Option<String>,

    /// Optional date the dataset was created (ISO 8601 or similar).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date_created: Option<String>,
}

/// A license that can be associated with images in the dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct License {
    /// Unique identifier for this license.
    pub id: LicenseId,

    /// Name of the license (e.g., "CC BY 4.0").
    pub name: String,

    /// Optional URL to the license text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

impl License {
    /// Creates a new license with the given properties.
    pub fn new(id: impl Into<LicenseId>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            url: None,
        }
    }

    /// Creates a new license with a URL.
    pub fn with_url(
        id: impl Into<LicenseId>,
        name: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            url: Some(url.into()),
        }
    }
}

/// An image in the dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Image {
    /// Unique identifier for this image.
    pub id: ImageId,

    /// Filename or path of the image.
    pub file_name: String,

    /// Width of the image in pixels.
    pub width: u32,

    /// Height of the image in pixels.
    pub height: u32,

    /// Optional license ID for this image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_id: Option<LicenseId>,

    /// Optional date the image was captured (ISO 8601 or similar).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date_captured: Option<String>,
}

impl Image {
    /// Creates a new image with the given properties.
    pub fn new(
        id: impl Into<ImageId>,
        file_name: impl Into<String>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            id: id.into(),
            file_name: file_name.into(),
            width,
            height,
            license_id: None,
            date_captured: None,
        }
    }

    /// Sets the license ID for this image.
    pub fn with_license(mut self, license_id: impl Into<LicenseId>) -> Self {
        self.license_id = Some(license_id.into());
        self
    }

    /// Sets the date captured for this image.
    pub fn with_date_captured(mut self, date: impl Into<String>) -> Self {
        self.date_captured = Some(date.into());
        self
    }
}

impl From<u64> for ImageId {
    fn from(id: u64) -> Self {
        ImageId::new(id)
    }
}

/// A category (class label) in the dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Category {
    /// Unique identifier for this category.
    pub id: CategoryId,

    /// Name of the category (e.g., "person", "car", "dog").
    pub name: String,

    /// Optional supercategory for hierarchical taxonomies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supercategory: Option<String>,
}

impl Category {
    /// Creates a new category with the given properties.
    pub fn new(id: impl Into<CategoryId>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            supercategory: None,
        }
    }

    /// Creates a new category with a supercategory.
    pub fn with_supercategory(
        id: impl Into<CategoryId>,
        name: impl Into<String>,
        supercategory: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            supercategory: Some(supercategory.into()),
        }
    }
}

impl From<u64> for CategoryId {
    fn from(id: u64) -> Self {
        CategoryId::new(id)
    }
}

/// An annotation (bounding box with label) in the dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Annotation {
    /// Unique identifier for this annotation.
    pub id: AnnotationId,

    /// ID of the image this annotation belongs to.
    pub image_id: ImageId,

    /// ID of the category (class) for this annotation.
    pub category_id: CategoryId,

    /// Bounding box in pixel coordinates (XYXY format).
    pub bbox: BBoxXYXY<Pixel>,

    /// Optional confidence score (e.g., from model predictions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,

    /// Additional attributes (e.g., "occluded", "truncated").
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, String>,
}

impl Annotation {
    /// Creates a new annotation with the minimum required fields.
    pub fn new(
        id: impl Into<AnnotationId>,
        image_id: impl Into<ImageId>,
        category_id: impl Into<CategoryId>,
        bbox: BBoxXYXY<Pixel>,
    ) -> Self {
        Self {
            id: id.into(),
            image_id: image_id.into(),
            category_id: category_id.into(),
            bbox,
            confidence: None,
            attributes: BTreeMap::new(),
        }
    }

    /// Adds a confidence score to the annotation.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Adds an attribute to the annotation.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

impl From<u64> for AnnotationId {
    fn from(id: u64) -> Self {
        AnnotationId::new(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let dataset = Dataset {
            info: DatasetInfo {
                name: Some("Test Dataset".into()),
                ..Default::default()
            },
            licenses: vec![],
            images: vec![Image::new(1u64, "image001.jpg", 640, 480)],
            categories: vec![Category::new(1u64, "person")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::from_xyxy(10.0, 20.0, 100.0, 200.0),
            )],
        };

        assert_eq!(dataset.images.len(), 1);
        assert_eq!(dataset.categories.len(), 1);
        assert_eq!(dataset.annotations.len(), 1);
    }

    #[test]
    fn test_annotation_builder_pattern() {
        let annotation =
            Annotation::new(1u64, 1u64, 1u64, BBoxXYXY::from_xyxy(0.0, 0.0, 50.0, 50.0))
                .with_confidence(0.95)
                .with_attribute("occluded", "false")
                .with_attribute("truncated", "true");

        assert_eq!(annotation.confidence, Some(0.95));
        assert_eq!(annotation.attributes.len(), 2);
        assert_eq!(
            annotation.attributes.get("occluded"),
            Some(&"false".to_string())
        );
    }
}
