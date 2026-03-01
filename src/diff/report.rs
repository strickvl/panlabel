//! Diff report types and text formatting.

use serde::Serialize;
use std::fmt;

/// Dataset diff report.
#[derive(Clone, Debug, Default, Serialize)]
pub struct DiffReport {
    /// Image-level counts.
    pub images: DiffCounts,
    /// Category-level counts.
    pub categories: DiffCounts,
    /// Annotation-level counts.
    pub annotations: DiffAnnotationCounts,
    /// Optional detail section.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<DiffDetail>,
}

/// Shared / only-in-A / only-in-B counts.
#[derive(Clone, Debug, Default, Serialize)]
pub struct DiffCounts {
    pub shared: usize,
    pub only_in_a: usize,
    pub only_in_b: usize,
}

/// Annotation diff counts.
#[derive(Clone, Debug, Default, Serialize)]
pub struct DiffAnnotationCounts {
    pub shared: usize,
    pub only_in_a: usize,
    pub only_in_b: usize,
    pub modified: usize,
}

/// Optional detail section for text/json output.
#[derive(Clone, Debug, Default, Serialize)]
pub struct DiffDetail {
    pub images_only_in_a: Vec<String>,
    pub images_only_in_b: Vec<String>,
    pub modified_annotations: Vec<ModifiedAnnotation>,
    pub max_items: usize,
}

/// One modified annotation item.
#[derive(Clone, Debug, Serialize)]
pub struct ModifiedAnnotation {
    pub file_name: String,
    pub annotation_id: u64,
    pub reason: String,
}

impl fmt::Display for DiffReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Images:      {} shared, {} only in A, {} only in B",
            self.images.shared, self.images.only_in_a, self.images.only_in_b
        )?;
        writeln!(
            f,
            "Categories:  {} shared, {} only in A, {} only in B",
            self.categories.shared, self.categories.only_in_a, self.categories.only_in_b
        )?;
        writeln!(
            f,
            "Annotations: {} shared, {} only in A, {} only in B",
            self.annotations.shared, self.annotations.only_in_a, self.annotations.only_in_b
        )?;
        writeln!(f, "             modified ({})", self.annotations.modified)?;

        if let Some(detail) = &self.detail {
            writeln!(f)?;
            writeln!(f, "Images only in A:")?;
            if detail.images_only_in_a.is_empty() {
                writeln!(f, "  - (none)")?;
            } else {
                for name in &detail.images_only_in_a {
                    writeln!(f, "  - {name}")?;
                }
            }

            writeln!(f)?;
            writeln!(f, "Images only in B:")?;
            if detail.images_only_in_b.is_empty() {
                writeln!(f, "  - (none)")?;
            } else {
                for name in &detail.images_only_in_b {
                    writeln!(f, "  - {name}")?;
                }
            }

            writeln!(f)?;
            writeln!(
                f,
                "Annotations modified (showing first {}):",
                detail.max_items
            )?;
            if detail.modified_annotations.is_empty() {
                writeln!(f, "  - (none)")?;
            } else {
                for item in &detail.modified_annotations {
                    writeln!(
                        f,
                        "  - {} ann#{}: {}",
                        item.file_name, item.annotation_id, item.reason
                    )?;
                }
            }
        }

        Ok(())
    }
}
