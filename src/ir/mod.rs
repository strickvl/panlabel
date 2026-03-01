//! Intermediate Representation (IR) for panlabel.
//!
//! This module defines the canonical, format-agnostic representation of
//! object detection datasets. It serves as the central "hub" that all
//! format conversions pass through, similar to how Pandoc uses an internal
//! AST for document conversion.
//!
//! # Design Principles
//!
//! 1. **Type Safety**: Use newtypes and marker types to prevent common errors
//!    at compile time (e.g., mixing pixel and normalized coordinates).
//!
//! 2. **Canonical Format**: The IR uses a single, well-defined coordinate
//!    system (XYXY in pixel space) to avoid ambiguity.
//!
//! 3. **Permissive Construction**: IR types allow "invalid" data to be
//!    represented (e.g., negative coordinates), so that validation can
//!    report issues rather than panic during parsing.
//!
//! # Example
//!
//! ```
//! use panlabel::ir::{Dataset, Image, Category, Annotation, BBoxXYXY, Pixel};
//!
//! let dataset = Dataset {
//!     images: vec![Image::new(1u64, "image.jpg", 640, 480)],
//!     categories: vec![Category::new(1u64, "person")],
//!     annotations: vec![
//!         Annotation::new(
//!             1u64, 1u64, 1u64,
//!             BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 200.0),
//!         )
//!     ],
//!     ..Default::default()
//! };
//! ```

mod bbox;
mod coord;
mod ids;
pub mod io_coco_json;
pub mod io_json;
pub mod io_tfod_csv;
pub mod io_voc_xml;
pub mod io_yolo;
mod model;
mod space;

// Re-export core types for convenient access
pub use bbox::BBoxXYXY;
pub use coord::Coord;
pub use ids::{AnnotationId, CategoryId, ImageId, LicenseId};
pub use model::{Annotation, Category, Dataset, DatasetInfo, Image, License};
pub use space::{Normalized, Pixel};
