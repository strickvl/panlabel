//! Coordinate space marker types.
//!
//! These are zero-sized types (ZSTs) used as type parameters to distinguish
//! between different coordinate systems at compile time.

use std::fmt;

/// Marker type for pixel coordinates (absolute values).
///
/// Pixel coordinates are integers or floats representing absolute positions
/// within an image, where (0, 0) is typically the top-left corner.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pixel {}

/// Marker type for normalized coordinates (0.0 to 1.0).
///
/// Normalized coordinates represent positions as fractions of the image
/// dimensions, making them resolution-independent.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Normalized {}

impl fmt::Debug for Pixel {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {} // This is unreachable since Pixel has no variants
    }
}

impl fmt::Debug for Normalized {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {} // This is unreachable since Normalized has no variants
    }
}
