//! Newtype IDs for type-safe identification of dataset elements.
//!
//! Using newtypes prevents accidentally mixing up different kinds of IDs
//! (e.g., passing an image ID where an annotation ID is expected).

use serde::{Deserialize, Serialize};
use std::fmt;

/// A unique identifier for an image in the dataset.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ImageId(pub u64);

impl ImageId {
    /// Creates a new ImageId.
    #[inline]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying u64 value.
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Debug for ImageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImageId({})", self.0)
    }
}

impl fmt::Display for ImageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A unique identifier for an annotation in the dataset.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnnotationId(pub u64);

impl AnnotationId {
    /// Creates a new AnnotationId.
    #[inline]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying u64 value.
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Debug for AnnotationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AnnotationId({})", self.0)
    }
}

impl fmt::Display for AnnotationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A unique identifier for a category in the dataset.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CategoryId(pub u64);

impl CategoryId {
    /// Creates a new CategoryId.
    #[inline]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying u64 value.
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Debug for CategoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CategoryId({})", self.0)
    }
}

impl fmt::Display for CategoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_equality() {
        assert_eq!(ImageId(1), ImageId(1));
        assert_ne!(ImageId(1), ImageId(2));
    }

    #[test]
    fn test_id_ordering() {
        assert!(ImageId(1) < ImageId(2));
        assert!(CategoryId(10) > CategoryId(5));
    }

    #[test]
    fn test_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(AnnotationId(1));
        set.insert(AnnotationId(2));
        set.insert(AnnotationId(1)); // duplicate
        assert_eq!(set.len(), 2);
    }
}
