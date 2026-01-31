//! Typed coordinate values using PhantomData for compile-time safety.

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// A 2D coordinate with a type-level marker for the coordinate space.
///
/// The `TSpace` parameter should be either [`Pixel`](super::Pixel) or
/// [`Normalized`](super::Normalized), ensuring that coordinates from
/// different spaces cannot be accidentally mixed.
#[derive(Clone, Copy, PartialEq)]
pub struct Coord<TSpace> {
    pub x: f64,
    pub y: f64,
    _space: PhantomData<TSpace>,
}

impl<TSpace> Coord<TSpace> {
    /// Creates a new coordinate with the given x and y values.
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            _space: PhantomData,
        }
    }

    /// Returns true if both coordinates are finite (not NaN or infinite).
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

impl<TSpace> std::fmt::Debug for Coord<TSpace> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Coord")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

impl<TSpace> Default for Coord<TSpace> {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

// Custom serde implementation to avoid TSpace: Serialize/Deserialize bounds
impl<TSpace> Serialize for Coord<TSpace> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Coord", 2)?;
        state.serialize_field("x", &self.x)?;
        state.serialize_field("y", &self.y)?;
        state.end()
    }
}

impl<'de, TSpace> Deserialize<'de> for Coord<TSpace> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct CoordData {
            x: f64,
            y: f64,
        }
        let data = CoordData::deserialize(deserializer)?;
        Ok(Coord::new(data.x, data.y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Pixel;

    #[test]
    fn test_coord_creation() {
        let coord: Coord<Pixel> = Coord::new(10.0, 20.0);
        assert_eq!(coord.x, 10.0);
        assert_eq!(coord.y, 20.0);
    }

    #[test]
    fn test_coord_is_finite() {
        let finite: Coord<Pixel> = Coord::new(10.0, 20.0);
        assert!(finite.is_finite());

        let nan: Coord<Pixel> = Coord::new(f64::NAN, 20.0);
        assert!(!nan.is_finite());

        let inf: Coord<Pixel> = Coord::new(10.0, f64::INFINITY);
        assert!(!inf.is_finite());
    }
}
